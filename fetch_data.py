#!/usr/bin/env python
"""Data fetching and preparation engine for TOG-main backtester.

Fully autonomous and standalone.
Handles:
  1. Instant cache retrieval for existing panels.
  2. Sub-range slicing when requested dates fall within an existing panel.
  3. Automated API discovery (Polymarket Gamma API) and download (Telonex REST API)
     for any new requested date range, compiling the exact panel on the fly.
"""

import os
import sys
import json
import time
import math
import threading
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import requests
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SQRT2 = math.sqrt(2.0)


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / SQRT2))


class RateLimiter:
    """Thread-safe token bucket rate limiter."""
    def __init__(self, rate_per_sec: float = 2.5, burst: float = 3.0):
        self.rate = float(rate_per_sec)
        self.capacity = float(burst)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0):
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait = deficit / self.rate
            time.sleep(min(wait, 2.0))


class TelonexClient:
    """Wrapper for Telonex REST API (api.telonex.io/v1)."""
    def __init__(self, api_key: str, base_url: str = "https://api.telonex.io/v1", timeout: int = 60):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def download(self, exchange: str, channel: str, date_str: str, dest_path: Path, **params) -> int:
        url = f"{self.base_url}/downloads/{exchange}/{channel}/{date_str}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest_path.with_suffix(dest_path.suffix + ".part")

        for attempt in range(5):
            try:
                with self.session.get(url, params=params, headers=self.headers(),
                                      timeout=self.timeout, stream=True, allow_redirects=True) as r:
                    if r.status_code == 429:
                        retry_after = float(r.headers.get("Retry-After", 2.0))
                        time.sleep(retry_after)
                        continue
                    if r.status_code == 401:
                        raise RuntimeError("Telonex 401 Unauthorized: Invalid API key.")
                    if r.status_code == 403:
                        raise RuntimeError("Telonex 403 Forbidden: Quota exceeded or subscription limit.")
                    if r.status_code == 404:
                        return 0
                    r.raise_for_status()

                    total = 0
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=512 * 1024):
                            if chunk:
                                f.write(chunk)
                                total += len(chunk)

                # Windows-safe atomic rename with retry
                for _ in range(5):
                    try:
                        tmp.replace(dest_path)
                        break
                    except PermissionError:
                        time.sleep(0.2)
                return total
            except Exception as e:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                if attempt == 4:
                    print(f"[Warn] Failed download for {dest_path.name}: {e}")
                    return 0
                time.sleep(2 ** attempt)
        return 0


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if not config_path:
        config_path = BASE_DIR / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dir(configured_path: str, default_name: str) -> Path:
    p = Path(configured_path) if configured_path else Path(default_name)
    if not p.is_absolute():
        p = BASE_DIR / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_chainlink_prices(data_dir: Path, dates: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    ts, px = [], []
    for d in sorted(dates):
        p = data_dir / "raw" / "polymarket" / "crypto_prices" / d / "btcusd.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p, columns=["timestamp_us", "price"])
        ts.append(t["timestamp_us"].to_numpy(zero_copy_only=False))
        px.append(np.asarray([float(v) for v in t["price"].to_pylist()]))
    if not ts:
        return np.array([], dtype="int64"), np.array([])
    ts = np.concatenate(ts).astype("int64")
    px = np.concatenate(px)
    o = np.argsort(ts)
    return ts[o], px[o]


def col_at(ts: np.ndarray, vals: np.ndarray, marks: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ts, marks, side="right") - 1
    out = np.full(len(marks), np.nan)
    ok = idx >= 0
    if ok.any():
        out[ok] = vals[idx[ok]]
    return out


def sample_quotes(quote_table, marks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    qts = quote_table["timestamp_us"].to_numpy(zero_copy_only=False).astype("int64")
    idx = np.searchsorted(qts, marks, side="right") - 1
    valid_mask = idx >= 0
    safe_idx = idx.copy()
    safe_idx[~valid_mask] = 0
    picks = pa.array(safe_idx)

    bids_raw = quote_table["bid_price"].take(picks).to_pylist()
    asks_raw = quote_table["ask_price"].take(picks).to_pylist()

    bids = np.full(len(marks), np.nan)
    asks = np.full(len(marks), np.nan)

    for i in range(len(marks)):
        if valid_mask[i]:
            try:
                bids[i] = float(bids_raw[i]) if bids_raw[i] not in (None, "") else np.nan
            except Exception:
                pass
            try:
                asks[i] = float(asks_raw[i]) if asks_raw[i] not in (None, "") else np.nan
            except Exception:
                pass
    return bids, asks


def discover_market_tokens(slugs: List[str], api_key: str = "", base_url: str = "https://api.telonex.io/v1") -> Dict[str, Any]:
    """Queries Polymarket Gamma API (and Telonex availability as fallback) to obtain market tokens for slugs."""
    catalog = {}
    auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def fetch_one(slug: str):
        # 1. Try Polymarket Gamma closed markets
        try:
            r = requests.get("https://gamma-api.polymarket.com/markets",
                             params={"slug": slug, "closed": "true"}, timeout=8)
            arr = r.json() if r.status_code == 200 else []
            # 2. Try Polymarket Gamma active markets
            if not arr:
                r = requests.get("https://gamma-api.polymarket.com/markets",
                                 params={"slug": slug}, timeout=8)
                arr = r.json() if r.status_code == 200 else []
            if arr:
                m = arr[0]
                tokens = json.loads(m.get("clobTokenIds", "[]")) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
                return slug, {
                    "market_id": m.get("id"),
                    "up_asset_id": tokens[0] if len(tokens) > 0 else None,
                    "down_asset_id": tokens[1] if len(tokens) > 1 else None
                }
        except Exception:
            pass

        # 3. Fallback to Telonex availability
        if api_key:
            try:
                r = requests.get(f"{base_url}/availability/polymarket",
                                 params={"slug": slug, "outcome": "Up"},
                                 headers=auth_headers, timeout=8)
                if r.status_code == 200:
                    aid = r.json().get("asset_id")
                    if aid:
                        return slug, {"market_id": None, "up_asset_id": aid, "down_asset_id": None}
            except Exception:
                pass

        return slug, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_one, slugs))
    for s, data in results:
        if data:
            catalog[s] = data
    return catalog


def fetch_and_build_panel(config: Dict[str, Any], start_date: str, end_date: str, dest_panel: Path) -> str:
    """Automatically pulls missing data from Polymarket/Telonex and builds the panel."""
    data_dir = resolve_dir(config.get("telonex", {}).get("data_dir", "data"), "data")
    api_key = config.get("telonex", {}).get("api_key")
    if not api_key:
        raise ValueError("Missing 'telonex.api_key' in config.json to download requested dates from Telonex API.")
    base_url = config.get("telonex", {}).get("base_url", "https://api.telonex.io/v1")
    client = TelonexClient(api_key=api_key, base_url=base_url)

    d_start = dt.date.fromisoformat(start_date)
    d_end = dt.date.fromisoformat(end_date)
    dates = [(d_start + dt.timedelta(days=i)).isoformat() for i in range((d_end - d_start).days + 1)]
    # Window settlement extends into the next day
    extra_dates = dates + [(d_end + dt.timedelta(days=1)).isoformat()]

    # 1. Download Chainlink btcusd
    print(f"[Data] Step 1/3: Checking Chainlink oracle prices for {len(extra_dates)} days...")
    for d in extra_dates:
        cp_path = data_dir / "raw" / "polymarket" / "crypto_prices" / d / "btcusd.parquet"
        if not cp_path.exists():
            print(f"[Data] Downloading Chainlink btcusd for {d}...")
            client.download("polymarket", "crypto_prices", d, cp_path, asset_id="btcusd")

    # 2. Discover market slugs
    print(f"[Data] Step 2/3: Discovering Polymarket 15m BTC markets from {start_date} to {end_date}...")
    start_ts = int(dt.datetime.fromisoformat(f"{start_date}T00:00:00+00:00").timestamp())
    end_ts = int(dt.datetime.fromisoformat(f"{end_date}T23:45:00+00:00").timestamp())
    slugs = [f"btc-updown-15m-{ts}" for ts in range(start_ts, end_ts + 900, 900)]
    catalog = discover_market_tokens(slugs, api_key=api_key, base_url=base_url)
    print(f"[Data] Found {len(catalog)} / {len(slugs)} markets on Polymarket.")

    # 3. Download quotes
    limiter = RateLimiter(rate_per_sec=2.5, burst=3.0)
    quote_tasks = []
    for slug, mdata in catalog.items():
        if not mdata.get("up_asset_id"):
            continue
        ts = int(slug.rsplit("-", 1)[1])
        wd = dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")
        qpath = data_dir / "raw" / "polymarket" / "quotes" / wd / f"{slug}__Up.parquet"
        quote_tasks.append((slug, mdata["up_asset_id"], wd, qpath))

    to_download = [t for t in quote_tasks if not t[3].exists()]
    if to_download:
        print(f"[Data] Downloading {len(to_download)} orderbook quote files from Telonex...")
        for i, (slug, aid, wd, qpath) in enumerate(to_download):
            limiter.acquire()
            client.download("polymarket", "quotes", wd, qpath, asset_id=aid)
            if (i + 1) % 20 == 0 or (i + 1) == len(to_download):
                print(f"[Data] Downloaded quotes: {i + 1}/{len(to_download)}")

    # 4. Build Panel Parquet
    print("[Data] Step 3/3: Building unified backtest panel parquet...")
    cts, cpx = load_chainlink_prices(data_dir, extra_dates)
    if cts.size == 0:
        raise RuntimeError("Failed to load Chainlink price data.")

    sigma_by_day = {}
    for d in dates:
        lo = np.searchsorted(cts, int(dt.datetime.fromisoformat(f"{d}T00:00:00+00:00").timestamp() * 1e6))
        hi = np.searchsorted(cts, int(dt.datetime.fromisoformat(f"{d}T00:00:00+00:00").timestamp() * 1e6) + 86400 * 1_000_000)
        seg = cpx[lo:hi]
        if seg.size > 100:
            r = np.diff(np.log(seg))
            r = r[np.isfinite(r)]
            sigma_by_day[d] = float(np.std(r))
    med_sigma = float(np.median(list(sigma_by_day.values()))) if sigma_by_day else 0.000045

    grid_offsets = sorted(set([1, 2, 3] + list(range(5, 895, 5))))
    grid_s = np.array(grid_offsets, dtype="int64")

    recs = {k: [] for k in
            ("slug", "t_s", "tau_s", "spot", "strike", "fair_up",
             "up_ask", "up_bid", "up_won",
             "exit_up_bid", "exit_up_ask", "sigma")}

    sorted_slugs = sorted(catalog.keys(), key=lambda s: int(s.rsplit("-", 1)[1]))
    processed_count = 0

    for slug in sorted_slugs:
        ts_start = int(slug.rsplit("-", 1)[1])
        wd = dt.datetime.fromtimestamp(ts_start, dt.timezone.utc).strftime("%Y-%m-%d")
        qpath = data_dir / "raw" / "polymarket" / "quotes" / wd / f"{slug}__Up.parquet"
        if not qpath.exists():
            continue

        q = pq.read_table(qpath, columns=["timestamp_us", "bid_price", "ask_price"])
        if q.num_rows < 5:
            continue

        open_us = int(ts_start * 1e6)
        close_us = int((ts_start + 900) * 1e6)

        strike = col_at(cts, cpx, np.array([open_us]))[0]
        settle = col_at(cts, cpx, np.array([close_us]))[0]
        if not np.isfinite(strike) or not np.isfinite(settle):
            continue

        uw = 1.0 if settle > strike else 0.0
        marks = open_us + grid_s * 1_000_000
        spot = col_at(cts, cpx, marks)
        bid, ask = sample_quotes(q, marks)

        sig = sigma_by_day.get(wd, med_sigma)
        tau = (close_us - marks) / 1e6

        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.log(spot / strike) / (sig * np.sqrt(tau))
        fair = norm_cdf(z)

        m = np.isfinite(spot) & np.isfinite(ask) & np.isfinite(bid) & np.isfinite(fair)
        if not m.any():
            continue

        recs["slug"].extend([slug] * int(m.sum()))
        recs["t_s"].extend(grid_s[m].tolist())
        recs["tau_s"].extend(tau[m].tolist())
        recs["spot"].extend(spot[m].tolist())
        recs["strike"].extend([strike] * int(m.sum()))
        recs["fair_up"].extend(fair[m].tolist())
        recs["up_ask"].extend(ask[m].tolist())
        recs["up_bid"].extend(bid[m].tolist())
        recs["up_won"].extend([uw] * int(m.sum()))
        recs["exit_up_bid"].extend(bid[m].tolist())
        recs["exit_up_ask"].extend(ask[m].tolist())
        recs["sigma"].extend([sig] * int(m.sum()))
        processed_count += 1

    tb = pa.table({k: pa.array(v) for k, v in recs.items()})
    dest_panel.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tb, dest_panel)
    print(f"[Data] Created dataset with {processed_count} markets and {tb.num_rows:,} rows.")
    return str(dest_panel)


def prepare_dataset(config: Dict[str, Any]) -> str:
    """Ensures the dataset covering start_date to end_date is prepared in local data_dir."""
    start_date = config.get("start_date", "2026-07-21")
    end_date = config.get("end_date", "2026-08-19")
    
    data_dir = resolve_dir(config.get("telonex", {}).get("data_dir", "data"), "data")
    frames_dir = data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Data] Target window: {start_date} to {end_date}")
    print(f"[Data] Using local project data directory: {data_dir}")

    # 1. Exact panel check
    exact_panel = frames_dir / f"btc_15m_{start_date}_{end_date}_panel.parquet"
    if exact_panel.exists():
        print(f"[Data] Found ready panel dataset: {exact_panel}")
        return str(exact_panel)

    # 2. Slice from local baseline panel if covered
    full_panel = frames_dir / "btc_15m_2026-07-21_2026-08-19_panel.parquet"
    if full_panel.exists():
        if start_date == "2026-07-21" and end_date == "2026-08-19":
            return str(full_panel)

        s_dt = dt.datetime.fromisoformat(f"{start_date}T00:00:00+00:00").timestamp()
        e_dt = dt.datetime.fromisoformat(f"{end_date}T23:59:59+00:00").timestamp()
        base_s_dt = dt.datetime.fromisoformat("2026-07-21T00:00:00+00:00").timestamp()
        base_e_dt = dt.datetime.fromisoformat("2026-08-19T23:59:59+00:00").timestamp()

        if s_dt >= base_s_dt and e_dt <= base_e_dt:
            print(f"[Data] Slicing sub-range [{start_date} .. {end_date}] from full baseline panel...")
            table = pq.read_table(full_panel)
            slugs = table["slug"].to_pylist()
            keep_mask = []
            for s in slugs:
                try:
                    ts = int(s.rsplit("-", 1)[1])
                    keep_mask.append(s_dt <= ts <= e_dt)
                except Exception:
                    keep_mask.append(False)

            keep_arr = np.array(keep_mask, dtype=bool)
            if keep_arr.any():
                sliced = table.filter(pa.array(keep_arr))
                pq.write_table(sliced, exact_panel)
                print(f"[Data] Sliced {sliced.num_rows:,} rows across {len(set(sliced['slug'].to_pylist()))} markets.")
                return str(exact_panel)

    # 3. Otherwise: Download & build dataset automatically via Telonex + Polymarket API
    print(f"[Data] Dataset for [{start_date} .. {end_date}] not cached. Initiating automated fetch...")
    return fetch_and_build_panel(config, start_date, end_date, exact_panel)


if __name__ == "__main__":
    cfg = load_config()
    res = prepare_dataset(cfg)
    print(f"[OK] Data ready at: {res}")
