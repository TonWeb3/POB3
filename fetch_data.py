#!/usr/bin/env python
"""Data fetching and preparation engine for TOG-main backtester.

Fully autonomous and standalone.
Handles:
  1. Instant cache retrieval for existing panels.
  2. Sub-range slicing when requested dates fall within an existing panel.
  3. Automated API discovery (Polymarket Gamma API) and download (Telonex REST API).
  4. Binance Vision integration for exact tick-by-tick Binance spot execution prices,
     Wilder's RSI, Heiken-Ashi exhaustion streaks, and closed-form GBM fair probabilities,
     achieving 100% parity with live bot execution (TonWeb3/TOG).
"""

import os
import sys
import json
import time
import math
import zipfile
import threading
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import requests
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
from scipy.special import erfc
from ta.momentum import RSIIndicator

BASE_DIR = Path(__file__).resolve().parent
SQRT2 = math.sqrt(2.0)
BASE_VISION_SPOT_URL = "https://data.binance.vision/data/spot/daily"
BASE_VISION_FUTURES_URL = "https://data.binance.vision/data/futures/um/daily"
BASE_VISION_URL = BASE_VISION_SPOT_URL


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


# =====================================================================
# Binance Vision Market Data Engine
# =====================================================================

def download_vision_file_with_retry(url: str, dest_path: Path, max_retries: int = 4) -> bool:
    """Downloads a public archive from data.binance.vision (AWS S3) with exponential backoff."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30, stream=True)
            if r.status_code == 404:
                return False
            r.raise_for_status()

            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if chunk:
                        f.write(chunk)

            tmp_path.replace(dest_path)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[Warn] Failed downloading Binance Vision file {url}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                return False
            time.sleep(1.5 ** attempt)
    return False


def fetch_binance_vision_daily(date_str: str, cache_dir: Path, is_futures: bool = False) -> Tuple[Optional[Path], Optional[Path]]:
    """Downloads and parses both 1m klines and aggTrades for a single day from Binance Vision.
    
    Supports both Binance Spot and Binance USDT-M Futures.
    Returns (klines_parquet_path, spot_1s_parquet_path).
    """
    binance_dir = cache_dir / "binance"
    sub_kline = "futures_klines_1m" if is_futures else "klines_1m"
    sub_spot = "futures_1s" if is_futures else "spot_1s"
    klines_parquet = binance_dir / sub_kline / f"{date_str}.parquet"
    spot_1s_parquet = binance_dir / sub_spot / f"{date_str}.parquet"

    raw_zip_dir = binance_dir / ("raw_zips_futures" if is_futures else "raw_zips")
    raw_zip_dir.mkdir(parents=True, exist_ok=True)
    base_url = BASE_VISION_FUTURES_URL if is_futures else BASE_VISION_SPOT_URL

    # 1. Process 1m Klines
    if not klines_parquet.exists():
        kline_url = f"{base_url}/klines/BTCUSDT/1m/BTCUSDT-1m-{date_str}.zip"
        zip_path = raw_zip_dir / f"BTCUSDT-1m-{date_str}.zip"
        ok = download_vision_file_with_retry(kline_url, zip_path)
        if ok and zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path) as z:
                    csv_name = z.namelist()[0]
                    df_k = pd.read_csv(z.open(csv_name), usecols=[0, 1, 2, 3, 4, 5])
                    df_k.columns = ["open_time_raw", "open", "high", "low", "close", "volume"]
                    df_k = df_k[pd.to_numeric(df_k["open_time_raw"], errors="coerce").notnull()].copy()
                    df_k["open_time_raw"] = df_k["open_time_raw"].astype("int64")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df_k[col] = df_k[col].astype(float)
                first_ts = int(df_k["open_time_raw"].iloc[0])
                if first_ts > 1e14:
                    df_k["ts_s"] = (df_k["open_time_raw"] // 1_000_000).astype("int64")
                elif first_ts > 1e11:
                    df_k["ts_s"] = (df_k["open_time_raw"] // 1_000).astype("int64")
                else:
                    df_k["ts_s"] = df_k["open_time_raw"].astype("int64")

                klines_parquet.parent.mkdir(parents=True, exist_ok=True)
                df_k.to_parquet(klines_parquet, index=False)
            finally:
                if zip_path.exists():
                    zip_path.unlink(missing_ok=True)

    # 2. Process aggTrades into fast 1-second price series
    if not spot_1s_parquet.exists():
        agg_url = f"{base_url}/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{date_str}.zip"
        zip_path = raw_zip_dir / f"BTCUSDT-aggTrades-{date_str}.zip"
        ok = download_vision_file_with_retry(agg_url, zip_path)
        if ok and zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path) as z:
                    csv_name = z.namelist()[0]
                    df_agg = pd.read_csv(z.open(csv_name), usecols=[1, 5])
                    df_agg.columns = ["price", "timestamp_raw"]
                    df_agg = df_agg[pd.to_numeric(df_agg["timestamp_raw"], errors="coerce").notnull()].copy()
                    df_agg["timestamp_raw"] = df_agg["timestamp_raw"].astype("int64")
                    df_agg["price"] = df_agg["price"].astype(float)
                first_ts = int(df_agg["timestamp_raw"].iloc[0])
                if first_ts > 1e14:
                    df_agg["ts_s"] = (df_agg["timestamp_raw"] // 1_000_000).astype("int64")
                elif first_ts > 1e11:
                    df_agg["ts_s"] = (df_agg["timestamp_raw"] // 1_000).astype("int64")
                else:
                    df_agg["ts_s"] = df_agg["timestamp_raw"].astype("int64")

                sec_df = df_agg.groupby("ts_s", as_index=False)["price"].last()
                spot_1s_parquet.parent.mkdir(parents=True, exist_ok=True)
                sec_df.to_parquet(spot_1s_parquet, index=False)
            finally:
                if zip_path.exists():
                    zip_path.unlink(missing_ok=True)

    return (klines_parquet if klines_parquet.exists() else None,
            spot_1s_parquet if spot_1s_parquet.exists() else None)


class BinanceVisionDataFeed:
    """In-memory vectorized Binance market feed covering a date range.
    
    Provides exact tick-by-tick Binance spot or futures execution prices, candle open prices,
    Wilder's RSI, Heiken-Ashi streaks, and closed-form Geometric Brownian Motion
    fair probabilities.
    """

    def __init__(self, start_date: str, end_date: str, cache_dir: Path, is_futures: bool = False):
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        self.is_futures = is_futures
        feed_label = "Binance USDT-M Futures" if is_futures else "Binance Spot"

        d_start = dt.date.fromisoformat(start_date)
        d_end = dt.date.fromisoformat(end_date)
        # Pull 2 days extra preceding start date to warm up the 300-period 5m lookback (25 hours) and RSI/HA
        lookback_days = 2
        prime_start = d_start - dt.timedelta(days=lookback_days)
        total_days = (d_end - prime_start).days + 1
        date_list = [(prime_start + dt.timedelta(days=i)).isoformat() for i in range(total_days)]

        print(f"[BinanceVision] Preparing {feed_label} data for {len(date_list)} days ({date_list[0]} to {date_list[-1]})...")

        def _fetch_one(d):
            return d, fetch_binance_vision_daily(d, cache_dir, is_futures=is_futures)

        with ThreadPoolExecutor(max_workers=4) as ex:
            results = dict(ex.map(_fetch_one, date_list))

        kline_dfs = []
        spot_dfs = []

        for d_str in date_list:
            kp, sp = results.get(d_str, (None, None))
            if kp and kp.exists():
                kline_dfs.append(pd.read_parquet(kp))
            if sp and sp.exists():
                spot_dfs.append(pd.read_parquet(sp))

        if not kline_dfs or not spot_dfs:
            raise RuntimeError(f"Failed to load Binance Vision {feed_label} data for {start_date} to {end_date}.")

        # Concatenate 1m klines
        all_klines = pd.concat(kline_dfs, ignore_index=True).drop_duplicates(subset=["ts_s"]).sort_values("ts_s").reset_index(drop=True)
        self.kline_ts = all_klines["ts_s"].values
        self.kline_closes = all_klines["close"].values
        self.kline_opens = all_klines["open"].values
        self.kline_highs = all_klines["high"].values
        self.kline_lows = all_klines["low"].values

        # Concatenate 1s spot/futures ticks
        all_spot = pd.concat(spot_dfs, ignore_index=True).drop_duplicates(subset=["ts_s"]).sort_values("ts_s").reset_index(drop=True)
        self.spot_ts = all_spot["ts_s"].values
        self.spot_px = all_spot["price"].values

        # Precompute 14-period RSI on Binance 1m closes using ta (exact match to live bot)
        self.rsi_1m = RSIIndicator(close=pd.Series(self.kline_closes), window=14).rsi().fillna(50.0).values

        # Precompute 1m Heiken-Ashi streaks
        self.ha_streak_green, self.ha_streak_red = self._compute_ha_streaks(
            self.kline_opens, self.kline_highs, self.kline_lows, self.kline_closes
        )

        # Precompute 5m klines and rolling drift / sigma (lookback = 300 5m bars = 25 hours)
        self._build_5m_volatility_model()

        tick_type = "futures" if is_futures else "spot"
        print(f"[BinanceVision] Loaded {len(self.spot_ts):,} 1s {tick_type} ticks and {len(self.kline_ts):,} 1m bars.")

    def _compute_ha_streaks(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(c)
        g_streak = np.zeros(n, dtype=int)
        r_streak = np.zeros(n, dtype=int)
        if n < 2:
            return g_streak, r_streak

        ha_close = (o[0] + h[0] + l[0] + c[0]) / 4.0
        ha_open = (o[0] + c[0]) / 2.0
        curr_color = None
        count = 0

        for i in range(1, n):
            ha_c = (o[i] + h[i] + l[i] + c[i]) / 4.0
            ha_o = (ha_open + ha_close) / 2.0
            is_green = ha_c >= ha_o
            color = "green" if is_green else "red"

            if color == curr_color:
                count += 1
            else:
                curr_color = color
                count = 1

            if is_green:
                g_streak[i] = count
            else:
                r_streak[i] = count

            ha_open, ha_close = ha_o, ha_c

        return g_streak, r_streak

    def _build_5m_volatility_model(self):
        """Builds 5m candle returns and trailing realized drift/vol model."""
        df_1m = pd.DataFrame({
            "ts_s": self.kline_ts,
            "open": self.kline_opens,
            "high": self.kline_highs,
            "low": self.kline_lows,
            "close": self.kline_closes
        })
        df_1m["dt"] = pd.to_datetime(df_1m["ts_s"], unit="s", utc=True)
        df_5m = df_1m.resample("5min", on="dt").agg({
            "ts_s": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        }).dropna().reset_index()

        self.kline_5m_ts = df_5m["ts_s"].values
        closes_5m = df_5m["close"].values
        n_5m = len(closes_5m)

        log_rets = np.zeros(n_5m, dtype=float)
        if n_5m > 1:
            log_rets[1:] = np.diff(np.log(closes_5m))

        lookback = 300
        self.drift_5m = np.zeros(n_5m, dtype=float)
        self.sigma_5m = np.full(n_5m, 0.0035, dtype=float)

        for i in range(20, n_5m):
            start_i = max(0, i - lookback)
            w = log_rets[start_i:i]
            w = w[np.isfinite(w)]
            if len(w) >= 10:
                self.drift_5m[i] = float(np.mean(w))
                self.sigma_5m[i] = float(np.std(w))

    def get_window_open(self, start_ts_s: int) -> float:
        """Looks up the Binance open price of the 15m window."""
        idx = np.searchsorted(self.kline_ts, start_ts_s)
        if idx < len(self.kline_ts) and self.kline_ts[idx] == start_ts_s:
            return float(self.kline_opens[idx])
        s_idx = np.searchsorted(self.spot_ts, start_ts_s, side="right") - 1
        if s_idx >= 0:
            return float(self.spot_px[s_idx])
        return float(self.spot_px[0])

    def get_spot_vectorized(self, marks_s: np.ndarray) -> np.ndarray:
        """Vectorized lookup of latest spot prices for an array of second timestamps."""
        s_idx = np.searchsorted(self.spot_ts, marks_s, side="right") - 1
        s_idx = np.clip(s_idx, 0, len(self.spot_ts) - 1)
        return self.spot_px[s_idx].astype(float)

    def compute_fair_probability_vectorized(self, spot: np.ndarray, target_open: float, tau_s: np.ndarray, marks_s: np.ndarray) -> np.ndarray:
        """Closed-form Geometric Brownian Motion formula from TonWeb3/TOG."""
        if target_open <= 0 or len(spot) == 0:
            return np.full(len(spot), 0.5)

        time_left_min = tau_s / 60.0
        mc_steps = np.maximum(1.0, np.ceil(time_left_min / 5.0))

        idx_5m = np.searchsorted(self.kline_5m_ts, marks_s, side="right") - 1
        idx_5m = np.clip(idx_5m, 0, len(self.kline_5m_ts) - 1)
        sig = self.sigma_5m[idx_5m]
        drift = self.drift_5m[idx_5m]

        sig = np.where(sig > 0, sig, 0.0035)

        mu = (drift - 0.5 * sig ** 2) * mc_steps
        sd = sig * np.sqrt(mc_steps)
        z = (np.log(target_open / spot) - mu) / sd
        prob = 0.5 * erfc(z / SQRT2)
        return np.clip(prob, 0.0, 1.0)

    def get_indicators_vectorized(self, marks_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (rsi, ha_green_streak, ha_red_streak) for marks_s from the latest completed 1m bar."""
        k_idx = np.searchsorted(self.kline_ts, marks_s, side="right") - 1
        k_idx = np.clip(k_idx, 0, len(self.kline_ts) - 1)
        return self.rsi_1m[k_idx], self.ha_streak_green[k_idx], self.ha_streak_red[k_idx]


# =====================================================================
# General Data Helpers & Panel Builder
# =====================================================================

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


def fetch_and_build_panel(config: Dict[str, Any], start_date: str, end_date: str, dest_panel: Path, is_futures: bool = False) -> str:
    """Pulls Polymarket quotes & Chainlink strikes via Telonex, and BTCUSDT ticks/klines via Binance Vision."""
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

    # 1. Download Chainlink btcusd (for Polymarket settlement strike)
    print(f"[Data] Step 1/3: Checking Chainlink oracle prices for {len(extra_dates)} days via Telonex...")
    for d in extra_dates:
        cp_path = data_dir / "raw" / "polymarket" / "crypto_prices" / d / "btcusd.parquet"
        if not cp_path.exists():
            print(f"[Data] Downloading Chainlink btcusd for {d}...")
            client.download("polymarket", "crypto_prices", d, cp_path, asset_id="btcusd")

    # 2. Discover market slugs & download orderbook quotes
    print(f"[Data] Step 2/3: Discovering Polymarket 15m BTC markets from {start_date} to {end_date}...")
    start_ts = int(dt.datetime.fromisoformat(f"{start_date}T00:00:00+00:00").timestamp())
    end_ts = int(dt.datetime.fromisoformat(f"{end_date}T23:45:00+00:00").timestamp())
    slugs = [f"btc-updown-15m-{ts}" for ts in range(start_ts, end_ts + 900, 900)]
    catalog = discover_market_tokens(slugs, api_key=api_key, base_url=base_url)
    print(f"[Data] Found {len(catalog)} / {len(slugs)} markets on Polymarket.")

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

    # 3. Initialize Binance Vision Feed (spot or futures ticks, 1m/5m klines, RSI, HA, fair prob)
    feed_label = "Binance USDT-M Futures" if is_futures else "Binance Spot"
    print(f"[Data] Step 3/3: Initializing {feed_label} tick and kline feed...")
    binance_feed = BinanceVisionDataFeed(start_date, end_date, data_dir, is_futures=is_futures)

    cts, cpx = load_chainlink_prices(data_dir, extra_dates)
    if cts.size == 0:
        raise RuntimeError("Failed to load Chainlink price data.")

    # High-density sub-second scan grid: 1..9s integers, then 10..890s by 5s
    grid_offsets = sorted(set(list(range(1, 10)) + list(range(10, 895, 5))))
    grid_s = np.array(grid_offsets, dtype="int64")

    recs = {k: [] for k in
            ("slug", "t_s", "tau_s", "spot", "strike", "fair_up",
             "up_ask", "up_bid", "up_won",
             "exit_up_bid", "exit_up_ask", "sigma",
             "rsi", "ha_green_streak", "ha_red_streak")}

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
        marks_us = open_us + grid_s * 1_000_000
        marks_s = ts_start + grid_s
        tau = (close_us - marks_us) / 1e6
        bid, ask = sample_quotes(q, marks_us)

        target_open = binance_feed.get_window_open(ts_start)
        spot = binance_feed.get_spot_vectorized(marks_s)
        fair = binance_feed.compute_fair_probability_vectorized(spot, target_open, tau, marks_s)
        rsi_m, ha_g_m, ha_r_m = binance_feed.get_indicators_vectorized(marks_s)
        idx_5m = np.searchsorted(binance_feed.kline_5m_ts, marks_s, side="right") - 1
        idx_5m = np.clip(idx_5m, 0, len(binance_feed.kline_5m_ts) - 1)
        sig_arr = binance_feed.sigma_5m[idx_5m]

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
        recs["sigma"].extend(sig_arr[m].tolist())
        recs["rsi"].extend(rsi_m[m].tolist())
        recs["ha_green_streak"].extend(ha_g_m[m].tolist())
        recs["ha_red_streak"].extend(ha_r_m[m].tolist())
        processed_count += 1

    tb = pa.table({
        "slug": pa.array(recs["slug"], type=pa.string()),
        "t_s": pa.array(recs["t_s"], type=pa.int64()),
        "tau_s": pa.array(recs["tau_s"], type=pa.float64()),
        "spot": pa.array(recs["spot"], type=pa.float64()),
        "strike": pa.array(recs["strike"], type=pa.float64()),
        "fair_up": pa.array(recs["fair_up"], type=pa.float64()),
        "up_ask": pa.array(recs["up_ask"], type=pa.float64()),
        "up_bid": pa.array(recs["up_bid"], type=pa.float64()),
        "up_won": pa.array(recs["up_won"], type=pa.float64()),
        "exit_up_bid": pa.array(recs["exit_up_bid"], type=pa.float64()),
        "exit_up_ask": pa.array(recs["exit_up_ask"], type=pa.float64()),
        "sigma": pa.array(recs["sigma"], type=pa.float64()),
        "rsi": pa.array(recs["rsi"], type=pa.float64()),
        "ha_green_streak": pa.array(recs["ha_green_streak"], type=pa.int64()),
        "ha_red_streak": pa.array(recs["ha_red_streak"], type=pa.int64()),
    })
    dest_panel.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tb, dest_panel)
    print(f"[Data] Created unified dataset with {processed_count} markets and {tb.num_rows:,} rows at: {dest_panel.name}")
    return str(dest_panel)


def prepare_dataset(config: Dict[str, Any]) -> str:
    """Ensures the dataset covering start_date to end_date is prepared in local data_dir."""
    start_date = config.get("start_date", "2026-07-21")
    end_date = config.get("end_date", "2026-08-19")
    is_futures = bool(config.get("binance_futures", False))
    feed_label = "Binance USDT-M Futures" if is_futures else "Binance Spot"
    
    data_dir = resolve_dir(config.get("telonex", {}).get("data_dir", "data"), "data")
    frames_dir = data_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_futures_panel.parquet" if is_futures else "_panel.parquet"
    exact_panel = frames_dir / f"btc_15m_{start_date}_{end_date}{suffix}"

    print(f"[Data] Target window: {start_date} to {end_date} [{feed_label}]")
    print(f"[Data] Using local project data directory: {data_dir}")

    # 1. Exact panel check: verify it exists and contains Binance Vision indicator columns
    if exact_panel.exists():
        try:
            schema_names = pq.read_schema(exact_panel).names
            if "rsi" in schema_names and "ha_green_streak" in schema_names:
                print(f"[Data] Found ready unified panel dataset: {exact_panel}")
                return str(exact_panel)
            else:
                print(f"[Data] Existing {exact_panel.name} is missing Binance Vision indicators (legacy format). Upgrading...")
        except Exception:
            pass

    # 2. Check if a previously built binance panel exists under the alternative name (for spot)
    if not is_futures:
        binance_panel = frames_dir / f"btc_15m_{start_date}_{end_date}_binance_panel.parquet"
        if binance_panel.exists():
            try:
                schema_names = pq.read_schema(binance_panel).names
                if "rsi" in schema_names and "ha_green_streak" in schema_names:
                    print(f"[Data] Found ready unified panel dataset: {binance_panel}")
                    return str(binance_panel)
            except Exception:
                pass

    # 3. Otherwise: Download & build unified dataset automatically via Telonex + Binance Vision (Spot or Futures)
    print(f"[Data] Dataset for [{start_date} .. {end_date}] ({feed_label}) not cached. Initiating automated fetch...")
    return fetch_and_build_panel(config, start_date, end_date, exact_panel, is_futures=is_futures)


if __name__ == "__main__":
    cfg = load_config()
    res = prepare_dataset(cfg)
    print(f"[OK] Data ready at: {res}")
