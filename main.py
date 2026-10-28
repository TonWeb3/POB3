#!/usr/bin/env python
"""Main standalone entrypoint for TOG-main Polymarket Backtester & Strategy Optimizer.

Run via:
    python Scripts/main.py
or:
    cd Scripts && python main.py
"""

import os
import sys
import json
from pathlib import Path

# Add current folder to sys.path so it works standalone
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fetch_data import load_config, prepare_dataset
from backtest import (
    run_simulation,
    export_results,
    run_optimization_suite,
    export_optimization_report
)


def print_banner(optimization_mode: bool, drawdown_mode: bool = False, binance_futures: bool = False):
    feed_label = "BINANCE FUTURES" if binance_futures else "BINANCE SPOT"
    if optimization_mode and drawdown_mode:
        mode_text = f"DUAL MODE: DRAWDOWN SUITE + OPTIMIZATION [{feed_label}]"
    elif optimization_mode:
        mode_text = f"OPTIMIZATION MODE (15m Structural Optimization) [{feed_label}]"
    elif drawdown_mode:
        mode_text = f"DRAWDOWN SUITE MODE (Testing All Drawdown Filters) [{feed_label}]"
    else:
        mode_text = f"SINGLE BACKTEST MODE (Original Strategy Baseline) [{feed_label}]"
    banner = f"""
================================================================================
          TOG-MAIN POLYMARKET BTC 15M QUANTITATIVE ENGINE
                      [{mode_text}]
================================================================================
    """
    print(banner)



def display_single_results(results: dict):
    if "error" in results:
        print(f"\n[!] BACKTEST ERROR: {results['error']}")
        return

    if results.get("mode") == "drawdown_suite":
        leaderboard = results.get("leaderboard", [])
        best = results.get("best_strategy", {})
        use_vetoes = results.get("use_vetoes", True)

        print("\n" + "=" * 95)
        print(f"            DRAWDOWN STRATEGIES LEADERBOARD (Micro-Vetoes: {'ON' if use_vetoes else 'OFF'})")
        print("=" * 95)
        print(f"{'Rank':<5} {'Strategy Name':<45} {'Win Rate':<10} {'Trades':<8} {'Net PnL ($)':<14} {'Profit Factor':<14}")
        print("-" * 95)
        for rank, s in enumerate(leaderboard, 1):
            print(f"#{rank:<4} {s['core_name'][:43]:<45} {s['win_rate']:<5.1f}%    {s['trades']:<8} ${s['net_pnl']:+,.2f}        {s['profit_factor']:.2f}")
        print("=" * 95)

        print("\n" + "*" * 95)
        print(f"  #1 BEST DRAWDOWN STRATEGY SPOTLIGHT: {best.get('core_name')}")
        print("*" * 95)
        print(f"  Category       : {best.get('category')}")
        print(f"  Description    : {best.get('description')}")
        print(f"  Win Rate       : {best.get('win_rate'):.2f}% ({best.get('winning_trades')} Wins / {best.get('losing_trades')} Losses)")
        print(f"  Net Profit     : ${best.get('net_pnl'):+,.2f} ({best.get('total_return_pct'):+.2f}%)")
        print(f"  Profit Factor  : {best.get('profit_factor'):.2f}")
        print(f"  Executed Trades: {best.get('trades'):,} ({best.get('trade_frequency_pct'):.1f}% market coverage)")
        print(f"  Max Drawdown   : -${best.get('max_drawdown'):,.2f} (-{best.get('max_drawdown_pct'):.2f}%)")
        print(f"  Avg Win / Loss : +${best.get('avg_win'):.2f} / ${best.get('avg_loss'):.2f} (Ratio: {best.get('win_loss_ratio'):.2f})")
        print(f"  Annual Sharpe  : {best.get('sharpe_ratio'):.2f}")
        print("-" * 95)
        print("  DIRECTIONAL PERFORMANCE:")
        print(f"    UP   Trades : {best.get('up_trades', 0):5d} | Win Rate: {best.get('up_win_rate', 0.0):5.1f}% | PnL: ${best.get('up_pnl', 0.0):+10.2f}")
        print(f"    DOWN Trades : {best.get('down_trades', 0):5d} | Win Rate: {best.get('down_win_rate', 0.0):5.1f}% | PnL: ${best.get('down_pnl', 0.0):+10.2f}")
        print("*" * 95)
        return

    s = results.get("summary", {})
    bs = results.get("breakdown_side", {})

    print("\n" + "=" * 80)
    print("                           FULL BACKTEST RESULTS")
    print("=" * 80)
    print(f" Strategy Mode        : {s.get('mode_label')}")
    print(f" Initial Capital      : ${s.get('initial_balance'):,.2f}")
    print(f" Final Capital        : ${s.get('final_balance'):,.2f}")
    print(f" Net Profit           : ${s.get('net_profit'):+,.2f} ({s.get('total_return_pct'):+.2f}%)")
    print(f" Peak Capital         : ${s.get('peak_balance'):,.2f}")
    print(f" Max Drawdown         : -${s.get('max_drawdown'):,.2f} (-{s.get('max_drawdown_pct'):.2f}%)")
    print("-" * 80)
    print(f" Total Markets        : {s.get('markets_evaluated'):,}")
    print(f" Executed Trades      : {s.get('total_trades'):,} ({s.get('trade_frequency_pct')}% market participation)")
    if s.get("flips_count", 0) > 0:
        print(f" Position Flips       : {s.get('flips_count', 0):,}")
    print(f" Vetoed Signals       : {s.get('veto_count'):,}")
    print(f" Win Rate             : {s.get('win_rate')}% ({s.get('winning_trades')} Wins / {s.get('losing_trades')} Losses)")
    print(f" Profit Factor        : {s.get('profit_factor')}")
    print(f" Avg Win / Avg Loss   : +${s.get('avg_win'):.2f} / ${s.get('avg_loss'):.2f} (Ratio: {s.get('win_loss_ratio')})")
    print(f" Total Taker Fees     : ${s.get('total_fees_paid'):.2f}")
    print(f" Annualized Sharpe    : {s.get('sharpe_ratio')}")
    print("-" * 80)
    print(" DIRECTIONAL PERFORMANCE:")
    print(f"   UP   Trades : {bs.get('UP', {}).get('trades'):5d} | Win Rate: {bs.get('UP', {}).get('win_rate'):5.1f}% | PnL: ${bs.get('UP', {}).get('pnl'):+10.2f}")
    print(f"   DOWN Trades : {bs.get('DOWN', {}).get('trades'):5d} | Win Rate: {bs.get('DOWN', {}).get('win_rate'):5.1f}% | PnL: ${bs.get('DOWN', {}).get('pnl'):+10.2f}")
    print("=" * 80)


def display_optimization_summary(suite_results: dict):
    leaderboard = suite_results.get("leaderboard", [])
    best = suite_results.get("best_strategy", {})
    reg = suite_results.get("regime_diagnostics", {})

    print("\n" + "=" * 90)
    print("                       TOP 10 OPTIMIZATION LEADERBOARD")
    print("=" * 90)
    print(f"{'Rank':<5} {'Strategy Name':<42} {'Vetoes':<8} {'Win Rate':<10} {'Trades':<8} {'Net PnL ($)':<12}")
    print("-" * 90)

    for rank, s in enumerate(leaderboard[:10], 1):
        veto_str = "ON" if s["use_vetoes"] else "OFF"
        print(f"#{rank:<4} {s['name']:<42} {veto_str:<8} {s['win_rate']:<5.1f}%    {s['trades']:<8} ${s['net_pnl']:+,.2f}")

    print("=" * 90)
    print("\n" + "*" * 90)
    print(f"  #1 BEST PERFORMING STRATEGY SPOTLIGHT: {best.get('name')}")
    print("*" * 90)
    print(f"  Category       : {best.get('category')}")
    print(f"  Win Rate       : {best.get('win_rate'):.2f}%")
    print(f"  Net Profit     : ${best.get('net_pnl'):+,.2f} ({best.get('total_return_pct'):+.2f}%)")
    print(f"  Profit Factor  : {best.get('profit_factor'):.2f}")
    print(f"  Executed Trades: {best.get('trades'):,} ({best.get('trade_frequency_pct'):.1f}% of markets)")
    print(f"  Max Drawdown   : -${best.get('max_drawdown'):,.2f} (-{best.get('max_drawdown_pct'):.2f}%)")
    print(f"  Avg Win / Loss : +${best.get('avg_win'):.2f} / ${best.get('avg_loss'):.2f} (Ratio: {best.get('win_loss_ratio'):.2f})")
    print("-" * 90)
    print("  Recommended config settings to deploy this strategy:")
    print("  " + json.dumps({
        "strategy": {
            "ev_threshold": float(best.get("params", {}).get("ev_threshold", 0.04)),
            "min_prob": float(best.get("params", {}).get("min_prob", 0.55)),
            "min_strike_distance_bp": float(best.get("params", {}).get("min_strike_distance_bp", 0.0)),
            "min_entry_second": int(best.get("params", {}).get("min_entry_second", 0)),
            "max_entry_second": int(best.get("params", {}).get("max_entry_second", 880)),
            "use_15m_ha": bool(best.get("params", {}).get("use_15m_ha", False)),
            "circuit_breaker_losses": int(best.get("params", {}).get("circuit_breaker_losses", 0)),
            "pause_markets": int(best.get("params", {}).get("pause_markets", 0)),
            "max_1h_vol_bp": float(best.get("params", {}).get("max_1h_vol_bp", 0.0)),
            "prior_body_max_bp": float(best.get("params", {}).get("prior_body_max_bp", 12.0)),
            "skip_hours": list(best.get("params", {}).get("skip_hours", [])),
            "skip_days": list(best.get("params", {}).get("skip_days", []))
        },
        "use_vetoes": best.get("use_vetoes", False)
    }, indent=4).replace("\n", "\n  "))
    print("*" * 90)


def main():
    cfg_path = CURRENT_DIR / "config.json"
    if not cfg_path.exists():
        print(f"[!] config.json not found at {cfg_path}")
        sys.exit(1)

    cfg = load_config(str(cfg_path))
    optimization_mode = bool(cfg.get("optimization_mode", False))
    drawdown_mode = bool(cfg.get("drawdown_mode", False))
    binance_futures = bool(cfg.get("binance_futures", False))
    print_banner(optimization_mode, drawdown_mode, binance_futures)


    output_dir = CURRENT_DIR / cfg.get("output_dir", "results")

    # Step 1: Prepare/Fetch Data
    try:
        data_file = prepare_dataset(cfg)
    except Exception as e:
        print(f"\n[!] Failed to prepare data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: If drawdown_mode is enabled, run the Drawdown Suite on the Full EV model
    if drawdown_mode:
        if optimization_mode:
            print("\n" + "=" * 90)
            print("  PHASE 1: DRAWDOWN SUITE (Evaluating All Drawdown Filters on Original Full EV)")
            print("=" * 90)

        try:
            dd_results = run_simulation(data_file, cfg)
            export_results(dd_results, str(output_dir))
            display_single_results(dd_results)

            print(f"\n[OK] Drawdown Suite files exported to: {output_dir}")
            print(f"     - Drawdown Comprehensive Report : {output_dir / 'drawdown_report.md'}")
            print(f"     - Summary Backtest Report       : {output_dir / 'summary_report.md'}")
            print(f"     - Ranked Leaderboard (CSV)      : {output_dir / 'drawdown_leaderboard.csv'}")
            print(f"     - Champion Strategy Trades      : {output_dir / 'trades.csv'}\n")
        except Exception as e:
            print(f"\n[!] Drawdown suite simulation error: {e}")
            import traceback
            traceback.print_exc()
            if not optimization_mode:
                sys.exit(1)

    # Step 3: If optimization_mode is enabled, run the 15m Structural Optimization Suite
    if optimization_mode:
        if drawdown_mode:
            print("\n" + "=" * 90)
            print("  PHASE 2: OPTIMIZATION SUITE (15m Structural Architecture Progression & Matrix)")
            print("=" * 90)

        try:
            suite_results = run_optimization_suite(data_file, cfg)
            export_optimization_report(suite_results, str(output_dir))
            display_optimization_summary(suite_results)

            print(f"\n[OK] Optimization files exported to: {output_dir}")
            print(f"     - Unified Optimization Report : {output_dir / 'optimization_report.md'}")
            print(f"     - Ranked Leaderboard (CSV)    : {output_dir / 'optimization_leaderboard.csv'}")
            print(f"     - Champion Strategy Trades    : {output_dir / 'best_strategy_trades.csv'}\n")
        except Exception as e:
            print(f"\n[!] Optimization suite error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Step 4: If neither mode is enabled, run a single backtest of the unmitigated baseline
    if not drawdown_mode and not optimization_mode:
        try:
            results = run_simulation(data_file, cfg)
            export_results(results, str(output_dir))
            display_single_results(results)

            print(f"\n[OK] Full backtest files exported to: {output_dir}")
            print(f"     - Summary Report : {output_dir / 'summary_report.md'}")
            print(f"     - Detailed Trades: {output_dir / 'trades.csv'}\n")
        except Exception as e:
            print(f"\n[!] Simulation error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
