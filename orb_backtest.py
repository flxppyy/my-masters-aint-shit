import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List
import math
import json
import pytz
from datetime import time, timedelta

# -------------------- User Configuration --------------------

CONFIG = {
    "data_csv_path": None,  # e.g., "/mnt/data/MNQ_minute.csv"
    "symbol": "MNQ",
    "timezone_data": "UTC",  # timezone of your timestamps in CSV (e.g., "UTC")
    "timezone_exchange": "America/New_York",  # used to align to 09:30 cash open

    # Session definition (New York time)
    "cash_open": "09:30",          # HH:MM
    "session_end": "16:00",        # HH:MM  (force flat at this time)
    "opening_range_minutes": 15,   # length of the opening range window (5/10/15/30 typical)

    # Entry logic
    "enter_on_break_of_ORH_or_ORL": True,
    "one_shot_per_day": True,  # only first breakout of the day

    # Filters (any Falsey means "ignore this filter")
    "min_overnight_gap_pct": None, # e.g., 0.2 means require >= 0.2% gap from prior session close
    "max_overnight_gap_pct": None, # e.g., 1.0 means require <= 1.0% gap
    "overnight_range_to_atr_ratio_min": None,  # e.g., 0.5
    "overnight_range_to_atr_ratio_max": None,  # e.g., 2.0
    "firstN_volume_minutes": 15,    # compare first N minutes volume vs its rolling history
    "firstN_volume_pctile_min": None,  # e.g., 60 -> require today's firstN volume >= 60th pctile
    "firstN_volume_lookback_days": 60,

    # Exits (choose one or combine; template supports: stop at other side of OR, PT = k*OR, time exit)
    "stop_at_opposite_side": True,    # classic ORB stop
    "profit_target_multiple_of_OR": 1.0,  # e.g., 1.0 means PT = 1x opening range size; None to disable
    "hard_exit_at_session_end": True,

    # Sizing & Costs
    "contracts": 1,  # MNQ micro contracts
    "mnq_point_value": 2.0,  # $ per index point for MNQ
    "mnq_tick_size": 0.25,
    "commission_per_side": 1.20,  # assumed per contract per side
    "slippage_ticks_per_side": 0.25,  # assumed average slippage per side

    # Risk controls
    "max_trades_per_day": 1,  # 1 for classic ORB; increase if you allow re-entries
    "max_hold_minutes": None, # e.g., 240 -> close after 4 hours regardless
}

# -------------------- Helpers --------------------

def dollars_from_points(points: float, point_value: float, contracts: int) -> float:
    return points * point_value * contracts

def dollars_from_ticks(ticks: float, tick_value: float, contracts: int) -> float:
    return ticks * tick_value * contracts

def tick_value(point_value: float, tick_size: float) -> float:
    return point_value * tick_size

def _parse_hhmm(s: str) -> Tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)

def localize_to_exchange_tz(ts: pd.Series, tz_data: str, tz_exch: str) -> pd.Series:
    s = pd.to_datetime(ts, utc=(tz_data.upper()=="UTC"))
    if tz_data.upper() != "UTC":
        s = pd.to_datetime(ts).dt.tz_localize(tz_data)
    return s.dt.tz_convert(tz_exch)

def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # Daily ATR on sessionized OHLC
    high = df["high"].resample("1D").max()
    low = df["low"].resample("1D").min()
    close = df["close"].resample("1D").last().shift(1)  # prior close
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    return atr

@dataclass
class Trade:
    date: pd.Timestamp
    direction: str  # "long" or "short"
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_points: float
    pnl_dollars: float
    reason_exit: str

# -------------------- Core Backtest --------------------

def run_orb_backtest(df_min: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Prepare data
    tz_exch = cfg["timezone_exchange"]
    tz_data = cfg["timezone_data"]

    df = df_min.copy()
    # standardize columns
    needed_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    if not needed_cols.issubset(set([c.lower() for c in df.columns])):
        raise ValueError(f"CSV must contain columns: {needed_cols}")
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if tz_data.upper() == "UTC":
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp_exch"] = df["timestamp"].dt.tz_convert(tz_exch)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(tz_data).dt.tz_convert(tz_exch)
        df["timestamp_exch"] = df["timestamp"]

    df = df.set_index("timestamp_exch").sort_index()

    # Session windows
    open_h, open_m = _parse_hhmm(cfg["cash_open"])
    end_h, end_m = _parse_hhmm(cfg["session_end"])
    or_minutes = int(cfg["opening_range_minutes"])

    # Compute daily prior close, overnight gap, and ATR-based filters
    # Create per-day groups in exchange TZ
    df["date"] = df.index.tz_convert(tz_exch).date
    daily_close = df.between_time(time(0,0), time(23,59))["close"].groupby(df["date"]).last()
    prior_close = daily_close.shift(1)

    # Simple ATR on daily highs/lows from minute bars (approx)
    daily_high = df["high"].groupby(df["date"]).max()
    daily_low = df["low"].groupby(df["date"]).min()
    daily_prior_close = daily_close.shift(1)
    tr = pd.concat([
        daily_high - daily_low,
        (daily_high - daily_prior_close).abs(),
        (daily_low - daily_prior_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()

    trades: List[Trade] = []

    contracts = cfg["contracts"]
    point_val = cfg["mnq_point_value"]
    tick_sz = cfg["mnq_tick_size"]
    tick_val = point_val * tick_sz

    comm = cfg["commission_per_side"]
    slip_ticks = cfg["slippage_ticks_per_side"] or 0.0

    # Iterate days
    for d, day_df in df.groupby("date"):
        # Require prior data for filters
        if d not in daily_close.index or pd.isna(prior_close.loc[d]):
            continue

        # Session slice (open to end)
        day_slice = day_df.between_time(time(open_h, open_m), time(end_h, end_m))
        if day_slice.empty:
            continue

        # Define OR window
        or_end_time = (day_slice.index[0].to_pydatetime().replace(second=0, microsecond=0) +
                       timedelta(minutes=or_minutes))
        or_window = day_slice[day_slice.index < or_end_time]
        if or_window.empty:
            continue

        ORH = or_window["high"].max()
        ORL = or_window["low"].min()
        OR_range = ORH - ORL

        # Compute filters
        pc = prior_close.loc[d]
        first_price = or_window["open"].iloc[0]
        overnight_gap_pct = (first_price - pc) / pc * 100.0 if pc and pc != 0 else np.nan

        # Overnight range approximated from previous day's high/low
        if pd.isna(atr.loc[d]):
            atr_ratio = np.nan
        else:
            overnight_range = (daily_high.shift(1).loc[d] - daily_low.shift(1).loc[d]) if d in daily_high.index else np.nan
            atr_ratio = overnight_range / atr.loc[d] if (atr.loc[d] and not pd.isna(overnight_range)) else np.nan

        # First-N volume filter
        N = cfg["firstN_volume_minutes"]
        firstN_vol = day_slice.head(N)["volume"].sum()
        lookback_days = cfg["firstN_volume_lookback_days"]
        lb_idx = daily_close.index.tolist()
        try:
            idx_pos = lb_idx.index(d)
        except ValueError:
            idx_pos = None

        vol_pass = True
        if cfg["firstN_volume_pctile_min"]:
            if idx_pos and idx_pos > lookback_days:
                past_days = lb_idx[idx_pos - lookback_days:idx_pos]
                past_vols = []
                for d2 in past_days:
                    ds = df[df["date"] == d2].between_time(time(open_h, open_m), time(end_h, end_m))
                    if ds.empty: 
                        continue
                    past_vols.append(ds.head(N)["volume"].sum())
                if past_vols:
                    pctile = (np.array(past_vols) < firstN_vol).mean() * 100.0
                    vol_pass = pctile >= cfg["firstN_volume_pctile_min"]
                else:
                    vol_pass = True  # insufficient history -> don't block
            else:
                vol_pass = True  # insufficient history

        # Apply gap/ATR filters
        def _range_check(val, minv, maxv) -> bool:
            if minv is not None and (pd.isna(val) or val < minv): return False
            if maxv is not None and (pd.isna(val) or val > maxv): return False
            return True

        gap_ok = _range_check(overnight_gap_pct, cfg["min_overnight_gap_pct"], cfg["max_overnight_gap_pct"])
        atr_ok = _range_check(atr_ratio, cfg["overnight_range_to_atr_ratio_min"], cfg["overnight_range_to_atr_ratio_max"])

        if not (gap_ok and atr_ok and vol_pass):
            continue  # skip this day

        # Trading after OR window
        post_or = day_slice[day_slice.index >= or_end_time]
        if post_or.empty:
            continue

        trades_today = 0
        in_position = False
        long_active = False
        short_active = False

        entry_time = None
        entry_price = None
        direction = None

        for ts, row in post_or.iterrows():
            if cfg["max_trades_per_day"] is not None and trades_today >= cfg["max_trades_per_day"]:
                break

            high_t = row["high"]
            low_t = row["low"]

            # Entry logic (stop orders at ORH/ORL)
            if not in_position and cfg["enter_on_break_of_ORH_or_ORL"]:
                if high_t >= ORH:
                    # Long entry at ORH + slippage
                    entry_price = ORH + slip_ticks * tick_sz
                    entry_time = ts
                    direction = "long"
                    in_position = True
                    long_active = True
                elif low_t <= ORL:
                    # Short entry at ORL - slippage
                    entry_price = ORL - slip_ticks * tick_sz
                    entry_time = ts
                    direction = "short"
                    in_position = True
                    short_active = True

                if in_position and cfg["one_shot_per_day"]:
                    # lock out further entries (but we still must manage exit)
                    trades_today += 1

            if in_position:
                # Exit rules
                exit_now = False
                exit_reason = None
                exit_price = None

                # Classic stop at opposite side of OR
                if cfg["stop_at_opposite_side"]:
                    if direction == "long" and low_t <= ORL:
                        exit_price = ORL - slip_ticks * tick_sz
                        exit_now = True
                        exit_reason = "stop_OR_opposite"
                    elif direction == "short" and high_t >= ORH:
                        exit_price = ORH + slip_ticks * tick_sz
                        exit_now = True
                        exit_reason = "stop_OR_opposite"

                # Profit target at k*OR
                if (not exit_now) and cfg["profit_target_multiple_of_OR"]:
                    k = cfg["profit_target_multiple_of_OR"]
                    if direction == "long":
                        target = entry_price + k * OR_range
                        if high_t >= target:
                            exit_price = target - slip_ticks * tick_sz
                            exit_now = True
                            exit_reason = f"pt_{k:.2f}xOR"
                    else:
                        target = entry_price - k * OR_range
                        if low_t <= target:
                            exit_price = target + slip_ticks * tick_sz
                            exit_now = True
                            exit_reason = f"pt_{k:.2f}xOR"

                # Time-based exit: end of session or max_hold_minutes
                if (not exit_now) and cfg["hard_exit_at_session_end"]:
                    if ts.time() >= time(end_h, end_m):
                        exit_price = row["close"]
                        exit_now = True
                        exit_reason = "session_end"

                if (not exit_now) and cfg["max_hold_minutes"]:
                    if (ts - entry_time) >= timedelta(minutes=cfg["max_hold_minutes"]):
                        exit_price = row["close"]
                        exit_now = True
                        exit_reason = "max_hold"

                if exit_now:
                    # Compute PnL in points
                    if direction == "long":
                        pnl_points = exit_price - entry_price
                    else:
                        pnl_points = entry_price - exit_price

                    # Costs: commissions + slippage already baked into fills; add commissions
                    total_comm = 2 * comm * contracts  # in/out
                    pnl_dollars = dollars_from_points(pnl_points, point_val, contracts) - total_comm

                    trades.append(Trade(
                        date=pd.Timestamp(d).tz_localize(tz_exch),
                        direction=direction,
                        entry_time=entry_time,
                        entry_price=float(entry_price),
                        exit_time=ts,
                        exit_price=float(exit_price),
                        pnl_points=float(pnl_points),
                        pnl_dollars=float(pnl_dollars),
                        reason_exit=exit_reason
                    ))
                    # Reset
                    in_position = False
                    long_active = short_active = False
                    entry_time = None
                    entry_price = None
                    direction = None

    # Summaries
    trades_df = pd.DataFrame([asdict(t) for t in trades])
    if not trades_df.empty:
        # KPIs
        total_pnl = trades_df["pnl_dollars"].sum()
        avg_trade = trades_df["pnl_dollars"].mean()
        hit_rate = (trades_df["pnl_dollars"] > 0).mean() * 100.0
        longs = trades_df[trades_df["direction"] == "long"]
        shorts = trades_df[trades_df["direction"] == "short"]
        pf = trades_df.loc[trades_df["pnl_dollars"] > 0, "pnl_dollars"].sum() / max(
            1.0, -trades_df.loc[trades_df["pnl_dollars"] < 0, "pnl_dollars"].sum()
        )

        # Drawdown (on equity curve)
        equity = trades_df["pnl_dollars"].cumsum()
        peak = equity.cummax()
        dd = equity - peak
        max_dd = dd.min()

        kpis = {
            "symbol": cfg["symbol"],
            "n_trades": int(len(trades_df)),
            "total_pnl_$": float(total_pnl),
            "avg_trade_$": float(avg_trade),
            "hit_rate_%": float(hit_rate),
            "profit_factor": float(pf),
            "max_drawdown_$": float(max_dd),
            "long_trades": int(len(longs)),
            "short_trades": int(len(shorts)),
        }
    else:
        kpis = {
            "symbol": cfg["symbol"],
            "n_trades": 0,
            "total_pnl_$": 0.0,
            "avg_trade_$": 0.0,
            "hit_rate_%": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_$": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }

    return trades_df, kpis