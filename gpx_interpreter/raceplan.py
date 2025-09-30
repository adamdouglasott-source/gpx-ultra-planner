# gpx_interpreter/raceplan.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date as _date, timedelta
from typing import Optional, List
import numpy as np
import pandas as pd


@dataclass
class PlanConfig:
    units: str = "miles"  # "miles" or "km"
    event_date: Optional[str] = None           # "YYYY-MM-DD"
    timezone: Optional[str] = "America/Los_Angeles"
    target_finish_hours: Optional[float] = None
    start_time_local: Optional[str] = None     # "HH:MM"
    fueling_g_per_hr: float = 90.0
    gel_size_g: float = 23.0
    split_len_units: float = 1.0               # display only; not required here
    pace_model: str = "gap"                    # "flat" | "gap" | "neg-split"


# -----------------------------
# Pace modeling helpers (simple, robust)
# -----------------------------

def _gap_effort_for_grade(g: float) -> float:
    """Crude effort multiplier by absolute grade (fraction).
    Tuned to be gentle; you already have a stronger model elsewhere for predictor."""
    a = abs(float(g))
    # buckets in fraction grade, symmetrical for up/down
    if a < 0.05:
        return 1.00
    if a < 0.08:
        return 1.03
    if a < 0.12:
        return 1.06
    if a < 0.18:
        return 1.10
    if a < 0.25:
        return 1.18
    return 1.28


def _synthesize_delta_time(df: pd.DataFrame, target_finish_hours: Optional[float], units: str) -> pd.Series:
    """Make plausible per-segment times when GPX has no timestamps."""
    dd = df.get("delta_dist_m", pd.Series([0.0] * len(df), index=df.index)).to_numpy(dtype=float)
    # baseline flat speed (conservative jogging / power-hike baseline)
    if units == "miles":
        baseline_mps = 1609.344 / (12 * 60)           # ~12:00 / mi
    else:
        baseline_mps = 1000.0 / (7 * 60 + 27)         # ~7:27 / km
    base_dt = np.divide(dd, np.maximum(baseline_mps, 1e-6))

    gr = df.get("grade", pd.Series(0.0, index=df.index)).fillna(0.0).to_numpy(dtype=float)
    eff = np.array([_gap_effort_for_grade(g) for g in gr])
    wdt = base_dt * eff
    total = wdt.sum()
    # scale to target finish if provided
    if target_finish_hours and target_finish_hours > 0 and total > 0:
        wdt = wdt * ((target_finish_hours * 3600.0) / total)
    # minimum per-point dt to avoid zeros
    return pd.Series(np.maximum(wdt, 0.1), index=df.index, dtype=float)


# -----------------------------
# Core: ETAs at waypoints
# -----------------------------

def _parse_start_dt(cfg: PlanConfig) -> Optional[datetime]:
    try:
        if not cfg.event_date or not cfg.start_time_local:
            return None
        yyyy, mm, dd = [int(x) for x in str(cfg.event_date).split("-")]
        hh, m = [int(x) for x in str(cfg.start_time_local).split(":")]
        # naive local time; Streamlit display will be fine with naive in most cases
        return datetime(yyyy, mm, dd, hh, m, 0)
    except Exception:
        return None


def compute_eta_table(df: pd.DataFrame, wdf: pd.DataFrame, cfg: PlanConfig) -> pd.DataFrame:
    """
    Return a table with ETA per waypoint name in wdf. Expects:
      df: track dataframe with columns lat, lon, cum_dist_m, delta_dist_m, delta_time_s (optional), grade (optional)
      wdf: waypoints with at least columns name, nearest_idx, dist_along_m
    """
    if df is None or df.empty or wdf is None or wdf.empty:
        return pd.DataFrame()

    # Segment times: use GPX if available; otherwise synthesize
    if "delta_time_s" in df.columns and float(pd.to_numeric(df["delta_time_s"], errors="coerce").fillna(0.0).sum()) > 0.0:
        dt = pd.to_numeric(df["delta_time_s"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        dt = _synthesize_delta_time(df, cfg.target_finish_hours, cfg.units).to_numpy(dtype=float)

    cum_t = np.cumsum(dt)
    # build result rows
    rows: List[dict] = []
    start_dt = _parse_start_dt(cfg)
    for _, r in wdf.iterrows():
        j = int(r.get("nearest_idx", 0))
        j = min(max(j, 0), len(df) - 1)
        t_s = float(cum_t[j])
        eta_clock = None
        if start_dt is not None:
            eta_clock = start_dt + timedelta(seconds=t_s)
        rows.append(
            dict(
                name=str(r.get("name", "")),
                category=str(r.get("category", "")),
                nearest_idx=j,
                dist_along_m=float(r.get("dist_along_m", float(df["cum_dist_m"].iloc[j]))),
                eta_s=t_s,
                eta_clock=eta_clock.isoformat(sep=" ") if isinstance(eta_clock, datetime) else None,
            )
        )
    out = pd.DataFrame(rows)
    # stable, by distance
    if "dist_along_m" in out.columns:
        out = out.sort_values("dist_along_m", kind="stable").reset_index(drop=True)
    return out


# -----------------------------
# Fueling between aid
# -----------------------------

def _fmt_hms(seconds: float) -> str:
    s = int(max(0, round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fueling_between_aid(wdf_eta: pd.DataFrame, cfg: PlanConfig) -> pd.DataFrame:
    """
    Compute fueling between aid stations (rows like: from -> to).
    Expects wdf_eta to have 'name', 'category', 'eta_s' (or eta_clock), 'dist_along_m'.
    """
    if wdf_eta is None or wdf_eta.empty:
        return pd.DataFrame()

    # consider rows where category contains 'aid' (aid or aid+dropbag)
    aid_rows = wdf_eta[wdf_eta["category"].astype(str).str.contains("aid", case=False, na=False)].copy()
    aid_rows = aid_rows.sort_values("dist_along_m", kind="stable").reset_index(drop=True)
    if len(aid_rows) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(aid_rows) - 1):
        a = aid_rows.iloc[i]
        b = aid_rows.iloc[i + 1]
        t_sec = float(b.get("eta_s", np.nan)) - float(a.get("eta_s", np.nan))
        if not np.isfinite(t_sec):
            continue
        carbs_g = (cfg.fueling_g_per_hr or 90.0) * (t_sec / 3600.0)
        gels = carbs_g / max(cfg.gel_size_g or 23.0, 1.0)
        rows.append(
            dict(
                _from=str(a["name"]),
                to=str(b["name"]),
                segment_time_hms=_fmt_hms(t_sec),
                segment_time_s=float(t_sec),
                carbs_g=round(float(carbs_g), 1),
                gels_equiv=round(float(gels), 1),
            )
        )
    return pd.DataFrame(rows)


# -----------------------------
# Weather ↔ ETA alignment
# -----------------------------

def align_weather_to_etas(wdf_eta: pd.DataFrame, wx_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each waypoint ETA, pick the nearest hourly weather row.
    Expects wx_df to have a 'time' column (datetime or ISO string) and columns like temperature_2m, etc.
    """
    if wdf_eta is None or wdf_eta.empty or wx_df is None or wx_df.empty:
        return pd.DataFrame()

    tmp_eta = wdf_eta.copy()
    tmp_eta["eta_clock"] = pd.to_datetime(tmp_eta["eta_clock"], errors="coerce", utc=False)

    tmp_wx = wx_df.copy()
    if "time" not in tmp_wx.columns:
        return pd.DataFrame()
    tmp_wx["time"] = pd.to_datetime(tmp_wx["time"], errors="coerce", utc=False)

    # nearest merge: for each eta, choose weather row with minimal |time - eta|
    # We'll do it by merging on the same date-hour first; if missing, fall back to nearest within +/- 6h.
    tmp_eta["eta_hour"] = tmp_eta["eta_clock"].dt.floor("h")
    tmp_wx["wx_hour"] = tmp_wx["time"].dt.floor("h")

    merged = pd.merge(tmp_eta, tmp_wx, left_on="eta_hour", right_on="wx_hour", how="left", suffixes=("", "_wx"))
    # For rows where we didn't get a match, fall back to nearest
    nohit = merged["time"].isna()
    if nohit.any():
        # Build a small nearest-time helper
        wx_times = tmp_wx["time"].to_numpy()
        def _nearest_time(t: pd.Timestamp) -> Optional[pd.Timestamp]:
            if pd.isna(t):
                return None
            i = int(np.argmin(np.abs((wx_times - t.to_datetime64()).astype("timedelta64[s]").astype(np.int64))))
            return pd.Timestamp(wx_times[i])

        fill_idx = merged.index[nohit]
        for idx in fill_idx:
            t = merged.at[idx, "eta_clock"]
            best = _nearest_time(t)
            if best is not None:
                row = tmp_wx.loc[tmp_wx["time"] == best].iloc[0]
                for col in tmp_wx.columns:
                    merged.at[idx, col] = row[col]

    keep_cols = ["name", "category", "dist_along_m", "eta_clock"]
    # keep a few common weather fields if present
    for c in ["time", "temperature_2m", "precipitation_probability", "wind_speed_10m"]:
        if c in merged.columns:
            keep_cols.append(c)

    out = merged[keep_cols].copy()
    return out