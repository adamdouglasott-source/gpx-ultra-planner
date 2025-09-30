# gpx_interpreter/ui/panel_weather.py
from __future__ import annotations

from typing import Optional, Dict
import pandas as pd
import streamlit as st

from .panel_loader import register, get_ns
from gpx_interpreter.raceplan import align_weather_to_etas
from gpx_interpreter.weather import get_weather_for_coords


@register("weather")
def render_weather(
    df: pd.DataFrame,
    *,
    # flexible inputs — any/all may be omitted and will be inferred
    start_lat: Optional[float] = None,
    start_lon: Optional[float] = None,
    timezone: Optional[str] = None,
    event_date: Optional[str] = None,
    wdf_eta: Optional[pd.DataFrame] = None,
    temp_units: str = "°F",
    # NEW: allow passing the whole plan config (e.g., PlanConfig) and we’ll read from it
    plan_config: object = None,
) -> Dict[str, pd.DataFrame] | None:
    """
    Fetches weather for the event date/coords and (if ETAs are available)
    aligns hourly forecasts to waypoint ETAs.

    Accepts either explicit (start_lat, start_lon, timezone, event_date)
    or a plan_config providing .event_date and .timezone.
    If coords are omitted, tries to infer from df's first point.
    """
    if df is None or df.empty:
        return None

    # Derive timezone/event_date from plan_config when not explicitly provided
    if plan_config is not None:
        try:
            if event_date is None and getattr(plan_config, "event_date", None):
                event_date = str(plan_config.event_date)
        except Exception:
            pass
        try:
            if timezone is None and getattr(plan_config, "timezone", None):
                timezone = str(plan_config.timezone)
        except Exception:
            pass

    # Infer coordinates if not provided
    try:
        if (start_lat is None or start_lon is None) and {"lat", "lon"} <= set(
            df.columns
        ):
            start_lat = float(df["lat"].iloc[0])
            start_lon = float(df["lon"].iloc[0])
    except Exception:
        pass

    # Sensible fallbacks
    timezone = timezone or "America/Los_Angeles"
    if event_date is None:
        # if no event_date, nothing to fetch; return quietly
        return {"weather_df": pd.DataFrame(), "wx_at_etas": pd.DataFrame()}

    if start_lat is None or start_lon is None:
        # still no coords — cannot fetch weather
        return {"weather_df": pd.DataFrame(), "wx_at_etas": pd.DataFrame()}

    NS, K = get_ns("weather")
    st.subheader("Weather")

    try:
        wx = get_weather_for_coords(
            str(event_date), float(start_lat), float(start_lon), timezone or "auto"
        )
        src = wx.get("source")
        wdf_wx: pd.DataFrame = (
            wx.get("data")
            if isinstance(wx.get("data"), pd.DataFrame)
            else pd.DataFrame()
        )
        st.caption(
            f"Weather source: {src or 'none'} | coords: {wx.get('lat')}, {wx.get('lon')}"
        )

        out: Dict[str, pd.DataFrame] = {
            "weather_df": pd.DataFrame(),
            "wx_at_etas": pd.DataFrame(),
        }

        if not wdf_wx.empty:
            # If we have ETAs, align; else preview first 24h
            if wdf_eta is not None and not wdf_eta.empty:
                wx_at_etas = align_weather_to_etas(wdf_eta, wdf_wx)
                if (
                    not wx_at_etas.empty
                    and temp_units == "°F"
                    and "temperature_2m" in wx_at_etas.columns
                ):
                    wx_at_etas = wx_at_etas.copy()
                    wx_at_etas["temperature_2m"] = (
                        wx_at_etas["temperature_2m"] * 9.0 / 5.0 + 32.0
                    )
                st.dataframe(
                    wx_at_etas if not wx_at_etas.empty else wdf_wx.head(24),
                    width="stretch",
                )
                out["wx_at_etas"] = (
                    wx_at_etas if not wx_at_etas.empty else pd.DataFrame()
                )
            else:
                prev = wdf_wx.head(24).copy()
                if temp_units == "°F" and "temperature_2m" in prev.columns:
                    prev["temperature_2m"] = prev["temperature_2m"] * 9.0 / 5.0 + 32.0
                st.dataframe(prev, width="stretch")

            out["weather_df"] = wdf_wx
            return out

        else:
            st.info("No weather data available (offline or lookup failed).")
            return out

    except Exception as e:
        st.info(f"Weather lookup skipped: {e}")
        return {"weather_df": pd.DataFrame(), "wx_at_etas": pd.DataFrame()}
