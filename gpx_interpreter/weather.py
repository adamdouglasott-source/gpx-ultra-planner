# gpx_interpreter/weather.py
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd
import requests


@dataclass
class _WxResult:
    source: str
    lat: float
    lon: float
    data: pd.DataFrame


def _coerce_date(d: Optional[str]) -> _dt.date:
    """Accepts 'YYYY-MM-DD' or None; returns a date."""
    if not d:
        return _dt.date.today()
    try:
        return _dt.date.fromisoformat(str(d))
    except Exception:
        return _dt.date.today()


def _to_df_from_openmeteo(hourly: Dict[str, Any]) -> pd.DataFrame:
    """Build a tidy hourly DataFrame from Open-Meteo 'hourly' payload."""
    # We expect an ISO8601 time list and parallel arrays for each variable
    times = hourly.get("time") or []
    df = pd.DataFrame({"time": times})
    for key in ["temperature_2m", "precipitation_probability", "wind_speed_10m"]:
        if key in hourly:
            df[key] = hourly[key]
    # Ensure proper dtypes
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=False)
        for key in ["temperature_2m", "precipitation_probability", "wind_speed_10m"]:
            if key in df.columns:
                df[key] = pd.to_numeric(df[key], errors="coerce")
    return df


def _fetch_openmeteo_forecast(date_: _dt.date, lat: float, lon: float, tz: str) -> Optional[_WxResult]:
    """
    Forecast endpoint, valid for near-term dates. Open-Meteo typically serves forecast +/- ~16 days.
    """
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
        "timezone": tz or "auto",
        "start_date": date_.isoformat(),
        "end_date": date_.isoformat(),
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        hourly = (js or {}).get("hourly")
        if not hourly:
            return None
        df = _to_df_from_openmeteo(hourly)
        return _WxResult(source="open-meteo-forecast", lat=float(js.get("latitude", lat)), lon=float(js.get("longitude", lon)), data=df)
    except Exception:
        return None


def _fetch_openmeteo_archive(date_: _dt.date, lat: float, lon: float, tz: str) -> Optional[_WxResult]:
    """
    ERA5 reanalysis archive endpoint for historical dates.
    """
    base = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
        "timezone": tz or "auto",
        "start_date": date_.isoformat(),
        "end_date": date_.isoformat(),
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        hourly = (js or {}).get("hourly")
        if not hourly:
            return None
        df = _to_df_from_openmeteo(hourly)
        return _WxResult(source="open-meteo-era5", lat=float(js.get("latitude", lat)), lon=float(js.get("longitude", lon)), data=df)
    except Exception:
        return None


def get_weather_for_coords(
    event_date: Optional[str],
    lat: float,
    lon: float,
    timezone: Optional[str] = "auto",
) -> Dict[str, Any]:
    """
    Fetch hourly weather for the given date & coordinates.

    Returns:
        {
            "source": "open-meteo-forecast" | "open-meteo-era5" | "none",
            "lat": <float>,
            "lon": <float>,
            "data": <pd.DataFrame with columns: time, temperature_2m, precipitation_probability, wind_speed_10m>
        }
    """
    date_ = _coerce_date(event_date)
    tz = timezone or "auto"

    # Decide which API to try first: if within +/- 16 days of today, prefer forecast
    today = _dt.date.today()
    horizon = 16  # days
    try_forecast_first = abs((date_ - today).days) <= horizon

    result: Optional[_WxResult] = None
    if try_forecast_first:
        result = _fetch_openmeteo_forecast(date_, lat, lon, tz) or _fetch_openmeteo_archive(date_, lat, lon, tz)
    else:
        result = _fetch_openmeteo_archive(date_, lat, lon, tz) or _fetch_openmeteo_forecast(date_, lat, lon, tz)

    if result is None or result.data is None or result.data.empty:
        # Graceful empty response
        return {"source": "none", "lat": float(lat), "lon": float(lon), "data": pd.DataFrame(columns=["time", "temperature_2m", "precipitation_probability", "wind_speed_10m"])}

    # Ensure canonical column order
    df = result.data.copy()
    cols: List[str] = ["time"]
    for k in ["temperature_2m", "precipitation_probability", "wind_speed_10m"]:
        if k in df.columns:
            cols.append(k)
    df = df[cols]

    return {"source": result.source, "lat": float(lat), "lon": float(lon), "data": df}