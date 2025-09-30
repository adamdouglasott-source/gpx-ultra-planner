# gpx_interpreter/ui/panel_map.py
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import MeasureControl
from branca.colormap import LinearColormap

from .panel_loader import register, get_ns


def _u_dist(units: str) -> float:
    # meters → (mi|km)
    return 1609.344 if units == "miles" else 1000.0


@register("map")
def render_map(
    df: pd.DataFrame,
    *,
    start_lat: Optional[float] = None,
    start_lon: Optional[float] = None,
    wdf: Optional[pd.DataFrame] = None,
    wdf_eta: Optional[pd.DataFrame] = None,
    wx_at_etas: Optional[pd.DataFrame] = None,
    temp_units: str = "°F",
    units: str = "miles",
) -> Dict[str, Any] | None:
    """
    Route map with:
      - OSM / terrain / satellite layers + hillshade overlay
      - grade-colored polyline
      - clickable waypoint markers with ETA + weather
      - measure tool

    start_lat/lon are optional; if omitted, inferred from df.
    """
    if df is None or df.empty:
        return None

    # Infer coordinates when not provided
    try:
        if (start_lat is None or start_lon is None) and {"lat", "lon"} <= set(
            df.columns
        ):
            start_lat = float(df["lat"].iloc[0])
            start_lon = float(df["lon"].iloc[0])
    except Exception:
        pass

    if start_lat is None or start_lon is None:
        st.info("No coordinates available to center the map.")
        return None

    NS, K = get_ns("map")
    st.subheader("Map")

    # Base map + layers
    m = folium.Map(location=[start_lat, start_lon], zoom_start=12, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OSM (street)", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="© OpenTopoMap, © OpenStreetMap contributors",
        name="OpenTopoMap (terrain)",
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
        attr="Tiles by Stamen, CC BY 3.0 — Data © OSM",
        name="Stamen Terrain",
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Sources: Esri, Maxar, Earthstar Geographics, GIS User Community",
        name="Esri Satellite",
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Hillshade/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Sources: Esri, USGS, NOAA",
        name="Esri Hillshade (overlay)",
        control=True,
        overlay=True,
        opacity=0.5,
    ).add_to(m)

    # Grade-colored polyline
    try:
        if "grade" in df.columns:
            g = np.clip(
                df["grade"].to_numpy(dtype=float), -0.30, 0.30
            )  # clamp for color scale
            norm = (g + 0.30) / 0.60
            cmap = LinearColormap(
                colors=["#2c7fb8", "#cccccc", "#d7301f"], vmin=0.0, vmax=1.0
            )
            coords = df[["lat", "lon"]].dropna().to_numpy()
            if len(coords) > 1:
                for i in range(0, len(coords) - 1):
                    c0, c1 = coords[i], coords[i + 1]
                    col = cmap(float(norm[min(i, len(norm) - 1)]))
                    folium.PolyLine(
                        [[float(c0[0]), float(c0[1])], [float(c1[0]), float(c1[1])]],
                        weight=4,
                        opacity=0.85,
                        color=col,
                    ).add_to(m)
            cmap.caption = "Grade: downhill → flat → uphill"
            cmap.add_to(m)
    except Exception as e:
        st.warning(f"Polyline coloring skipped: {e}")

    # Build quick lookup dicts for ETAs and weather at ETAs
    eta_by_name: Dict[str, Dict[str, Any]] = {}
    if wdf_eta is not None and not wdf_eta.empty:
        for _, r in wdf_eta.iterrows():
            eta_by_name[str(r.get("name", ""))] = r.to_dict()

    wx_by_name: Dict[str, Dict[str, Any]] = {}
    if wx_at_etas is not None and not wx_at_etas.empty:
        for _, r in wx_at_etas.iterrows():
            wx_by_name[str(r.get("name", ""))] = r.to_dict()

    # Waypoint markers with ETA + weather
    try:
        if wdf is not None and not wdf.empty:
            for _, r in wdf.iterrows():
                nm = str(r.get("name", ""))
                cat = (r.get("category", "") or "").lower()
                color = (
                    "green"
                    if "aid+dropbag" in cat
                    else ("blue" if "aid" in cat else "gray")
                )

                eta_txt = ""
                if nm in eta_by_name and eta_by_name[nm].get("eta_clock"):
                    try:
                        from datetime import datetime as _dt

                        eta_txt = _dt.fromisoformat(
                            str(eta_by_name[nm]["eta_clock"])
                        ).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        eta_txt = str(eta_by_name[nm]["eta_clock"])

                wx_txt = ""
                if nm in wx_by_name:
                    w = wx_by_name[nm]
                    parts = []
                    # Temp
                    if (
                        "temperature_2m" in w
                        and w["temperature_2m"] is not None
                        and not pd.isna(w["temperature_2m"])
                    ):
                        val = float(w["temperature_2m"])
                        if temp_units == "°F":
                            val = val * 9.0 / 5.0 + 32.0
                        parts.append(f"Temp: {val:.1f}{temp_units}")
                    # Precip (NaN-safe)
                    if (
                        "precipitation_probability" in w
                        and w["precipitation_probability"] is not None
                        and not pd.isna(w["precipitation_probability"])
                    ):
                        try:
                            parts.append(
                                f"Precip: {int(float(w['precipitation_probability']))}%"
                            )
                        except Exception:
                            pass
                    # Wind
                    if (
                        "wind_speed_10m" in w
                        and w["wind_speed_10m"] is not None
                        and not pd.isna(w["wind_speed_10m"])
                    ):
                        parts.append(f"Wind: {w['wind_speed_10m']} m/s")
                    wx_txt = " | ".join(parts)

                # Distance display
                dist_txt = ""
                try:
                    d_m = float(r.get("dist_along_m", np.nan))
                    if not np.isnan(d_m):
                        d_units = d_m / _u_dist(units)
                        dist_txt = f"{d_units:.1f} {'mi' if units=='miles' else 'km'}"
                except Exception:
                    pass

                html = f"<b>{nm}</b><br>Type: {cat or 'other'}"
                if dist_txt:
                    html += f"<br>Dist: {dist_txt}"
                if eta_txt:
                    html += f"<br>ETA: {eta_txt}"
                if wx_txt:
                    html += f"<br>{wx_txt}"

                lat = float(r.get("lat", np.nan))
                lon = float(r.get("lon", np.nan))
                if not (np.isnan(lat) or np.isnan(lon)):
                    folium.Marker(
                        [lat, lon],
                        tooltip=f"{nm} [{cat or 'other'}]",
                        popup=folium.Popup(html, max_width=320),
                        icon=folium.Icon(color=color, icon="info-sign"),
                    ).add_to(m)
    except Exception as e:
        st.warning(f"Waypoint markers skipped: {e}")

    # Measure control + layer toggle
    primary_unit = "miles" if units == "miles" else "kilometers"
    m.add_child(
        MeasureControl(
            position="topleft",
            primary_length_unit=primary_unit,
            secondary_length_unit="meters",
            primary_area_unit="sqmeters",
            active_color="#d7301f",
            completed_color="#2c7fb8",
        )
    )
    folium.LayerControl(collapsed=False).add_to(m)

    from streamlit_folium import st_folium

    st_folium(m, width=900, height=520)

    return {"center": (start_lat, start_lon)}
