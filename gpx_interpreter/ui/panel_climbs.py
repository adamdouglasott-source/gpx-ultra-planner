# gpx_interpreter/ui/panel_climbs.py
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

from .panel_loader import register, get_ns
from gpx_interpreter.plots import plot_elevation_with_major_climbs


def _unit_factor(units: str, kind: str) -> float:
    # Convert internal meters to display units and vice versa
    if kind == "dist":
        return 1609.344 if units == "miles" else 1000.0
    if kind == "elev":
        return 3.28084 if units == "miles" else 1.0
    return 1.0


@register("climbs")
def render_climbs(df: pd.DataFrame, *, units: str = "miles") -> Dict[str, Any] | None:
    """
    Elevation — Major climbs (tunable).
    Produces: elevation plot (with shaded, numbered climbs) and a climbs table.
    Returns a dict with the detected climbs list for downstream use if needed.
    """
    if df is None or df.empty:
        return None

    NS, K = get_ns("climbs")
    st.subheader("Elevation — Major climbs (tunable)")

    # --- Controls (namespaced keys) ---
    col_a, col_b, col_c, col_d = st.columns(4)
    smooth = col_a.number_input(
        "Elevation smoothing (points)",
        min_value=0,
        max_value=99,
        value=5,
        step=1,
        key=K("smooth"),
    )

    # Defaults requested earlier
    if units == "miles":
        default_min_gain_disp = 350.0  # feet
        default_min_len_disp = 1.0  # miles
        default_min_avg_grade = 0.005  # fraction
        default_dip_tol_disp = 50.0  # feet
        default_merge_gap_m = 20.0  # meters
        default_max_descent_disp = 100.0  # feet
        default_max_descent_dist_disp = 0.10  # miles
    else:
        default_min_gain_disp = 106.68  # meters
        default_min_len_disp = 1.60934  # km
        default_min_avg_grade = 0.005
        default_dip_tol_disp = 15.24  # meters
        default_merge_gap_m = 20.0  # meters
        default_max_descent_disp = 30.48  # meters
        default_max_descent_dist_disp = 0.160934  # km

    min_gain_display = col_b.number_input(
        f"Min gain ({'ft' if units=='miles' else 'm'})",
        min_value=0.0,
        value=default_min_gain_disp,
        step=25.0 if units == "miles" else 5.0,
        key=K("min_gain_display"),
    )
    min_len_display = col_c.number_input(
        f"Min length ({'mi' if units=='miles' else 'km'})",
        min_value=0.0,
        value=default_min_len_disp,
        step=0.10,
        key=K("min_len_display"),
    )
    min_avg_grade = col_d.number_input(
        "Min avg grade (fraction)",
        min_value=0.0,
        max_value=1.0,
        value=default_min_avg_grade,
        step=0.001,
        format="%.3f",
        key=K("min_avg_grade"),
    )

    col_e, col_f, col_g, col_h = st.columns(4)
    dip_tolerance_display = col_e.number_input(
        f"Allow dips within climb up to ({'ft' if units=='miles' else 'm'})",
        min_value=0.0,
        value=default_dip_tol_disp,
        step=5.0 if units == "miles" else 1.0,
        key=K("dip_tol_disp"),
    )
    merge_gap_m = col_f.number_input(
        "Merge if saddle descent ≤ (m)",
        min_value=0.0,
        value=default_merge_gap_m,
        step=5.0,
        key=K("merge_gap_m"),
    )
    max_descent_display = col_g.number_input(
        f"Max descent allowed ({'ft' if units=='miles' else 'm'})",
        min_value=0.0,
        value=default_max_descent_disp,
        step=10.0 if units == "miles" else 2.0,
        help="If you descend more than this while 'in a climb', the climb ends.",
        key=K("max_descent_disp"),
    )
    max_descent_dist_display = col_h.number_input(
        f"Max descent distance allowed ({'mi' if units=='miles' else 'km'})",
        min_value=0.0,
        value=default_max_descent_dist_disp,
        step=0.05,
        help="If you descend longer than this distance, the climb ends.",
        key=K("max_descent_dist_disp"),
    )

    # Convert display units → meters (internal)
    if units == "miles":
        min_gain_m = float(min_gain_display) / _unit_factor(units, "elev")  # ft → m
        min_len_m = float(min_len_display) * _unit_factor(units, "dist")  # mi → m
        dip_tol_m = float(dip_tolerance_display) / _unit_factor(units, "elev")
        max_descent_m = float(max_descent_display) / _unit_factor(units, "elev")
        max_descent_distance_m = float(max_descent_dist_display) * _unit_factor(
            units, "dist"
        )
    else:
        min_gain_m = float(min_gain_display)  # already meters
        min_len_m = float(min_len_display) * _unit_factor(units, "dist")  # km → m
        dip_tol_m = float(dip_tolerance_display)
        max_descent_m = float(max_descent_display)
        max_descent_distance_m = float(max_descent_dist_display) * _unit_factor(
            units, "dist"
        )

    # Fixed grade color bins
    color_bins = [
        (0, 5, "#fee8c8"),
        (5, 10, "#fdbb84"),
        (10, 15, "#fc8d59"),
        (15, 20, "#e34a33"),
        (20, 99, "#b30000"),
    ]

    # --- Backward-compatible call into plotter: try smooth_points, else fallback ---
    try:
        major: List[Dict[str, Any]] = plot_elevation_with_major_climbs(
            df,
            units=units,
            min_gain_m=min_gain_m,
            min_length_m=min_len_m,
            min_avg_grade=float(min_avg_grade),
            dip_tolerance_m=dip_tol_m,
            merge_if_descent_gap_m=float(merge_gap_m),
            max_descent_m=max_descent_m,
            max_descent_distance_m=max_descent_distance_m,
            color_bins=color_bins,
            smooth_points=int(smooth),  # may not be supported in older versions
        )
    except TypeError:
        # Older signature: call without smooth_points
        major = plot_elevation_with_major_climbs(
            df,
            units=units,
            min_gain_m=min_gain_m,
            min_length_m=min_len_m,
            min_avg_grade=float(min_avg_grade),
            dip_tolerance_m=dip_tol_m,
            merge_if_descent_gap_m=float(merge_gap_m),
            max_descent_m=max_descent_m,
            max_descent_distance_m=max_descent_distance_m,
            color_bins=color_bins,
        )
    except Exception as e:
        st.error(f"Elevation plot failed: {e}")
        major = []

    # Render figure
    try:
        fig = plt.gcf()
        fig.set_size_inches(10, 3)
        fig.set_dpi(110)
        st.pyplot(fig, width="stretch", clear_figure=True)
        plt.close(fig)
    except Exception:
        pass

    # --- Climbs table ---
    rows: List[Dict[str, Any]] = []
    dist_div = _unit_factor(units, "dist")
    elev_mul = _unit_factor(units, "elev")
    dist_label = "mi" if units == "miles" else "km"
    elev_label = "ft" if units == "miles" else "m"

    for idx, c in enumerate(major, start=1):
        try:
            start_u = float(c.get("start_dist_m", 0.0)) / dist_div
            end_u = float(c.get("end_dist_m", 0.0)) / dist_div
            length_u = float(c.get("length_m", 0.0)) / dist_div
            gain_disp = float(c.get("gain_m", 0.0)) * elev_mul
            avg_grade = float(c.get("avg_grade", 0.0)) * 100.0
            rows.append(
                {
                    "climb": idx,
                    f"start ({dist_label})": f"{start_u:.2f}",
                    f"end ({dist_label})": f"{end_u:.2f}",
                    f"distance ({dist_label})": f"{length_u:.2f}",
                    f"gain ({elev_label})": f"{gain_disp:.0f}",
                    "avg grade (%)": f"{avg_grade:.1f}",
                }
            )
        except Exception:
            continue

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    else:
        st.info("No climbs matched the current criteria — try loosening thresholds.")

    return {"climbs": major}
