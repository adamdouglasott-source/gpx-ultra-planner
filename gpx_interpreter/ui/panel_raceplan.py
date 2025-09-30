# gpx_interpreter/ui/panel_raceplan.py
from __future__ import annotations

import re
import pandas as pd
import streamlit as st

from .panel_loader import register, get_ns
from gpx_interpreter.raceplan import PlanConfig, compute_eta_table, fueling_between_aid


def _parse_hhmm_to_hours(txt: str | None) -> float | None:
    if not txt:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", txt.strip())
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) / 60.0


@register("raceplan")
def render_raceplan(
    df: pd.DataFrame, *, wdf: pd.DataFrame | None, units: str = "miles"
) -> dict | None:
    if df is None or df.empty:
        return None

    NS, K = get_ns("raceplan")
    st.subheader("Race Plan — ETAs & Fueling")

    if st.session_state.get("predictor_new_value", False):
        pred_h = float(st.session_state.get("predicted_finish_hours", 0.0) or 0.0)
        pred_hhmm = st.session_state.get("predicted_finish_hhmm", "")
        st.session_state[K("use_predicted")] = True
        st.session_state[K("target_hhmm")] = pred_hhmm
        st.session_state[K("target_hours")] = pred_h
        st.session_state["predictor_new_value"] = False

    c_top1, c_top2, c_top3 = st.columns(3)
    event_date = c_top1.date_input("Event date", key=K("event_date"))
    start_time = c_top2.text_input(
        "Start time (HH:MM local)", value="", key=K("start_time")
    )
    timezone = c_top3.text_input("Timezone", "America/Los_Angeles", key=K("tz"))

    has_pred = (st.session_state.get("predicted_finish_hours", 0.0) or 0.0) > 0.0
    use_pred = st.checkbox(
        "Use predicted finish from reference course",
        value=st.session_state.get(K("use_predicted"), has_pred),
        key=K("use_predicted"),
    )

    c4, c5 = st.columns(2)
    tf_hhmm = c4.text_input(
        "Target finish (HH:MM)",
        value=st.session_state.get(K("target_hhmm"), ""),
        key=K("target_hhmm"),
    )
    tf_hours = c5.number_input(
        "…or hours (decimal)",
        min_value=0.0,
        value=float(st.session_state.get(K("target_hours"), 0.0) or 0.0),
        step=0.25,
        key=K("target_hours"),
        disabled=bool(use_pred and has_pred),
    )

    if use_pred and has_pred:
        maybe_manual = _parse_hhmm_to_hours(tf_hhmm)
        target_finish_h = (
            maybe_manual
            if (maybe_manual and maybe_manual > 0)
            else float(st.session_state.get("predicted_finish_hours", 0.0) or 0.0)
        )
    else:
        target_finish_h = (
            _parse_hhmm_to_hours(tf_hhmm)
            if tf_hhmm
            else (tf_hours if tf_hours > 0 else None)
        )

    pace_model = st.selectbox(
        "Pace model", ["flat", "gap", "neg-split"], index=1, key=K("pace_model")
    )

    cfg = PlanConfig(
        units=units,
        event_date=str(event_date),
        timezone=timezone,
        target_finish_hours=(
            target_finish_h if (target_finish_h and target_finish_h > 0) else None
        ),
        start_time_local=start_time or None,
        fueling_g_per_hr=90.0,
        gel_size_g=23.0,
        split_len_units=1.0,
        pace_model=pace_model,
    )

    wdf = wdf if (wdf is not None) else pd.DataFrame()
    if wdf.empty:
        st.info("Add waypoints (Aid panel) to compute ETAs and fueling.")
        return {"wdf_eta": pd.DataFrame(), "fuel_df": pd.DataFrame(), "config": cfg}

    wdf_eta = compute_eta_table(df, wdf, cfg)
    if wdf_eta is None or wdf_eta.empty:
        st.info("No ETAs computed — check target finish and pace model.")
        return {"wdf_eta": pd.DataFrame(), "fuel_df": pd.DataFrame(), "config": cfg}

    st.markdown("**ETAs**")
    tmp = wdf_eta.copy()
    show_cols = [
        c for c in ["eta_clock", "name", "category", "dist_along_m"] if c in tmp.columns
    ]
    if "dist_along_m" in tmp.columns:
        tmp["dist"] = tmp["dist_along_m"] / (1609.344 if units == "miles" else 1000.0)
        show_cols = [
            c for c in ["eta_clock", "name", "category", "dist"] if c in tmp.columns
        ]
    st.dataframe(tmp[show_cols], width="stretch")

    fuel_res = fueling_between_aid(wdf_eta, cfg)
    fuel_df = (
        fuel_res if (fuel_res is not None and not fuel_res.empty) else pd.DataFrame()
    )
    if (
        not fuel_df.empty
        and ("to" in fuel_df.columns)
        and ("dropbag_allowed" in wdf.columns)
    ):
        m = dict(zip(wdf["name"], wdf["dropbag_allowed"]))
        fuel_df = fuel_df.copy()
        fuel_df["to_has_dropbag"] = fuel_df["to"].map(lambda x: bool(m.get(x, False)))

    if not fuel_df.empty:
        st.markdown("**Fueling between aid**")
        st.dataframe(fuel_df, width="stretch")

    return {"wdf_eta": wdf_eta, "fuel_df": fuel_df, "config": cfg}
