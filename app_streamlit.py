# app_streamlit.py
from __future__ import annotations

import os
import tempfile
import streamlit as st

# Core logic
from gpx_interpreter.parser import load_gpx_to_dataframe
from gpx_interpreter.utils import fmt_hms

# Panel loader + panel modules that self-register on import
from gpx_interpreter.ui.panel_loader import get as panel
from gpx_interpreter.ui import panel_climbs  # registers "climbs"
from gpx_interpreter.ui import panel_aid  # registers "aid"
import gpx_interpreter.ui.panel_predictor  # noqa: F401  # registers "predictor"
from gpx_interpreter.ui import panel_raceplan  # registers "raceplan"
from gpx_interpreter.ui import panel_weather  # registers "weather"
from gpx_interpreter.ui import panel_map  # registers "map"
from gpx_interpreter.ui import panel_poles  # registers "poles"
import gpx_interpreter.ui.panel_export  # noqa: F401  # registers "export"

# Finish-time predictor helpers (used by the predictor panel below)
from gpx_interpreter.predictor import (
    PredictParams,
    predict_finish_time,
    extract_total_time_seconds_from_gpx_df,
)

st.set_page_config(page_title="GPX Ultra Planner — Core Panels", layout="wide")
st.title("🏃 GPX Ultra Planner")

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
units = st.sidebar.selectbox("Units", ["miles", "km"], index=0)

# ------------------------------------------------------------------------------
# Upload GPX course (target course)
# ------------------------------------------------------------------------------
gpx_file = st.file_uploader("Upload target GPX course", type=["gpx"], key="main_gpx")


def _save_tmp(uploaded, suffix: str = ".gpx") -> str | None:
    if uploaded is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name


# Persist across reruns
if "gpx_path" not in st.session_state:
    st.session_state["gpx_path"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None

if gpx_file is not None:
    st.session_state["gpx_path"] = _save_tmp(gpx_file, ".gpx")
    st.session_state["df"] = load_gpx_to_dataframe(st.session_state["gpx_path"])
    if st.session_state["df"] is not None and not st.session_state["df"].empty:
        st.success(f"Loaded {len(st.session_state['df'])} track points.")
    else:
        st.warning("GPX parsed but appears empty.")

gpx_path = st.session_state["gpx_path"]
df = st.session_state["df"]

# ------------------------------------------------------------------------------
# Panels: only render once a GPX is uploaded
# ------------------------------------------------------------------------------
if df is None or gpx_path is None or df.empty:
    st.info("Upload a target GPX to enable the panels.")
else:
    # 1) Elevation + Major Climbs
    climbs_out = panel("climbs")(df=df, units=units)

    # 2) Aid / Dropbag / Crew
    aid_out = panel("aid")(df=df, gpx_path=gpx_path, units=units)
    wdf = (aid_out or {}).get("wdf")

    # 3) Finish Time Predictor (self-contained here with unique keys)
    st.markdown("---")
    st.header("Finish Time Predictor")

    col_ref1, col_ref2 = st.columns([2, 1])
    with col_ref1:
        ref_file = st.file_uploader(
            "Reference GPX (a past race/effort with timestamps preferred)",
            type=["gpx"],
            key="pred_ref_gpx",
        )
    with col_ref2:
        ref_time_txt = st.text_input(
            "Reference finish (HH:MM or HH:MM:SS)",
            value="",
            key="pred_ref_time",
            help="If your reference GPX has timestamps, you can leave this blank.",
        )

    with st.expander("Advanced predictor settings", expanded=False):
        alpha = st.slider(
            "Blend α (physics ↔ Riegel)",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            key="pred_alpha",
        )
        k_riegel = st.slider(
            "Riegel exponent k",
            min_value=1.00,
            max_value=1.30,
            value=1.15,
            step=0.01,
            key="pred_k",
        )
        vert_sens = st.slider(
            "Vertical sensitivity (climb tax)",
            min_value=0.05,
            max_value=0.40,
            value=0.20,
            step=0.01,
            key="pred_vert",
        )
        use_alt = st.checkbox("Adjust for average altitude", value=True, key="pred_alt")
        use_tech = st.checkbox(
            "Adjust for technicality (grade volatility)", value=True, key="pred_tech"
        )

    def _parse_hms(txt: str | None) -> int | None:
        if not txt:
            return None
        import re

        m = re.match(r"^(\d+):(\d{2})(?::(\d{2}))?$", txt.strip())
        if not m:
            return None
        h = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3) or 0)
        return h * 3600 + mm * 60 + ss

    # Load the reference DF (if provided)
    if ref_file is not None:
        ref_path = _save_tmp(ref_file, ".gpx")
        ref_df = load_gpx_to_dataframe(ref_path)
    else:
        ref_df = None

    # Determine the reference finish time: user input OR timestamp span in GPX
    ref_time_s = _parse_hms(ref_time_txt) if ref_time_txt else None
    if ref_time_s is None and ref_df is not None and not ref_df.empty:
        ref_time_s = extract_total_time_seconds_from_gpx_df(ref_df)

    if ref_df is None or ref_df.empty:
        st.info(
            "Provide a reference GPX (with timestamps if possible) to run a prediction."
        )
    elif not ref_time_s:
        st.warning(
            "Enter the reference finish time (HH:MM or HH:MM:SS), or use a GPX with timestamps."
        )
    else:
        params = PredictParams(
            alpha_blend=alpha,
            k_riegel=k_riegel,
            vert_k_sensitivity=vert_sens,
            use_altitude=use_alt,
            use_technicality=use_tech,
        )
        res = predict_finish_time(df, ref_df, float(ref_time_s), params)

        st.success(
            f"**Predicted finish**: {fmt_hms(res['time_s'])}  \n"
            f"(physics: {fmt_hms(res['physics_s'])}, riegel: {fmt_hms(res['riegel_s'])}, k={res['k_adj']:.3f})"
        )

        st.caption("Details")
        st.write(
            {
                "Target distance (km)": (
                    float(df["cum_dist_m"].iloc[-1]) / 1000.0 if not df.empty else None
                ),
                "Reference time (s)": float(ref_time_s),
                "Blend α": alpha,
                "Riegel k": k_riegel,
                "Vertical sensitivity": vert_sens,
                "Adjust altitude": use_alt,
                "Adjust technicality": use_tech,
            }
        )

    # 3) Finish Time Predictor (uses reference GPX; publishes to session_state)
    panel("predictor")(df=df, units=units)

    # 4) Race Plan (ETAs + Fueling)
    st.markdown("---")
    plan_out = panel("raceplan")(df=df, wdf=wdf, units=units)
    wdf_eta = (plan_out or {}).get("wdf_eta")
    fuel_df = (plan_out or {}).get("fuel_df")
    plan_cfg = (plan_out or {}).get("config")

    # 5) Weather (forecast/historical + align to ETAs)
    wx_out = panel("weather")(df=df, wdf_eta=wdf_eta, plan_config=plan_cfg)
    wx_df = (wx_out or {}).get("wx_df")
    wx_at_etas = (wx_out or {}).get("wx_at_etas")

    # 6) Map (layers, grade-colored route, markers with ETA + weather)
    panel("map")(
        df=df,
        wdf=wdf,
        wdf_eta=wdf_eta,
        wx_at_etas=wx_at_etas,
        units=units,
    )

# 7) Poles (recommendation + explainer)
panel("poles")(
    df=df,
    units=units,
)

# ---- Gather inputs for Export panel safely ----
import pandas as _pd

# From Aid panel
aid_out = locals().get("aid_out") or {}
wdf = aid_out.get("wdf", _pd.DataFrame())
gpx_path = aid_out.get("gpx_path", locals().get("gpx_path", None))

# From Race Plan panel (may be named raceplan_out or plan_out in your file)
raceplan_out = locals().get("raceplan_out") or locals().get("plan_out") or {}
wdf_eta = raceplan_out.get("wdf_eta", _pd.DataFrame())
fuel_df = raceplan_out.get("fuel_df", _pd.DataFrame())

# From Weather panel
weather_out = locals().get("weather_out") or {}
wx_at_etas = weather_out.get("wx_at_etas", _pd.DataFrame())

# From earlier parsing step; make sure df exists
df = locals().get("df", _pd.DataFrame())

# Units must already exist (sidebar choice). If not, default to miles.
units = locals().get("units", "miles")

# 8) Export (PDF)
panel("export")(
    df=df,
    wdf=wdf,
    wdf_eta=wdf_eta,
    fuel_df=fuel_df,
    wx_at_etas=wx_at_etas,
    units=units,
    course_title=os.path.basename(gpx_path) if gpx_path else "course",
)
