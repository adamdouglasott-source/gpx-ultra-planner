# gpx_interpreter/ui/panel_predictor.py
from __future__ import annotations

import re
import tempfile
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

from .panel_loader import register, get_ns
from gpx_interpreter.parser import load_gpx_to_dataframe


def _save_tmp(uploaded, suffix=".gpx") -> Optional[str]:
    if not uploaded:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def _hours_to_hhmm(h: float | None) -> str:
    if h is None or not (h >= 0):
        return ""
    total_minutes = int(round(h * 60))
    return f"{total_minutes//60:02d}:{total_minutes%60:02d}"


def _reference_finish_hours_from_df(ref_df: pd.DataFrame) -> float | None:
    if ref_df is None or ref_df.empty or "delta_time_s" not in ref_df.columns:
        return None
    dt = pd.to_numeric(ref_df["delta_time_s"], errors="coerce").fillna(0.0)
    if "moving" in ref_df.columns:
        moving = dt[ref_df["moving"] == True]  # noqa: E712
        total_s = float(moving.sum()) if moving.size else float(dt.sum())
    else:
        total_s = float(dt.sum())
    return (total_s / 3600.0) if total_s > 0 else None


def _gap_effort_for_grade(g: float) -> float:
    a = abs(float(g))
    return (
        1.00
        if a < 0.05
        else (
            1.03
            if a < 0.08
            else (
                1.06
                if a < 0.12
                else (1.10 if a < 0.18 else (1.18 if a < 0.25 else 1.28))
            )
        )
    )


def _effective_time_seconds(df: pd.DataFrame, baseline_mps: float) -> float:
    dd = (
        pd.to_numeric(df.get("delta_dist_m", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
    )
    gr = (
        pd.to_numeric(df.get("grade", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    )
    n = min(len(dd), len(gr))
    if n == 0:
        return 0.0
    eff = np.array([_gap_effort_for_grade(float(g)) for g in gr[:n]], dtype=float)
    mps = max(float(baseline_mps), 1e-6)
    seg = (dd[:n] / mps) * eff
    return float(np.maximum(seg, 0.1).sum())


def _course_features(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"dist_m": 0.0, "gain_m": 0.0}
    dist_m = float(
        pd.to_numeric(df.get("cum_dist_m", 0.0), errors="coerce").fillna(0.0).iloc[-1]
        if "cum_dist_m" in df.columns
        else 0.0
    )
    gain_m = float(
        pd.to_numeric(df.get("delta_elev_m", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0)
        .sum()
    )
    return {"dist_m": dist_m, "gain_m": gain_m}


@register("predictor")
def render_finish_time_predictor(
    df: pd.DataFrame, *, units: str = "miles"
) -> dict | None:
    if df is None or df.empty:
        return None

    NS, K = get_ns("predictor")
    st.subheader("Finish Time Predictor — Reference Course")

    ref_file = st.file_uploader(
        "Reference GPX (past race/effort; timestamps preferred)",
        type=["gpx"],
        key=K("ref_gpx"),
        help="If the file has timestamps, we'll read its finish time; otherwise provide the finish time below.",
    )
    ref_hhmm = st.text_input(
        "Reference finish time (HH:MM) — only used if the GPX has no timestamps",
        value="",
        key=K("ref_hhmm"),
    )

    baseline_mps = (
        (1609.344 / (12 * 60)) if units == "miles" else (1000.0 / (7 * 60 + 27))
    )

    ref_df: Optional[pd.DataFrame] = None
    ref_path = _save_tmp(ref_file, ".gpx") if ref_file else None
    if ref_path:
        try:
            ref_df = load_gpx_to_dataframe(ref_path)
            st.success(f"Loaded reference: {len(ref_df)} points.")
        except Exception as e:
            st.error(f"Failed to parse reference GPX: {e}")

    ref_finish_h: Optional[float] = None
    if ref_df is not None and not ref_df.empty:
        ref_finish_h = _reference_finish_hours_from_df(ref_df)
    if ref_finish_h is None and ref_hhmm.strip():
        m = re.match(r"^(\d{1,2}):(\d{2})$", ref_hhmm.strip())
        if m:
            ref_finish_h = int(m.group(1)) + int(m.group(2)) / 60.0

    tgt_feats = _course_features(df)
    ref_feats = (
        _course_features(ref_df)
        if ref_df is not None
        else {"dist_m": 0.0, "gain_m": 0.0}
    )

    pred_hours: Optional[float] = None
    if ref_finish_h and ref_finish_h > 0 and ref_feats["dist_m"] > 0:
        ref_eff_s = _effective_time_seconds(
            ref_df if ref_df is not None else df, baseline_mps
        )
        scale = ((ref_finish_h * 3600.0) / ref_eff_s) if ref_eff_s > 0 else 1.0
        tgt_eff_s = _effective_time_seconds(df, baseline_mps)
        base_pred_s = tgt_eff_s * scale

        gain_ref = max(ref_feats["gain_m"], 1.0)
        gain_tgt = max(tgt_feats["gain_m"], 1.0)
        gain_factor = 1.0 + 0.06 * ((gain_tgt - gain_ref) / 1000.0)
        gain_factor = max(0.85, gain_factor)

        dist_ref = max(ref_feats["dist_m"], 1.0)
        dist_tgt = max(tgt_feats["dist_m"], 1.0)
        dist_factor = (dist_tgt / dist_ref) ** 0.05

        pred_hours = max(0.1, (base_pred_s * gain_factor * dist_factor) / 3600.0)

        hhmm = _hours_to_hhmm(float(pred_hours))
        st.success(f"Predicted finish: **{hhmm}** ({pred_hours:.2f} h)")
        st.session_state["predicted_finish_hours"] = float(pred_hours)
        st.session_state["predicted_finish_hhmm"] = hhmm
        st.session_state["predicted_finish_source"] = "reference-gpx"
        st.session_state["predictor_new_value"] = True
        st.caption(f"Published to Race Plan → default target finish: {hhmm}")
    else:
        st.info(
            "Provide a reference GPX with timestamps or enter a manual reference finish time to generate a prediction."
        )
        st.session_state["predictor_new_value"] = False

    return {
        "predicted_hours": pred_hours,
        "target_features": tgt_feats,
        "reference_features": ref_feats,
    }
