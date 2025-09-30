# gpx_interpreter/ui/panel_poles.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from .panel_loader import register, get_ns
from gpx_interpreter.poles import (
    PolesConfig,
    poles_benefit,
    optimistic_improvement_bins,
    evidence_aligned_improvement_bins,
)


@register("poles")
def render_poles(df: pd.DataFrame, *, units: str = "miles") -> dict | None:
    if df is None or df.empty:
        return None

    NS, K = get_ns("poles")
    st.subheader("Poles recommendation")

    model = st.selectbox(
        "Poles improvement model",
        ["Evidence-aligned", "Optimistic"],
        index=0,
        key=K("model"),
    )
    must_carry = st.checkbox("Must carry (race rule)", True, key=K("must_carry"))
    min_grade = st.number_input(
        "Min uphill grade (fraction)",
        0.0,
        1.0,
        value=0.05,
        step=0.01,
        format="%.2f",
        key=K("min_grade"),
    )
    min_len_m = st.number_input(
        "Min uphill length (m)",
        min_value=0.0,
        value=150.0,
        step=25.0,
        key=K("min_len_m"),
    )
    weight_g = st.number_input(
        "Poles weight (grams)", min_value=0.0, value=280.0, step=10.0, key=K("weight_g")
    )
    penalty = st.number_input(
        "Carry penalty frac per km per kg",
        min_value=0.0,
        value=0.0008,
        step=0.0001,
        format="%.4f",
        key=K("penalty"),
    )
    modeled = st.checkbox(
        "Use modeled splits for poles if GPX has no time", True, key=K("modeled")
    )
    detailed = st.checkbox("Show detailed explanation", True, key=K("detailed"))

    bins_choice = (
        evidence_aligned_improvement_bins()
        if model == "Evidence-aligned"
        else optimistic_improvement_bins()
    )
    dfin = df.copy()
    if modeled and float(dfin.get("delta_time_s", pd.Series([0.0])).sum()) <= 0.0:
        # If your app has a synthesizer function, call it here; else proceed with zeros.
        pass

    cfg = PolesConfig(
        units=units,
        must_carry=must_carry,
        min_uphill_grade=min_grade,
        min_uphill_length_m=min_len_m,
        poles_weight_kg=max(0.0, float(weight_g)) / 1000.0,
        carry_penalty_frac_per_km_per_kg=float(penalty),
        improvement_bins=bins_choice,
    )
    res = poles_benefit(dfin, cfg)

    st.info(
        {
            "recommend": res.get("recommend"),
            "net_time_saved_s": round(res.get("net_time_saved_s", 0.0), 1),
            "gross_time_saved_s": round(res.get("gross_time_saved_s", 0.0), 1),
            "carry_penalty_s": round(res.get("carry_penalty_s", 0.0), 1),
            "benefit_dist_units": round(res.get("benefit_distance_units", 0.0), 2),
            "benefit_frac_%": round(
                100.0 * res.get("benefit_fraction_of_course", 0.0), 1
            ),
            "model": model,
        }
    )

    if detailed:
        net = float(res.get("net_time_saved_s", 0.0))
        gross = float(res.get("gross_time_saved_s", 0.0))
        penalty_s = float(res.get("carry_penalty_s", 0.0))
        dist_units_val = float(res.get("benefit_distance_units", 0.0))
        frac = float(res.get("benefit_fraction_of_course", 0.0))
        total_dist_units = (
            dist_units_val / max(1e-9, frac) if frac > 0 else float("nan")
        )

        st.markdown("### What these numbers mean")
        st.markdown(
            f"""
- **Pole-friendly distance**: **{dist_units_val:.2f} {'mi' if units=='miles' else 'km'}** (~{frac*100:.1f}% of course; total ≈ {total_dist_units:.1f} {'mi' if units=='miles' else 'km'}).
- **Gross time saved** on those climbs: **{gross:.1f} s**.
- **Carry penalty** across the rest: **{penalty_s:.1f} s**.
- **Net effect** = gross − penalty = **{net:.1f} s** → **{"✅ Recommend poles" if net>0 else "❌ Not recommended"}**.
"""
        )

    return {"poles_result": res}
