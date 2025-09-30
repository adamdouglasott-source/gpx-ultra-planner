# gpx_interpreter/ui/panel_aid.py
from __future__ import annotations
import pandas as pd
import streamlit as st

from .panel_loader import register, get_ns
from gpx_interpreter.aidstations import waypoints_table


@register("aid")
def render_aid(
    df: pd.DataFrame, *, gpx_path: str | None, units: str = "miles"
) -> dict | None:
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        return {"wdf": pd.DataFrame()}

    NS, K = get_ns("aid")
    st.subheader("Aid / Dropbag / Crew")

    aid_regex = st.text_input(
        "Aid regex", r"\b(aid|as\s*\d*|aid\s*station)\b", key=K("aid_regex")
    )
    dropbag_regex = st.text_input(
        "Dropbag regex", r"\b(drop\s*bag|dropbag|bag\s*drop)\b", key=K("dropbag_regex")
    )
    water_regex = st.text_input(
        "Water regex", r"\b(water|h2o|hydration|spring|well)\b", key=K("water_regex")
    )
    crew_regex = st.text_input(
        "Crew regex", r"\b(crew|crew\s*access|pacer|support)\b", key=K("crew_regex")
    )
    checkpoint_regex = st.text_input(
        "Checkpoint regex", r"\b(check(point)?|cp\s*\d*)\b", key=K("checkpoint_regex")
    )

    if not gpx_path:
        st.info("Upload a GPX file above to parse waypoints.")
        return {"wdf": pd.DataFrame()}

    try:
        wdf = waypoints_table(
            gpx_path,
            df,
            units=units,
            aid_regex=aid_regex,
            dropbag_regex=dropbag_regex,
            water_regex=water_regex,
            crew_regex=crew_regex,
            checkpoint_regex=checkpoint_regex,
        )
    except Exception as e:
        st.error(f"Failed to parse waypoints: {e}")
        wdf = pd.DataFrame()

    if wdf is not None and not wdf.empty:
        defaults = [
            n
            for n, c in zip(wdf["name"], wdf["category"])
            if isinstance(c, str) and ("dropbag" in c.lower())
        ]
        dropbag_sites = st.multiselect(
            "Drop bag allowed at:",
            options=list(wdf["name"]),
            default=defaults,
            key=K("dropbag_sites"),
        )
        wdf = wdf.copy()
        wdf["dropbag_allowed"] = wdf["name"].isin(dropbag_sites)

        def _norm_cat(cat, allow):
            c = (cat or "").lower().replace("+dropbag", "").strip()
            c = " ".join(c.split())
            if allow:
                if "aid" in c:
                    return "aid+dropbag"
                elif c:
                    return c + "+dropbag"
                else:
                    return "dropbag"
            return c or "other"

        wdf["category"] = [
            _norm_cat(c, allow)
            for c, allow in zip(wdf["category"], wdf["dropbag_allowed"])
        ]
        st.dataframe(wdf, width="stretch")
    else:
        st.info("No waypoints detected with the current regex settings.")

    return {"wdf": wdf if (wdf is not None) else pd.DataFrame()}
