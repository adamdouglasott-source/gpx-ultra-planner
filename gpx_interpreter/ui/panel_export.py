# gpx_interpreter/ui/panel_export.py
from __future__ import annotations

import os
import tempfile
import pandas as pd
import streamlit as st

from .panel_loader import register
from gpx_interpreter.report import build_pdf_report, PdfInputs


@register("export")
def render_export(
    df: pd.DataFrame | None,
    *,
    wdf: pd.DataFrame | None,
    wdf_eta: pd.DataFrame | None,
    fuel_df: pd.DataFrame | None,
    wx_at_etas: pd.DataFrame | None,
    units: str = "miles",
    course_title: str = "course",
) -> dict | None:
    st.subheader("Export — Single-page PDF")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("Upload and parse a GPX file first.")
        return None

    col1, col2 = st.columns([1, 3])
    filename = col1.text_input(
        "Filename",
        value="ultra_planner_report.pdf",
        key="pdf_filename",
        help="Name of the file to save.",
    )

    if col2.button("Generate PDF", type="primary", key="pdf_generate"):
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_path = tmp.name
            tmp.close()

            build_pdf_report(
                tmp_path,
                PdfInputs(
                    course_title=course_title,
                    units=units,
                    df=df,
                    wdf=wdf,
                    wdf_eta=wdf_eta,
                    fuel_df=fuel_df,
                    wx_at_etas=wx_at_etas,
                ),
            )

            with open(tmp_path, "rb") as fh:
                st.download_button(
                    "Download PDF",
                    data=fh,
                    file_name=(filename or "ultra_planner_report.pdf"),
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"PDF export failed: {e}")

    return None