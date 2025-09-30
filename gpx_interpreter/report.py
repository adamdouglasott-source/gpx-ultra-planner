# gpx_interpreter/report.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import pandas as pd

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


@dataclass
class PdfInputs:
    course_title: str
    units: str = "miles"  # "miles" | "km"
    df: Optional[pd.DataFrame] = None
    wdf: Optional[pd.DataFrame] = None
    wdf_eta: Optional[pd.DataFrame] = None
    fuel_df: Optional[pd.DataFrame] = None
    wx_at_etas: Optional[pd.DataFrame] = None


def _metric(total_m: float, units: str, label: str) -> str:
    if units == "miles":
        if label.lower().startswith("gain") or label.lower().startswith("drop"):
            return f"{total_m * 3.28084:,.0f} ft"
        return f"{total_m / 1609.344:,.1f} mi"
    else:
        if label.lower().startswith("gain") or label.lower().startswith("drop"):
            return f"{total_m:,.0f} m"
        return f"{total_m / 1000.0:,.1f} km"


def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _clip_table(df: pd.DataFrame, keep_cols: Sequence[str], max_rows: int = 18) -> pd.DataFrame:
    cols = [c for c in keep_cols if c in df.columns]
    out = df[cols].copy()
    if len(out) > max_rows:
        out = out.iloc[:max_rows].copy()
        out.loc[len(out)] = ["…"] * len(cols)
    return out


def _table(title: str, df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    styles = getSampleStyleSheet()
    elems = [Spacer(0, 6), Paragraph(f"<b>{title}</b>", styles["Heading4"]), Spacer(0, 3)]
    data = [list(df.columns)] + [[("" if pd.isna(v) else v) for v in row] for row in df.values]
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#333333")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FCFCFC")]),
            ]
        )
    )
    elems.append(t)
    return elems


def build_pdf_report(out_path: str, inputs: PdfInputs) -> None:
    """
    Build a single-page (or two if long tables) PDF summary:
    - Header + course totals
    - Waypoints summary
    - ETAs table
    - Fueling between aid
    - Weather aligned to ETAs (if available)
    """
    styles = getSampleStyleSheet()
    story = []

    df = _safe_df(inputs.df)
    wdf = _safe_df(inputs.wdf)
    wdf_eta = _safe_df(inputs.wdf_eta)
    fuel_df = _safe_df(inputs.fuel_df)
    wx_at_etas = _safe_df(inputs.wx_at_etas)

    # Header
    story.append(Paragraph(f"<b>Ultra Planner — {inputs.course_title}</b>", styles["Title"]))
    story.append(Spacer(0, 6))

    # Course totals (from df)
    try:
        gain_m = float((df.get("delta_elev_m", 0.0).clip(lower=0)).sum())
        drop_m = float((-df.get("delta_elev_m", 0.0).clip(upper=0)).sum())
        dist_m = float(df["cum_dist_m"].iloc[-1]) if not df.empty else 0.0
    except Exception:
        gain_m = drop_m = dist_m = 0.0

    totals_text = (
        f"Distance: {_metric(dist_m, inputs.units, 'distance')}  |  "
        f"Gain: {_metric(gain_m, inputs.units, 'gain')}  |  "
        f"Drop: {_metric(drop_m, inputs.units, 'drop')}"
    )
    story.append(Paragraph(totals_text, styles["Normal"]))

    # Waypoints
    if not wdf.empty:
        keep = ["name", "category", "dist_along_m"]
        wdisp = wdf.copy()
        if "dist_along_m" in wdisp.columns:
            if inputs.units == "miles":
                wdisp["dist"] = (wdisp["dist_along_m"] / 1609.344).round(2)
            else:
                wdisp["dist"] = (wdisp["dist_along_m"] / 1000.0).round(2)
            keep = ["name", "category", "dist"]
        story += _table("Waypoints", _clip_table(wdisp, keep))

    # ETAs
    if not wdf_eta.empty:
        keep = [c for c in ["eta_clock", "name", "category"] if c in wdf_eta.columns]
        tmp = wdf_eta[keep].copy()
        story += _table("ETAs", _clip_table(tmp, keep))

    # Fueling
    if not fuel_df.empty:
        keep = [c for c in ["_from", "to", "segment_time_hms", "carbs_g", "gels_equiv"] if c in fuel_df.columns]
        # prettify column headers
        mapping = {"_from": "from", "segment_time_hms": "time"}
        tmp = fuel_df[keep].rename(columns=mapping).copy()
        story += _table("Fueling between aid", _clip_table(tmp, list(tmp.columns)))

    # Weather at ETAs
    if not wx_at_etas.empty:
        keep = [c for c in ["name", "eta_clock", "temperature_2m", "precipitation_probability", "wind_speed_10m"] if c in wx_at_etas.columns]
        tmp = wx_at_etas[keep].copy()
        story += _table("Weather near ETA", _clip_table(tmp, keep))

    doc = SimpleDocTemplate(out_path, pagesize=LETTER, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    doc.build(story)