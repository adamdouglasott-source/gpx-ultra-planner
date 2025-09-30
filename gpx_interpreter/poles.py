# gpx_interpreter/poles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class PolesConfig:
    # Units & course parsing
    units: str = "miles"  # "miles" or "km"

    # Race rules
    must_carry: bool = True  # if True, you carry poles for entire course

    # When do poles help (course-physics thresholds)
    min_uphill_grade: float = 0.05  # fraction (e.g., 0.05 = 5%)
    min_uphill_length_m: float = 150.0  # meters of *contiguous* uphill needed

    # Gear
    poles_weight_kg: float = 0.28  # default ~280 g
    carry_penalty_frac_per_km_per_kg: float = (
        0.0008  # fractional time penalty per km per kg
    )

    # Expected improvement on uphill by grade bins (tuple of (min%, max%, improvement_fraction))
    improvement_bins: List[Tuple[float, float, float]] = None


def optimistic_improvement_bins() -> List[Tuple[float, float, float]]:
    """
    A slightly optimistic set of uphill savings based on small lab & field deltas.
    Fractions are multiplicative time reductions on uphill segments (e.g., 0.03 = 3% faster).
    Bins are in *grade percent*, not fraction.
    """
    return [
        (5, 8, 0.020),
        (8, 12, 0.030),
        (12, 18, 0.045),
        (18, 25, 0.060),
        (25, 90, 0.070),
    ]


def evidence_aligned_improvement_bins() -> List[Tuple[float, float, float]]:
    """
    More conservative “evidence-aligned” bins (reflecting mixed findings on flats/downs,
    modest but real benefits on steeper uphills).
    """
    return [
        (5, 8, 0.010),
        (8, 12, 0.020),
        (12, 18, 0.030),
        (18, 25, 0.040),
        (25, 90, 0.050),
    ]


def _contiguous_uphill_spans(
    df: pd.DataFrame, min_grade: float, min_len_m: float
) -> List[Tuple[int, int]]:
    """
    Find contiguous spans of indices where grade >= min_grade (fraction),
    and the *distance* across the span is at least min_len_m.
    Returns list of (start_idx, end_idx) inclusive.
    """
    if (
        df is None
        or df.empty
        or "grade" not in df.columns
        or "delta_dist_m" not in df.columns
    ):
        return []

    g = pd.to_numeric(df["grade"], errors="coerce").fillna(0.0).to_numpy()
    dd = pd.to_numeric(df["delta_dist_m"], errors="coerce").fillna(0.0).to_numpy()

    spans: List[Tuple[int, int]] = []
    n = len(df)

    i = 0
    while i < n:
        if g[i] >= min_grade:
            j = i
            dist = 0.0
            while j < n and g[j] >= min_grade:
                dist += float(dd[j])
                j += 1
            if dist >= min_len_m:
                spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return spans


def _grade_to_improvement_fraction(
    grade_fraction: float, bins_pct: List[Tuple[float, float, float]]
) -> float:
    """
    Map a grade (fraction) to an uphill time-saving fraction using bins in percent.
    """
    pct = abs(grade_fraction) * 100.0
    for lo, hi, frac in bins_pct:
        if pct >= lo and pct < hi:
            return frac
    return 0.0


def _synthesize_delta_time(df: pd.DataFrame, units: str) -> np.ndarray:
    """Fallback segment times if GPX lacks timing. Uses a mild GAP-like effort scaler."""
    dd = (
        pd.to_numeric(
            df.get("delta_dist_m", pd.Series(0.0, index=df.index)), errors="coerce"
        )
        .fillna(0.0)
        .to_numpy()
    )
    if units == "miles":
        baseline_mps = 1609.344 / (12 * 60)  # ~12:00/mi
    else:
        baseline_mps = 1000.0 / (7 * 60 + 27)  # ~7:27/km
    dt = dd / max(baseline_mps, 1e-6)

    gr = (
        pd.to_numeric(df.get("grade", pd.Series(0.0, index=df.index)), errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )
    # gentle effort bumps by grade magnitude
    eff = np.ones_like(gr, dtype=float)
    abs_g = np.abs(gr)
    eff += np.where(abs_g >= 0.05, 0.03, 0.0)
    eff += np.where(abs_g >= 0.08, 0.03, 0.0)
    eff += np.where(abs_g >= 0.12, 0.04, 0.0)
    eff += np.where(abs_g >= 0.18, 0.05, 0.0)
    return np.maximum(dt * eff, 0.1)


def poles_benefit(df: pd.DataFrame, cfg: PolesConfig) -> Dict[str, Any]:
    """
    Compute whether poles are recommended on this course.
    Returns a dict with:
      - recommend: bool
      - net_time_saved_s, gross_time_saved_s, carry_penalty_s
      - benefit_distance_units
      - benefit_fraction_of_course
      - detail table of spans (optional)
    """
    if cfg.improvement_bins is None:
        cfg.improvement_bins = evidence_aligned_improvement_bins()

    # Inputs
    dd = (
        pd.to_numeric(
            df.get("delta_dist_m", pd.Series(0.0, index=df.index)), errors="coerce"
        )
        .fillna(0.0)
        .to_numpy()
    )
    total_dist_m = float(np.nansum(dd))
    if total_dist_m <= 0:
        return {
            "recommend": False,
            "net_time_saved_s": 0.0,
            "gross_time_saved_s": 0.0,
            "carry_penalty_s": 0.0,
            "benefit_distance_units": 0.0,
            "benefit_fraction_of_course": 0.0,
            "spans": [],
        }

    # Segment times (observed or synthetic)
    if (
        "delta_time_s" in df.columns
        and pd.to_numeric(df["delta_time_s"], errors="coerce").fillna(0.0).sum() > 0
    ):
        dt = pd.to_numeric(df["delta_time_s"], errors="coerce").fillna(0.0).to_numpy()
    else:
        dt = _synthesize_delta_time(df, cfg.units)

    # Find contiguous uphill spans that meet thresholds
    spans = _contiguous_uphill_spans(df, cfg.min_uphill_grade, cfg.min_uphill_length_m)

    gross_saved = 0.0
    benefit_dist_m = 0.0
    details: List[Dict[str, Any]] = []

    grades = (
        pd.to_numeric(df.get("grade", pd.Series(0.0, index=df.index)), errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )

    for i, j in spans:
        # span totals
        seg_dist_m = float(np.nansum(dd[i : j + 1]))
        seg_time_s = float(np.nansum(dt[i : j + 1]))
        if seg_dist_m <= 0 or seg_time_s <= 0:
            continue

        # average grade over the span (distance-weighted)
        w = dd[i : j + 1]
        g = grades[i : j + 1]
        avg_grade = float(np.nansum(g * w) / max(np.nansum(w), 1e-9))

        # improvement fraction from bins
        imp = _grade_to_improvement_fraction(
            avg_grade, cfg.improvement_bins
        )  # 0.03 = 3% faster
        saved = seg_time_s * imp
        gross_saved += saved
        benefit_dist_m += seg_dist_m

        details.append(
            dict(
                start_idx=int(i),
                end_idx=int(j),
                length_m=float(seg_dist_m),
                avg_grade=float(avg_grade),
                improvement_frac=float(imp),
                time_saved_s=float(saved),
            )
        )

    # Carry penalty: applied to the distance you carry poles while not benefiting.
    # If must_carry: entire course distance counts; otherwise you could “stow until climbs” (simple proxy here).
    non_benefit_dist_m = (
        total_dist_m if cfg.must_carry else max(total_dist_m - benefit_dist_m, 0.0)
    )
    non_benefit_km = non_benefit_dist_m / 1000.0
    carry_penalty = (
        non_benefit_km
        * float(cfg.poles_weight_kg)
        * float(cfg.carry_penalty_frac_per_km_per_kg)
    )
    # Penalty is a *fraction of reference time*. Use total observed/synthetic time as reference:
    ref_time_s = float(np.nansum(dt))
    carry_penalty_s = float(carry_penalty * ref_time_s)

    net = float(gross_saved - carry_penalty_s)

    # Display distance units
    units_div = 1609.344 if cfg.units == "miles" else 1000.0
    benefit_units = benefit_dist_m / units_div
    frac_course = (benefit_dist_m / total_dist_m) if total_dist_m > 0 else 0.0

    return {
        "recommend": bool(net > 0),
        "net_time_saved_s": float(net),
        "gross_time_saved_s": float(gross_saved),
        "carry_penalty_s": float(carry_penalty_s),
        "benefit_distance_units": float(benefit_units),
        "benefit_fraction_of_course": float(frac_course),
        "spans": details,
    }
