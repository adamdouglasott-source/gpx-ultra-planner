from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass
class PredictParams:
    alpha_blend: float = 0.7           # weight on physics-ish branch
    k_riegel: float = 1.15             # distance exponent for endurance
    vert_k_sensitivity: float = 0.2    # climb cost coefficient (per vertical meter per km)
    use_altitude: bool = True
    use_technicality: bool = True

def extract_total_time_seconds_from_gpx_df(df: pd.DataFrame) -> float | None:
    """If GPX includes timestamps, return total elapsed seconds; else None."""
    if df is None or df.empty or "time" not in df.columns:
        return None
    t = pd.to_datetime(df["time"], errors="coerce")
    if t.notna().sum() < 2:
        return None
    total = (t.iloc[-1] - t.iloc[0]).total_seconds()
    return float(total) if (total and total > 0) else None

def _course_features(df: pd.DataFrame) -> dict:
    dd = df["delta_dist_m"].to_numpy(dtype=float)
    de = df["delta_elev_m"].to_numpy(dtype=float)
    dist_km = float(np.sum(dd) / 1000.0)
    gain_m = float(np.sum(np.clip(de, 0, None)))

    # Grade volatility as a proxy for technicality – lightly smoothed
    g = df.get("grade", pd.Series(np.zeros(len(df)))).fillna(0.0).to_numpy(dtype=float)
    if len(g) >= 51:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        w = 51
        med = np.median(swv(g, w), axis=-1)
        pad = w//2
        g_smooth = np.pad(med, (pad, pad), mode="edge")
    else:
        g_smooth = g
    vol = float(np.std(g_smooth))

    alt = float(np.nanmean(df["ele_m"])) if "ele_m" in df.columns else 0.0
    return {"dist_km": dist_km, "gain_m": gain_m, "vol": vol, "alt": alt}

def predict_finish_time(target_df: pd.DataFrame, ref_df: pd.DataFrame, ref_time_s: float, p: PredictParams) -> dict:
    t = _course_features(target_df); r = _course_features(ref_df)

    # Physics-ish branch: add a climb tax proportional to gain per horizontal distance
    c = p.vert_k_sensitivity
    phys_ref = r["dist_km"] * (1.0 + c * (r["gain_m"]/max(r["dist_km"]*1000.0, 1e-6)))
    phys_tar = t["dist_km"] * (1.0 + c * (t["gain_m"]/max(t["dist_km"]*1000.0, 1e-6)))

    alt_fac = (1.0 + 0.00003*(t["alt"] - r["alt"])) if p.use_altitude else 1.0
    tech_fac = (1.0 + 0.5*max(t["vol"] - r["vol"], 0.0)) if p.use_technicality else 1.0
    physics_s = ref_time_s * (phys_tar / max(phys_ref, 1e-9)) * alt_fac * tech_fac

    # Riegel branch with a small vertical term
    k = p.k_riegel
    riegel_s = ref_time_s * (max(t["dist_km"], 1e-9) / max(r["dist_km"], 1e-9))**k
    riegel_s *= (1.0 + 0.00004*(t["gain_m"] - r["gain_m"]))  # light gain correction

    # Blend
    a = p.alpha_blend
    time_s = a*physics_s + (1.0 - a)*riegel_s

    return {
        "time_s": float(time_s),
        "physics_s": float(physics_s),
        "riegel_s": float(riegel_s),
        "k_adj": float(k),
    }
