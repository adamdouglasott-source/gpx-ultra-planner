import numpy as np
import pandas as pd
import lxml.etree as ET

R_EARTH_M = 6371000.0

def _haversine_pair_m(lat1_arr, lon1_arr, lat2_arr, lon2_arr):
    """Vectorized pairwise distance between successive points (len-1 result)."""
    lat1 = np.radians(np.asarray(lat1_arr, dtype=float))
    lon1 = np.radians(np.asarray(lon1_arr, dtype=float))
    lat2 = np.radians(np.asarray(lat2_arr, dtype=float))
    lon2 = np.radians(np.asarray(lon2_arr, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R_EARTH_M * c

def _haversine_vec_m(lat1, lon1, lat2_arr, lon2_arr):
    """Distance from a single (lat1,lon1) to each (lat2[i],lon2[i])."""
    lat1r = np.radians(float(lat1)); lon1r = np.radians(float(lon1))
    lat2 = np.radians(np.asarray(lat2_arr, dtype=float))
    lon2 = np.radians(np.asarray(lon2_arr, dtype=float))
    dlat = lat2 - lat1r
    dlon = lon2 - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R_EARTH_M * c

def load_gpx_to_dataframe(path: str) -> pd.DataFrame:
    """Parse GPX to a tidy DataFrame with distances, elevation deltas, grade, and delta_time_s."""
    tree = ET.parse(path)
    ns = {"g":"http://www.topografix.com/GPX/1/1"}
    trkpts = tree.findall(".//g:trk//g:trkseg//g:trkpt", ns)
    lats, lons, eles, times = [], [], [], []
    for p in trkpts:
        lats.append(float(p.get("lat")))
        lons.append(float(p.get("lon")))
        e = p.find("g:ele", ns); t = p.find("g:time", ns)
        eles.append(float(e.text) if e is not None else float("nan"))
        times.append(pd.to_datetime(t.text) if t is not None else pd.NaT)

    df = pd.DataFrame({"lat":lats, "lon":lons, "ele_m":eles, "time":times})
    if df.empty:
        return df

    # Distances
    dd = np.zeros(len(df), dtype=float)
    if len(df) >= 2:
        dd[1:] = _haversine_pair_m(df["lat"][:-1], df["lon"][:-1], df["lat"][1:], df["lon"][1:])
    df["delta_dist_m"] = dd
    df["cum_dist_m"] = np.cumsum(dd)

    # Elevation deltas
    ele = df["ele_m"].ffill().fillna(0.0).to_numpy(dtype=float)
    de = np.zeros_like(ele)
    if len(ele) >= 2:
        de[1:] = np.diff(ele)
    df["delta_elev_m"] = de

    # Grade (safe divide)
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.divide(de, dd, out=np.zeros_like(de), where=dd>0)
    df["grade"] = np.clip(grade, -1.0, 1.0)

    # Time deltas (robust to NaT and non-monotonic stamps)
    if df["time"].notna().sum() >= 2:
        t = pd.to_datetime(df["time"], errors="coerce")
        dt = t.diff().dt.total_seconds().fillna(0.0)
        dt = dt.mask(dt < 0, 0.0)
        df["delta_time_s"] = dt.to_numpy(dtype=float)
    else:
        df["delta_time_s"] = 0.0

    return df
