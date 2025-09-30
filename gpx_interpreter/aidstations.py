import re, numpy as np, pandas as pd
import lxml.etree as ET
from .parser import _haversine_vec_m

def waypoints_table(gpx_path: str, df: pd.DataFrame, *, units="miles",
                    aid_regex=r"\b(aid|as\s*\d*|aid\s*station)\b",
                    dropbag_regex=r"\b(drop\s*bag|dropbag|bag\s*drop)\b",
                    water_regex=r"\b(water|h2o|hydration|spring|well)\b",
                    crew_regex=r"\b(crew|crew\s*access|pacer|support)\b",
                    checkpoint_regex=r"\b(check(point)?|cp\s*\d*)\b") -> pd.DataFrame:
    tree = ET.parse(gpx_path)
    ns = {"g":"http://www.topografix.com/GPX/1/1"}
    rows = []
    for w in tree.findall(".//g:wpt", ns):
        lat = float(w.get("lat")); lon = float(w.get("lon"))
        name_el = (
            w.find("g:name", ns)
            if w.find("g:name", ns) is not None
            else w.find("g:desc", ns)
            if w.find("g:desc", ns) is not None
            else w.find("g:sym", ns)
        )
        name = name_el.text if name_el is not None else ""
        rows.append({"name": name, "lat": lat, "lon": lon})
    wdf = pd.DataFrame(rows)
    if wdf.empty or df is None or df.empty:
        return pd.DataFrame()

    # Vectorized nearest track point per waypoint
    track_lat = df["lat"].to_numpy(dtype=float)
    track_lon = df["lon"].to_numpy(dtype=float)
    nearest_idx = []
    dist_along = []
    for _, r in wdf.iterrows():
        dists = _haversine_vec_m(float(r["lat"]), float(r["lon"]), track_lat, track_lon)
        i = int(np.argmin(dists))
        nearest_idx.append(i)
        dist_along.append(float(df["cum_dist_m"].iloc[i]))
    wdf["nearest_idx"] = nearest_idx
    wdf["dist_along_m"] = dist_along
    wdf["dist_along_units"] = [m/(1609.344 if units=="miles" else 1000.0) for m in dist_along]

    def classify(nm: str) -> str:
        s = (nm or "").lower()
        if re.search(dropbag_regex, s): return "aid+dropbag" if re.search(aid_regex, s) else "dropbag"
        if re.search(aid_regex, s): return "aid"
        if re.search(water_regex, s): return "water"
        if re.search(crew_regex, s): return "crew"
        if re.search(checkpoint_regex, s): return "checkpoint"
        return "other"
    wdf["category"] = [classify(n) for n in wdf["name"]]
    return wdf.sort_values("dist_along_m").reset_index(drop=True)
