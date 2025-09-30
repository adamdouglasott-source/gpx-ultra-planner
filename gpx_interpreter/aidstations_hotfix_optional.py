# gpx_interpreter/aidstations.py (optional hotfix)
import re
import numpy as np
import pandas as pd
from lxml import etree as ET
from .parser import _haversine_vec_m  # use vectorized scalar->array distance

def _extract_waypoints(path: str):
    ns = {"g":"http://www.topografix.com/GPX/1/1"}
    tree = ET.parse(path)
    wpts = tree.findall(".//g:wpt", ns)
    rows = []
    for w in wpts:
        lat = float(w.get("lat")); lon = float(w.get("lon"))
        name_el = w.find("g:name", ns); desc_el = w.find("g:desc", ns)
        name = name_el.text.strip() if name_el is not None and name_el.text else ""
        desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
        rows.append(dict(name=name, desc=desc, lat=lat, lon=lon))
    return pd.DataFrame(rows)

def waypoints_table(path: str, df: pd.DataFrame, *, units: str="miles",
                    aid_regex=r"\b(aid|as\s*\d*|aid\s*station)\b",
                    dropbag_regex=r"\b(drop\s*bag|dropbag|bag\s*drop)\b",
                    water_regex=r"\b(water|h2o|hydration|spring|well)\b",
                    crew_regex=r"\b(crew|crew\s*access|pacer|support)\b",
                    checkpoint_regex=r"\b(check(point)?|cp\s*\d*)\b") -> pd.DataFrame:
    wdf = _extract_waypoints(path)
    if wdf.empty or df is None or df.empty:
        return pd.DataFrame()

    # nearest track index & distance along
    lat_arr = df["lat"].to_numpy(dtype=float); lon_arr = df["lon"].to_numpy(dtype=float)
    cum = df["cum_dist_m"].to_numpy(dtype=float)

    idxs = []
    dist_along = []
    for _, r in wdf.iterrows():
        dists = _haversine_vec_m(r["lat"], r["lon"], lat_arr, lon_arr)
        j = int(np.argmin(dists))
        idxs.append(j)
        dist_along.append(float(cum[j]))

    wdf["nearest_idx"] = idxs
    wdf["dist_along_m"] = dist_along

    # classify
    def _match(rx, s):
        try:
            return bool(re.search(rx, s or "", flags=re.I))
        except re.error:
            return False

    cats = []
    for _, r in wdf.iterrows():
        s = (str(r.get("name","")) + " " + str(r.get("desc",""))).strip()
        is_aid = _match(aid_regex, s)
        is_drop = _match(dropbag_regex, s)
        is_water = _match(water_regex, s)
        is_crew = _match(crew_regex, s)
        is_cp = _match(checkpoint_regex, s)
        cat = []
        if is_aid: cat.append("aid")
        if is_water: cat.append("water")
        if is_crew: cat.append("crew")
        if is_cp: cat.append("checkpoint")
        cat = "+".join(cat) if cat else "other"
        if is_drop:
            cat = (cat + "+dropbag") if cat != "other" else "dropbag"
        cats.append(cat)
    wdf["category"] = cats

    return wdf
