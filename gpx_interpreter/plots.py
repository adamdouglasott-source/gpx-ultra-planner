import numpy as np, pandas as pd, matplotlib.pyplot as plt

def detect_major_climbs(df: pd.DataFrame, *, min_gain_m=100.0, min_length_m=1000.0,
                        min_avg_grade=0.005, dip_tolerance_m=15.0,
                        merge_if_descent_gap_m=20.0, max_descent_m=30.0,
                        max_descent_distance_m=200.0):
    climbs = []
    start = None
    gain = 0.0
    desc_g = 0.0
    desc_d = 0.0
    for i in range(1, len(df)):
        de = float(df["delta_elev_m"].iloc[i])
        dd = float(df["delta_dist_m"].iloc[i])
        if de >= 0:
            if start is None:
                start = i-1
                gain = 0.0; desc_g = 0.0; desc_d = 0.0
            gain += de
            # small dips forgiveness handled below
            if desc_g > 0:
                # if we started climbing again, reduce the accumulated descent
                take = min(desc_g, de)
                desc_g -= take
                desc_d = max(0.0, desc_d - dd)
        else:
            if start is not None:
                desc_g += -de
                desc_d += dd
                # if exceeded allowed descent inside a climb, close it
                if desc_g > max_descent_m or desc_d > max_descent_distance_m:
                    end = i-1
                    length_m = float(df["cum_dist_m"].iloc[end] - df["cum_dist_m"].iloc[start])
                    tot_gain = float(gain)
                    if (tot_gain >= min_gain_m) and (length_m >= min_length_m):
                        avg_grade = tot_gain / max(length_m, 1e-6)
                        if avg_grade >= min_avg_grade:
                            climbs.append((start, end, tot_gain, length_m, avg_grade))
                    start = None; gain = 0.0; desc_g = 0.0; desc_d = 0.0
            # allow small dips up to dip_tolerance_m
            if start is not None and desc_g <= dip_tolerance_m:
                continue
    if start is not None:
        end = len(df)-1
        length_m = float(df["cum_dist_m"].iloc[end] - df["cum_dist_m"].iloc[start])
        tot_gain = float(gain)
        if (tot_gain >= min_gain_m) and (length_m >= min_length_m):
            avg_grade = tot_gain / max(length_m, 1e-6)
            if avg_grade >= min_avg_grade:
                climbs.append((start, end, tot_gain, length_m, avg_grade))

    # merge with tiny saddle
    merged = []
    for c in climbs:
        if not merged:
            merged.append(list(c))
        else:
            s0,e0,g0,l0,a0 = merged[-1]
            s1,e1,g1,l1,a1 = c
            elev_e0 = float(df["ele_m"].iloc[e0])
            elev_s1 = float(df["ele_m"].iloc[s1])
            saddle = max(0.0, elev_e0 - elev_s1)
            if saddle <= merge_if_descent_gap_m:
                new_s, new_e = s0, e1
                new_l = float(df["cum_dist_m"].iloc[new_e] - df["cum_dist_m"].iloc[new_s])
                new_g = g0 + g1
                merged[-1] = [new_s, new_e, new_g, new_l, new_g/max(new_l,1e-6)]
            else:
                merged.append(list(c))
    out = []
    for s,e,g,l,a in merged:
        out.append(dict(
            start_idx=int(s), end_idx=int(e),
            start_dist_m=float(df["cum_dist_m"].iloc[s]),
            end_dist_m=float(df["cum_dist_m"].iloc[e]),
            length_m=float(l), gain_m=float(g), avg_grade=float(a)
        ))
    return out

def plot_elevation_with_major_climbs(df: pd.DataFrame, *, units="miles",
                                     min_gain_m=100.0, min_length_m=1000.0, min_avg_grade=0.005,
                                     dip_tolerance_m=15.0, merge_if_descent_gap_m=20.0,
                                     max_descent_m=30.0, max_descent_distance_m=200.0,
                                     color_bins=None):
    if color_bins is None:
        color_bins = [(0,5,"#fee8c8"),(5,10,"#fdbb84"),(10,15,"#fc8d59"),(15,20,"#e34a33"),(20,99,"#b30000")]
    x = df["cum_dist_m"].to_numpy()
    y = df["ele_m"].to_numpy()
    u = x / (1609.344 if units=="miles" else 1000.0)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(u, y, lw=1.0, color="#444444")
    major = detect_major_climbs(df,
        min_gain_m=min_gain_m, min_length_m=min_length_m, min_avg_grade=min_avg_grade,
        dip_tolerance_m=dip_tolerance_m, merge_if_descent_gap_m=merge_if_descent_gap_m,
        max_descent_m=max_descent_m, max_descent_distance_m=max_descent_distance_m
    )
    for c in major:
        g = c["avg_grade"]*100.0
        col = "#fc8d59"
        for lo,hi,cc in color_bins:
            if lo <= g < hi: col = cc; break
        ax.axvspan(c["start_dist_m"]/ (1609.344 if units=="miles" else 1000.0),
                   c["end_dist_m"]/ (1609.344 if units=="miles" else 1000.0),
                   color=col, alpha=0.35)
    ax.set_xlabel(f"Distance ({'mi' if units=='miles' else 'km'})")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Elevation with major climbs")
    return major
