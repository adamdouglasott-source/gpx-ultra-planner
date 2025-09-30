def fmt_hms(seconds: float) -> str:
    try:
        seconds = int(round(float(seconds)))
    except Exception:
        return "--:--:--"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}"
