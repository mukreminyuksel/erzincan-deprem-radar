from datetime import datetime, timedelta, timezone
from html import escape
import math
import hashlib


USGS_FEED_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"

QUICK_WINDOWS = {
    "Son 1 saat": timedelta(hours=1),
    "Son 6 saat": timedelta(hours=6),
    "Son 12 saat": timedelta(hours=12),
    "Son 24 saat": timedelta(hours=24),
    "Son 3 gün": timedelta(days=3),
    "Son 5 gün": timedelta(days=5),
    "Son 7 gün": timedelta(days=7),
    "Son 15 gün": timedelta(days=15),
    "Son 30 gün": timedelta(days=30),
}


def safe_html(value):
    """Escape external text before embedding it in custom Streamlit HTML."""
    if value is None:
        return ""
    return escape(str(value), quote=True)


def to_utc_naive(value):
    """Return a UTC naive datetime for compatibility with existing service params."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def has_active_sources(active_sources):
    return bool(active_sources)


def duration_from_quick_window(label):
    return QUICK_WINDOWS[label]


def distance_km(lat1, lon1, lat2, lon2):
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return round(radius * 2 * math.asin(math.sqrt(a)), 1)


def usgs_feed_url_for_window(start_dt, end_dt):
    start = to_utc_naive(start_dt)
    end = to_utc_naive(end_dt)
    hours = max(0, (end - start).total_seconds() / 3600)
    if hours <= 1:
        feed_name = "all_hour.geojson"
    elif hours <= 24:
        feed_name = "all_day.geojson"
    elif hours <= 24 * 7:
        feed_name = "all_week.geojson"
    elif hours <= 24 * 30:
        feed_name = "all_month.geojson"
    else:
        return None
    return f"{USGS_FEED_BASE}/{feed_name}"


def parse_usgs_feed_features(features, lat, lon, radius_km, min_mag):
    rows = []
    for feature in features:
        try:
            props = feature.get("properties") or {}
            coords = (feature.get("geometry") or {}).get("coordinates") or []
            if len(coords) < 2:
                continue
            mag = props.get("mag")
            if mag is None or float(mag) < min_mag:
                continue
            event_lon, event_lat = float(coords[0]), float(coords[1])
            if distance_km(lat, lon, event_lat, event_lon) > radius_km:
                continue
            depth = float(coords[2]) if len(coords) > 2 and coords[2] is not None else None
            event_time = datetime.fromtimestamp(props["time"] / 1000, tz=timezone.utc)
            rows.append(
                {
                    "zaman": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "buyukluk": float(mag),
                    "derinlik": round(abs(depth), 1) if depth is not None else None,
                    "konum": props.get("place", ""),
                    "lat": event_lat,
                    "lon": event_lon,
                    "kaynak": "USGS-Fast",
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def estimate_energy_joules(magnitude):
    """Approximate seismic energy: log10(E joules) = 1.5M + 4.8."""
    return 10 ** (1.5 * float(magnitude) + 4.8)


def source_agreement_summary(events):
    sources = sorted({str(event.get("kaynak", "")) for event in events if event.get("kaynak")})
    magnitudes = [float(event["buyukluk"]) for event in events if event.get("buyukluk") is not None]
    return {
        "source_count": len(sources),
        "sources": sources,
        "magnitude_min": min(magnitudes) if magnitudes else None,
        "magnitude_max": max(magnitudes) if magnitudes else None,
    }


def activity_level(score):
    score = float(score)
    if score < 30:
        return "Sakin"
    if score < 60:
        return "Dikkat"
    if score < 80:
        return "Yüksek"
    return "Çok Yüksek"


def nearest_fault_vertex_distance(lat, lon, faults):
    nearest = {"fault_name": "", "distance_km": None}
    for fault in faults:
        lats = fault.get("lats") or []
        lons = fault.get("lons") or []
        for fault_lat, fault_lon in zip(lats, lons):
            dist = distance_km(lat, lon, fault_lat, fault_lon)
            if nearest["distance_km"] is None or dist < nearest["distance_km"]:
                nearest = {
                    "fault_name": fault.get("fay_adi") or "Adlandırılmamış",
                    "distance_km": dist,
                }
    return nearest


def event_signature(zaman, lat, lon, magnitude):
    key = f"{zaman}|{round(float(lat), 3)}|{round(float(lon), 3)}|{round(float(magnitude), 1)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
