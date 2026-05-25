from __future__ import annotations  # Python 3.9 uyumu — PEP 585 generics + PEP 604 unions lazy eval

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
    
    # 1 derece enlem yaklasik 111 km'dir.
    # 1 derece boylam Turkiye enlemlerinde (~39-40 N) yaklasik 85 km'dir.
    # En muhafazakar (safe-margin) degerleri kullanarak hizli bounding-box kontrolu yapacagiz.
    
    for fault in faults:
        lats = fault.get("lats") or []
        lons = fault.get("lons") or []
        if not lats or not lons:
            continue
            
        # Precomputed bounding box yoksa dinamik olarak al
        min_lat = fault.get("min_lat")
        if min_lat is None:
            min_lat = min(lats)
            max_lat = max(lats)
            min_lon = min(lons)
            max_lon = max(lons)
        else:
            max_lat = fault["max_lat"]
            min_lon = fault["min_lon"]
            max_lon = fault["max_lon"]
            
        current_min = nearest["distance_km"]
        if current_min is not None:
            # Enlem bazli min/max farki (111 km/derece)
            lat_margin = current_min / 111.0
            # Boylam bazli min/max farki (Turkiye enlemlerinde en guvenli pay 80 km/derece)
            lon_margin = current_min / 80.0
            
            if lat < min_lat - lat_margin or lat > max_lat + lat_margin:
                continue
            if lon < min_lon - lon_margin or lon > max_lon + lon_margin:
                continue
                
        # Bounding box icindeyse veya daha yakin olma ihtimali varsa detayli tarama yap
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


# ---------------------------------------------------------------------------
# Tektonik Plaka Hareketi — Euler Kutbu Rotasyonu
# Kaynak: NNR-MORVEL56 (Argus, Gordon & DeMets 2011, G-cubed)
# ---------------------------------------------------------------------------

# NNR-MORVEL56 Euler kutbu parametreleri (Tablo 1)
# euler_lat, euler_lon: derece; omega: derece/milyon-yıl (sağ-el kuralı)
_PLATE_EULER_POLES: dict[str, dict] = {
    "AN": {"euler_lat": 32.93, "euler_lon":  34.12, "omega_deg_myr": 0.938},   # Anadolu
    "EU": {"euler_lat": 48.85, "euler_lon": -106.50, "omega_deg_myr": 0.2228}, # Avrasya
    "AF": {"euler_lat": 49.36, "euler_lon":  -80.44, "omega_deg_myr": 0.2677}, # Afrika
    "AR": {"euler_lat": 50.44, "euler_lon":   -5.65, "omega_deg_myr": 0.5589}, # Arabistan
}

_R_EARTH_MM = 6_371_000_000.0   # mm cinsinden Dünya yarıçapı


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _sph_to_cart(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Coğrafi koordinatı birim kartezyen vektöre çevirir (x=ECEF-X, y=Y, z=Z)."""
    phi = _deg2rad(lat_deg)
    lam = _deg2rad(lon_deg)
    return (
        math.cos(phi) * math.cos(lam),
        math.cos(phi) * math.sin(lam),
        math.sin(phi),
    )


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """3B vektör çarpımı."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def plate_velocity_at_point(
    plate_id: str,
    point_lat: float,
    point_lon: float,
) -> dict:
    """Euler kutbu rotasyonu ile noktadaki anlık hız vektörünü hesaplar.

    Parameters
    ----------
    plate_id  : str  — 'AN', 'EU', 'AF', 'AR'
    point_lat : float — enlem (kuzey pozitif, derece)
    point_lon : float — boylam (doğu pozitif, derece)

    Returns
    -------
    dict şu anahtarlarla:
        v_north_mm_yr : float  — kuzey bileşeni (mm/yıl)
        v_east_mm_yr  : float  — doğu bileşeni   (mm/yıl)
        speed_mm_yr   : float  — toplam hız büyüklüğü (mm/yıl)
        azimuth_deg   : float  — hareket yönü (K=0, D=90, ...)

    Notes
    -----
    Euler açısal hızı ω (derece/Myr) → radyan/yıl:
        ω_rad_yr = ω_deg_myr × (π/180) × 1e-6

    Yüzey hızı:  v⃗ = ω⃗ × r⃗ × R_Earth   (mm/yıl)
    Burada ω⃗ ve r⃗ birim Kartezyen vektörler, R_Earth mm cinsindendir.
    """
    if plate_id not in _PLATE_EULER_POLES:
        raise ValueError(
            f"Bilinmeyen plaka ID: {plate_id!r}. "
            f"Geçerli değerler: {list(_PLATE_EULER_POLES)}"
        )

    pole = _PLATE_EULER_POLES[plate_id]
    omega_rad_yr = pole["omega_deg_myr"] * (math.pi / 180.0) * 1e-6   # rad/yıl

    # Euler kutbu ve nokta → birim Kartezyen
    ex, ey, ez = _sph_to_cart(pole["euler_lat"], pole["euler_lon"])
    px, py, pz = _sph_to_cart(point_lat, point_lon)

    # Omega vektörü: ω⃗ = omega_rad_yr × ê_pole
    wx, wy, wz = omega_rad_yr * ex, omega_rad_yr * ey, omega_rad_yr * ez

    # Yüzey hız vektörü (Kartezyen, mm/yıl): v⃗ = ω⃗ × p̂ × R
    vx, vy, vz = _cross((wx, wy, wz), (px, py, pz))
    vx *= _R_EARTH_MM
    vy *= _R_EARTH_MM
    vz *= _R_EARTH_MM

    # Kartezyen hızı lokal ENU (East-North-Up) bileşenlerine dönüştür
    phi = _deg2rad(point_lat)
    lam = _deg2rad(point_lon)
    # Lokal North birimi: (-sin(φ)cos(λ), -sin(φ)sin(λ), cos(φ))
    # Lokal East birimi:  (-sin(λ),        cos(λ),         0      )
    v_north = (-math.sin(phi) * math.cos(lam)) * vx \
            + (-math.sin(phi) * math.sin(lam)) * vy \
            + math.cos(phi) * vz
    v_east  = (-math.sin(lam)) * vx + math.cos(lam) * vy

    speed = math.sqrt(v_north ** 2 + v_east ** 2)
    azimuth = math.degrees(math.atan2(v_east, v_north)) % 360.0

    return {
        "v_north_mm_yr": round(v_north, 3),
        "v_east_mm_yr":  round(v_east,  3),
        "speed_mm_yr":   round(speed,   3),
        "azimuth_deg":   round(azimuth, 1),
    }


# ---------------------------------------------------------------------------
# Reasenberg & Jones (1989) Artçı Deprem Olasılığı + Omori-Utsu Yasası
# Kaynak: Reasenberg & Jones (1989), Science 243:1173-1176,
#         DOI:10.1126/science.243.4895.1173
#         Omori (1894), J. Coll. Sci. Imp. Univ. Tokyo 7:111-200
#         Utsu, Ogata & Matsu'ura (1995), J. Phys. Earth 43:1-33
#         Öztürk et al. (2011), J. Seismol. — Türkiye için a, b kalibrasyonu
# ---------------------------------------------------------------------------

def omori_utsu_rate(t_days: float, K: float, c: float = 0.05, p: float = 1.0) -> float:
    """Omori-Utsu artçı deprem hızı (deprem/gün).

    n(t) = K / (t + c)^p

    Parameters
    ----------
    t_days : float
        Mainshock'tan sonra geçen gün.
    K, c, p : float
        Omori-Utsu parametreleri (K: produktivite, c: erken-zaman düzleştiricisi,
        p: bozunum üsteli; Türkiye için p ≈ 0.9-1.1).

    Returns
    -------
    float : Artçı deprem hızı (gün başına olay sayısı).
    """
    if t_days < 0:
        return 0.0
    return K / (t_days + c) ** p


def reasenberg_jones_probability(
    M_main: float,
    M_min: float,
    t1: float,
    t2: float,
    b: float = 1.0,
    p: float = 1.0,
    c: float = 0.05,
    a: float = -1.67,
) -> dict:
    """Reasenberg-Jones artçı olasılığı + Poisson beklenen sayısı.

    Parameters
    ----------
    M_main : float — mainshock büyüklüğü
    M_min  : float — minimum artçı büyüklüğü (örn. M5)
    t1, t2 : float — tahmin penceresi (gün cinsinden mainshock'tan itibaren)
    b      : float — Gutenberg-Richter b-değeri (Türkiye ≈ 1.0)
    p      : float — Omori-Utsu p (Türkiye ≈ 1.0)
    c      : float — Omori-Utsu c (gün, ≈ 0.05)
    a      : float — Reasenberg-Jones a (Türkiye için Öztürk 2011 ≈ -1.67)

    Returns
    -------
    dict :
        {
            "probability": P(N>=1) Poisson olasılığı (0-1),
            "expected":    Beklenen artçı sayısı (μ),
            "K":           Üretilmiş K parametresi,
        }

    Notes
    -----
    K = 10^(a + b·(M_main − M_min))
    μ = K · ln((t2+c)/(t1+c))                       p = 1
    μ = K · ((t1+c)^(1-p) − (t2+c)^(1-p)) / (p − 1)  p ≠ 1
    P(N >= 1) = 1 − e^(−μ)

    Kaynak: Reasenberg & Jones 1989, *Science* 243:1173-1176.
    """
    K = 10 ** (a + b * (M_main - M_min))
    if p == 1.0:
        mu = K * math.log((t2 + c) / (t1 + c))
    else:
        mu = K * ((t1 + c) ** (1 - p) - (t2 + c) ** (1 - p)) / (p - 1)
    mu = max(0.0, mu)
    probability = 1.0 - math.exp(-mu)
    return {"probability": probability, "expected": mu, "K": K}


def plate_velocity_vector(
    plate_id: str,
    point_lat: float,
    point_lon: float,
    years: int,
) -> tuple[float, float]:
    """Euler kutbu rotasyonu ile `years` yıl için yer değiştirme vektörünü döndürür.

    Parameters
    ----------
    plate_id  : str   — 'AN', 'EU', 'AF', 'AR'
    point_lat : float — başlangıç enlemi (derece)
    point_lon : float — başlangıç boylamı (derece)
    years     : int   — ileriye veya geriye simülasyon süresi (yıl)

    Returns
    -------
    (delta_lat, delta_lon) : tuple[float, float]
        Derece cinsinden konumsal kayma.
        Küçük süreler için lineer yaklaşım kullanılır (< ~1000 yıl güvenli).
        Büyük süreler için Runge-Kutta entegrasyonu önerilir.

    Notes
    -----
    1° enlem ≈ 111 320 m = 111 320 000 mm
    1° boylam ≈ 111 320 × cos(φ) mm
    """
    vel = plate_velocity_at_point(plate_id, point_lat, point_lon)
    mm_per_deg_lat = 111_320_000.0
    mm_per_deg_lon = 111_320_000.0 * math.cos(_deg2rad(point_lat))

    delta_lat = (vel["v_north_mm_yr"] * years) / mm_per_deg_lat
    delta_lon = (vel["v_east_mm_yr"]  * years) / mm_per_deg_lon

    return (round(delta_lat, 8), round(delta_lon, 8))


# ════════════════════════════════════════════════════════════════════════════
# GEM Global Active Faults — Türkiye fay koordinatları
# ────────────────────────────────────────────────────────────────────────────
# Kaynak: GEM Science Tools, gem-global-active-faults
# Lisans: CC BY 4.0
# URL: https://github.com/GEMScienceTools/gem-global-active-faults
# Akademik atıf:
#   Styron, R. & Pagani, M. (2020). The GEM Global Active Faults Database.
#   Earthquake Spectra 36(1_suppl), 160-180. DOI:10.1177/8755293020944182
# ════════════════════════════════════════════════════════════════════════════

GEM_FAULTS_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/"
    "gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson"
)

# Türkiye bbox (Karadeniz kıyısından Doğu Akdeniz'e)
TURKEY_BBOX = {"lat_min": 35.5, "lat_max": 42.5, "lon_min": 25.0, "lon_max": 45.5}


def _coord_in_turkey(c) -> bool:
    """[lon, lat] noktasının Türkiye bbox içinde olup olmadığı."""
    if not c or len(c) < 2:
        return False
    lon, lat = c[0], c[1]
    return (TURKEY_BBOX["lon_min"] <= lon <= TURKEY_BBOX["lon_max"]
            and TURKEY_BBOX["lat_min"] <= lat <= TURKEY_BBOX["lat_max"])


def fetch_gem_faults_turkey(timeout: int = 10):
    """
    GEM Global Active Faults veritabanından Türkiye faylarını çek.

    Returns
    -------
    list[dict] — her bir GeoJSON Feature; geometry tipi LineString veya MultiLineString.
                 Boş liste = network/timeout hatası (caller fallback yapmalı).

    Notes
    -----
    @st.cache_data ile sarmalanabilir; tek başına saf Python.
    """
    try:
        import requests
        r = requests.get(GEM_FAULTS_URL, timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    turkey_faults = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        in_tr = False
        if gtype == "LineString":
            if any(_coord_in_turkey(c) for c in coords):
                in_tr = True
        elif gtype == "MultiLineString":
            if any(_coord_in_turkey(c) for seg in coords for c in seg):
                in_tr = True
        if in_tr:
            turkey_faults.append(feat)
    return turkey_faults


def gem_fault_traces_for_plotly(features: list):
    """
    GEM features listesini Plotly Scattermapbox-uyumlu trace listesine dönüştür.

    Returns
    -------
    list[dict] — her trace: {"lats": [...], "lons": [...], "name": str, "props": dict}
                 Geometri LineString veya MultiLineString'in her segmenti tek trace.
    """
    traces = []
    for feat in features:
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        name = props.get("name") or props.get("ns_name") or "Anonim Fay"
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "LineString":
            segments = [coords]
        elif gtype == "MultiLineString":
            segments = coords
        else:
            continue
        for seg in segments:
            lats = [c[1] for c in seg if c and len(c) >= 2]
            lons = [c[0] for c in seg if c and len(c) >= 2]
            if len(lats) < 2:
                continue
            traces.append({"lats": lats, "lons": lons, "name": name, "props": props})
    return traces


# ════════════════════════════════════════════════════════════════════════════
# KAF Segment Koordinatları — AFAD + Şengör et al. 2005 onaylı
# ────────────────────────────────────────────────────────────────────────────
# Kaynak: Şengör, A.M.C. et al. (2005). The North Anatolian Fault: A new look.
#   Annual Review of Earth and Planetary Sciences 33, 37-112.
#   DOI:10.1146/annurev.earth.32.101802.120415
# AFAD: deprem.afad.gov.tr (resmi diri fay haritası)
# ════════════════════════════════════════════════════════════════════════════

KAF_SEGMENTLER_AFAD = [
    {"id": "S01", "ad": "Karlıova-Erzincan",
     "lat1": 39.45, "lon1": 40.70, "lat2": 39.75, "lon2": 39.50,
     "kayma_hizi_mm_yil": 18, "uzunluk_km": 130,
     "kaynak": "Şengör 2005 Ann. Rev. EPS 33; AFAD diri fay"},
    {"id": "S02", "ad": "Erzincan-Suşehri",
     "lat1": 39.75, "lon1": 39.50, "lat2": 40.10, "lon2": 37.90,
     "kayma_hizi_mm_yil": 19, "uzunluk_km": 140,
     "kaynak": "Şengör 2005; Kozacı 2007 BSSA 97"},
    {"id": "S03", "ad": "Suşehri-Tosya",
     "lat1": 40.10, "lon1": 37.90, "lat2": 41.00, "lon2": 34.10,
     "kayma_hizi_mm_yil": 20, "uzunluk_km": 340,
     "kaynak": "Barka 1996 BSSA 86; Stein 1997 GJI 128"},
    {"id": "S04", "ad": "Tosya-Gerede",
     "lat1": 41.00, "lon1": 34.10, "lat2": 40.80, "lon2": 32.20,
     "kayma_hizi_mm_yil": 21, "uzunluk_km": 170,
     "kaynak": "Barka 1996; 1943 + 1944 kırıkları"},
    {"id": "S05", "ad": "Gerede-İzmit",
     "lat1": 40.80, "lon1": 32.20, "lat2": 40.77, "lon2": 29.90,
     "kayma_hizi_mm_yil": 22, "uzunluk_km": 200,
     "kaynak": "Şengör 2005; 1944 + 1999 kırıkları"},
    {"id": "S06", "ad": "İzmit-Marmara (Prens Adaları)",
     "lat1": 40.77, "lon1": 29.90, "lat2": 40.70, "lon2": 27.00,
     "kayma_hizi_mm_yil": 22, "uzunluk_km": 250,
     "kaynak": "Şengör 2005; Le Pichon 2001 EPSL; Parsons 2004 Nature"},
]
