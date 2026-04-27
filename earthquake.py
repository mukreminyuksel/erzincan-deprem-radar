import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
import concurrent.futures
import numpy as np
import json
import os

st.set_page_config(
    page_title="Erzincan Deprem Radari",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

ERZ_LAT = 39.7333
ERZ_LON = 39.4917

# ─── Tema ───────────────────────────────────────────────────────────────────
if "tema" not in st.session_state:
    st.session_state.tema = "dark"

DARK = st.session_state.tema == "dark"

BG       = "#050d1a" if DARK else "#eef2f7"
BG2      = "#080f1e" if DARK else "#dde6f0"   # grafik plot bg - beyaz degil, mavi-gri
BG3      = "#0d1b2a" if DARK else "#e4ecf5"
BORDER   = "#1a4a8a" if DARK else "#94b4d0"
TEXT     = "#e0e8f0" if DARK else "#0d1f30"   # daha koyu, kontrast yüksek
SUBTEXT  = "#6a8ab0" if DARK else "#2a4a6a"
GRID     = "#1a3050" if DARK else "#94b4cc"   # daha belirgin grid
CARD_BG  = "rgba(255,255,255,0.03)" if DARK else "rgba(255,255,255,0.7)"
ANNOT    = "rgba(200,215,230,0.55)" if DARK else "rgba(15,40,65,0.75)"  # zon etiketleri

def mag_color(m):
    try: m = float(m)
    except: return "#999"
    if m < 2:  return "#43A047"
    if m < 3:  return "#F9A825"
    if m < 4:  return "#FB8C00"
    if m < 5:  return "#E53935"
    if m < 6:  return "#7B1FA2"
    if m < 7:  return "#4A148C"
    return "#B71C1C"

def mag_emoji(m):
    try: m = float(m)
    except: return "⚪"
    if m < 2:  return "🟢"
    if m < 3:  return "🟡"
    if m < 4:  return "🟠"
    if m < 5:  return "🔴"
    if m < 6:  return "🟣"
    if m < 7:  return "⚫"
    return "💀"

def mag_label(m):
    try: m = float(m)
    except: return "?"
    if m < 2:  return "Hafif"
    if m < 3:  return "Kucuk"
    if m < 4:  return "Orta"
    if m < 5:  return "Buyuk"
    if m < 6:  return "Cok Buyuk"
    if m < 7:  return "Siddetli"
    return "Yikici"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 1)

# ─── Veri cekiciler ─────────────────────────────────────────────────────────

def fetch_usgs(lat, lon, radius_km, min_mag, start_dt, end_dt):
    try:
        start, end = start_dt, end_dt
        r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
            "format": "geojson", "latitude": lat, "longitude": lon,
            "maxradiuskm": radius_km, "minmagnitude": min_mag,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=14)
        rows = []
        for f in r.json().get("features", []):
            p = f["properties"]; c = f["geometry"]["coordinates"]
            t = datetime.utcfromtimestamp(p["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            rows.append({"zaman": t, "buyukluk": p.get("mag"),
                         "derinlik": round(abs(c[2]), 1) if c[2] is not None else None,
                         "konum": p.get("place", ""), "lat": c[1], "lon": c[0], "kaynak": "USGS"})
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

def fetch_emsc(lat, lon, radius_km, min_mag, start_dt, end_dt):
    try:
        start, end = start_dt, end_dt
        r = requests.get("https://www.seismicportal.eu/fdsnws/event/1/query", params={
            "format": "json", "lat": lat, "lon": lon,
            "maxradius": radius_km / 111, "minmag": min_mag,
            "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=14)
        rows = []
        for f in r.json().get("features", []):
            p = f["properties"]; c = f["geometry"]["coordinates"]
            rows.append({"zaman": p.get("time", "")[:19].replace("T", " "),
                         "buyukluk": p.get("mag"),
                         "derinlik": round(abs(c[2]), 1) if len(c) > 2 and c[2] else None,
                         "konum": p.get("flynn_region", ""),
                         "lat": c[1], "lon": c[0], "kaynak": "EMSC"})
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

def fetch_afad(lat, lon, radius_km, min_mag, start_dt, end_dt):
    try:
        start, end = start_dt, end_dt
        margin = radius_km / 111
        r = requests.get("https://deprem.afad.gov.tr/apiv2/event/filter", params={
            "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "minlat": lat - margin, "maxlat": lat + margin,
            "minlon": lon - margin, "maxlon": lon + margin,
            "minmag": min_mag, "format": "json", "limit": 1000,
            "orderby": "timedesc",
        }, headers={"User-Agent": "Mozilla/5.0"}, timeout=14)
        rows = []
        for d in r.json():
            try:
                dlat = float(d.get("latitude", 0))
                dlon = float(d.get("longitude", 0))
                if haversine(lat, lon, dlat, dlon) > radius_km:
                    continue
                rows.append({"zaman": d.get("date", "")[:19].replace("T", " "),
                             "buyukluk": float(d.get("magnitude", 0)),
                             "derinlik": abs(float(d.get("depth", 0))),
                             "konum": d.get("location", ""),
                             "lat": dlat, "lon": dlon, "kaynak": "AFAD"})
            except:
                continue
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

def fetch_kandilli(lat, lon, radius_km, min_mag, start_dt, end_dt):
    # Kandilli UTC+3 kullanir — UTC'ye ceviriyoruz
    try:
        r = requests.get("http://www.koeri.boun.edu.tr/scripts/lst0.asp",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=18)
        content = r.content.decode("iso-8859-9", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
        pre = soup.find("pre")
        if not pre:
            return [], "HATA: pre tag yok"
        cutoff = start_dt
        rows = []
        for line in pre.get_text().strip().split("\n"):
            line = line.strip()
            if not line or len(line) < 50:
                continue
            try:
                parts = line.split()
                if len(parts) < 8:
                    continue
                dt_local = datetime.strptime(parts[0] + " " + parts[1], "%Y.%m.%d %H:%M:%S")
                dt = dt_local - timedelta(hours=3)  # UTC+3 → UTC
                if dt < cutoff:
                    continue
                dlat, dlon = float(parts[2]), float(parts[3])
                depth = abs(float(parts[4]))
                mag_raw = parts[6]
                if mag_raw == "-.-":
                    continue
                mag = float(mag_raw)
                if mag < min_mag or haversine(lat, lon, dlat, dlon) > radius_km:
                    continue
                loc = " ".join(parts[8:]).strip()
                rows.append({"zaman": dt.strftime("%Y-%m-%d %H:%M:%S"),
                             "buyukluk": mag, "derinlik": depth,
                             "konum": loc, "lat": dlat, "lon": dlon,
                             "kaynak": "Kandilli"})
            except:
                continue
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

def fetch_gfz(lat, lon, radius_km, min_mag, start_dt, end_dt):
    try:
        start, end = start_dt, end_dt
        r = requests.get("https://geofon.gfz-potsdam.de/fdsnws/event/1/query", params={
            "format": "text", "lat": lat, "lon": lon,
            "maxradius": radius_km / 111, "minmagnitude": min_mag,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=14)
        rows = []
        for line in r.text.strip().split("\n"):
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 11:
                continue
            try:
                rows.append({
                    "zaman": parts[1][:19].replace("T", " "),
                    "buyukluk": float(parts[10]) if parts[10] else None,
                    "derinlik": abs(float(parts[4])) if parts[4] else None,
                    "konum": parts[12].strip() if len(parts) > 12 else "",
                    "lat": float(parts[2]), "lon": float(parts[3]),
                    "kaynak": "GFZ",
                })
            except:
                continue
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

def fetch_iris(lat, lon, radius_km, min_mag, start_dt, end_dt):
    try:
        start, end = start_dt, end_dt
        r = requests.get("https://service.iris.edu/fdsnws/event/1/query", params={
            "format": "text", "latitude": lat, "longitude": lon,
            "maxradius": radius_km / 111, "minmagnitude": min_mag,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=14)
        rows = []
        for line in r.text.strip().split("\n"):
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 11:
                continue
            try:
                rows.append({
                    "zaman": parts[1][:19].replace("T", " "),
                    "buyukluk": float(parts[10]) if parts[10] else None,
                    "derinlik": abs(float(parts[4])) if parts[4] else None,
                    "konum": parts[12].strip() if len(parts) > 12 else "",
                    "lat": float(parts[2]), "lon": float(parts[3]),
                    "kaynak": "IRIS",
                })
            except:
                continue
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

def fetch_afad_html(lat, lon, radius_km, min_mag, start_dt, end_dt):
    """Son 100 depremi AFAD HTML tablosundan çeker — magnitude tipi (ML/MW) içerir."""
    try:
        r = requests.get(
            "https://deprem.afad.gov.tr/last-earthquakes.html",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=14,
        )
        soup = BeautifulSoup(r.content.decode("utf-8", errors="replace"), "html.parser")
        rows = []
        for tr in soup.select("tbody tr"):
            cells = [td.get_text(strip=True) for td in tr.select("td")]
            if len(cells) < 7:
                continue
            try:
                zaman_str = cells[0]          # "2026-04-27 07:22:59"
                dlat      = float(cells[1])
                dlon      = float(cells[2])
                depth     = abs(float(cells[3]))
                mag_type  = cells[4]          # ML / MW / Md
                mag       = float(cells[5])
                konum     = cells[6]

                if mag < min_mag:
                    continue
                if haversine(lat, lon, dlat, dlon) > radius_km:
                    continue
                dt = datetime.strptime(zaman_str, "%Y-%m-%d %H:%M:%S")
                if dt < start_dt or dt > end_dt:
                    continue
                rows.append({
                    "zaman":    zaman_str,
                    "buyukluk": mag,
                    "derinlik": depth,
                    "konum":    konum,
                    "lat": dlat, "lon": dlon,
                    "kaynak":   f"AFAD-Web({mag_type})",
                })
            except Exception:
                continue
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

def fetch_ingv(lat, lon, radius_km, min_mag, start_dt, end_dt):
    try:
        start, end = start_dt, end_dt
        r = requests.get("https://webservices.ingv.it/fdsnws/event/1/query", params={
            "format": "text", "lat": lat, "lon": lon,
            "maxradius": radius_km / 111, "minmagnitude": min_mag,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=14)
        rows = []
        for line in r.text.strip().split("\n"):
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 11:
                continue
            try:
                rows.append({
                    "zaman": parts[1][:19].replace("T", " "),
                    "buyukluk": float(parts[10]) if parts[10] else None,
                    "derinlik": abs(float(parts[4])) if parts[4] else None,
                    "konum": parts[12].strip() if len(parts) > 12 else "",
                    "lat": float(parts[2]), "lon": float(parts[3]),
                    "kaynak": "INGV",
                })
            except:
                continue
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

ALL_FETCHERS = {
    "USGS":      fetch_usgs,
    "EMSC":      fetch_emsc,
    "AFAD":      fetch_afad,
    "AFAD-Web":  fetch_afad_html,
    "Kandilli":  fetch_kandilli,
    "GFZ":       fetch_gfz,
    "IRIS":      fetch_iris,
    "INGV":      fetch_ingv,
}

@st.cache_data(ttl=25, show_spinner=False)
def fetch_all(lat, lon, radius_km, min_mag, start_dt, end_dt, active_sources):
    statuses = {}
    all_rows = []
    fetchers = {k: v for k, v in ALL_FETCHERS.items() if k in active_sources}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fetchers)) as ex:
        futures = {ex.submit(fn, lat, lon, radius_km, min_mag, start_dt, end_dt): name
                   for name, fn in fetchers.items()}
        for future, name in futures.items():
            rows, status = future.result()
            statuses[name] = (status, len(rows))
            all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(), statuses

    df = pd.DataFrame(all_rows)
    df["buyukluk"] = pd.to_numeric(df["buyukluk"], errors="coerce")
    df["derinlik"] = pd.to_numeric(df["derinlik"], errors="coerce").abs()
    df = df.dropna(subset=["buyukluk", "lat", "lon"])
    df["zaman"] = pd.to_datetime(df["zaman"], errors="coerce")
    df = df.dropna(subset=["zaman"])
    df["uzaklik_km"] = df.apply(lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1)
    df = df.sort_values("zaman", ascending=False).reset_index(drop=True)

    # Tekillestirme — Kandilli UTC+3 duzeltmesi yapildi, toleranslar genisletildi
    # Farkli aglarin ayni depremi rapor etme suresi gercekte 0-90sn arasidir
    seen, keep = [], []
    for i, row in df.iterrows():
        dup = any(
            abs((row["zaman"] - s["zaman"]).total_seconds()) < 120  # 2 dk tolerans
            and abs(row["lat"] - s["lat"]) < 0.15                    # ~16 km
            and abs(row["lon"] - s["lon"]) < 0.15
            and abs(row["buyukluk"] - s["buyukluk"]) < 0.5           # 0.5 mag tolerans
            for s in seen
        )
        if not dup:
            keep.append(i)
            seen.append(row)

    df = df.loc[keep].reset_index(drop=True)
    df["renk"]     = df["buyukluk"].apply(mag_color)
    df["emoji"]    = df["buyukluk"].apply(mag_emoji)
    df["sinif"]    = df["buyukluk"].apply(mag_label)
    df["boyut"]    = df["buyukluk"].apply(lambda m: max(6, float(m) ** 2.1))
    df["zaman_str"] = df["zaman"].dt.strftime("%d.%m.%Y %H:%M:%S")
    return df, statuses

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, .stApp {{ background: {BG}; color: {TEXT}; }}
  /* Geniş layout — boşlukları sıkılaştır */
  .block-container {{
    padding: 0.4rem 1rem 1rem 1rem !important;
    max-width: 100% !important;
  }}
  /* Sidebar iç padding'i de sıkılaştır */
  [data-testid="stSidebar"] > div:first-child {{ padding-top: 0.5rem !important; }}
  /* Streamlit elementleri arası dikey boşluğu azalt */
  [data-testid="stVerticalBlock"] > div {{ gap: 0.4rem !important; }}
  div[data-testid="element-container"] {{ margin-bottom: 0 !important; }}

  /* Autorefresh sırasında sayfa soluklaşmasın — tüm seçicileri kapsa */
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"],
  .stApp,
  .main,
  section.main,
  .block-container {{
    opacity: 1 !important;
    transition: none !important;
    animation: none !important;
  }}
  [data-testid="stStatusWidget"],
  [data-testid="stConnectionStatus"],
  div[class*="ConnectionStatus"] {{ display: none !important; }}

  /* Header'ı arka planı şeffaf yap ama sidebar toggle butonunu sakla — sadece deploy/menü gizle */
  [data-testid="stHeader"] {{
    background: transparent !important;
    height: 0 !important;
  }}
  [data-testid="stToolbar"],
  [data-testid="stToolbarActions"],
  [data-testid="stMainMenu"],
  #MainMenu,
  [data-testid="stDecoration"],
  .stDeployButton,
  [data-testid="stAppDeployButton"] {{ display: none !important; }}
  /* Sidebar açma/kapama butonunu görünür tut */
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"],
  [data-testid="stSidebarHeader"] button {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999 !important;
  }}

  .radar-header {{
    background: linear-gradient(135deg, {BG3} 0%, {BG2} 100%);
    border: 1px solid {BORDER}; border-radius: 10px;
    padding: 0.55rem 1.1rem; margin-bottom: 0.4rem;
  }}
  .src-pill {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 16px;
    font-size: 0.75rem; font-weight: 700; margin: 2px;
  }}
  .src-ok  {{ background: {"#0d2b12" if DARK else "#e8f5e9"}; color: {"#66bb6a" if DARK else "#2e7d32"}; border: 1px solid {"#2e7d32" if DARK else "#a5d6a7"}; }}
  .src-err {{ background: {"#2b0d0d" if DARK else "#ffebee"}; color: {"#ef9a9a" if DARK else "#c62828"}; border: 1px solid {"#7d2e2e" if DARK else "#ef9a9a"}; }}

  .stat-box {{
    background: {BG3}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 0.45rem 0.5rem; text-align: center;
  }}
  .eq-scroll-container {{
    height: 720px;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-width: thin;
    scrollbar-color: {BORDER} {BG3};
  }}
  .eq-scroll-container::-webkit-scrollbar {{ width: 5px; }}
  .eq-scroll-container::-webkit-scrollbar-track {{ background: {BG3}; border-radius: 4px; }}
  .eq-scroll-container::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}

  .eq-card {{
    padding: 7px 10px; border-radius: 7px; margin: 3px 0;
    font-size: 0.82rem; border-left: 4px solid;
    background: {CARD_BG}; color: {TEXT} !important;
  }}
  .eq-card span, .eq-card b, .eq-card i {{
    color: inherit;
  }}
  .blink {{ animation: blink 1.2s step-start infinite; }}
  @keyframes blink {{ 50% {{ opacity: 0.15; }} }}

  .chart-title {{
    font-size: 0.95rem; font-weight: 700; color: {TEXT};
    margin: 0.1rem 0 0.15rem 0;
  }}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Ayarlar")

    # Tema toggle
    tema_secim = st.radio("Tema", ["Karanlik", "Aydinlik"],
                           index=0 if DARK else 1, horizontal=True)
    if (tema_secim == "Karanlik") != DARK:
        st.session_state.tema = "dark" if tema_secim == "Karanlik" else "light"
        st.rerun()

    st.markdown("---")
    radius_km = st.slider("Yaricap (km)", 50, 600, 100, 10)
    min_mag   = st.slider("Min. Buyukluk", 0.5, 5.0, 1.0, 0.5)

    ZAMAN_SECENEKLER = {
        "Son 1 gun": 1, "Son 3 gun": 3, "Son 7 gun": 7, "Son 14 gun": 14,
        "Son 30 gun": 30, "Son 3 ay": 90, "Son 6 ay": 180,
        "Son 1 yil": 365, "Ozel Tarih Araligi": -1,
    }
    zaman_secim = st.selectbox("Zaman Araligi", list(ZAMAN_SECENEKLER.keys()), index=2)

    if zaman_secim == "Ozel Tarih Araligi":
        today = datetime.utcnow().date()
        d_start = st.date_input("Baslangic", value=today - timedelta(days=30),
                                max_value=today)
        d_end   = st.date_input("Bitis", value=today, max_value=today)
        if d_start >= d_end:
            st.warning("Baslangic tarihi bitis tarihinden once olmali.")
            d_start = d_end - timedelta(days=1)
        query_start = datetime(d_start.year, d_start.month, d_start.day, 0, 0, 0)
        query_end   = datetime(d_end.year,   d_end.month,   d_end.day,   23, 59, 59)
        days_label  = f"{d_start.strftime('%d.%m.%Y')} – {d_end.strftime('%d.%m.%Y')}"
    else:
        days       = ZAMAN_SECENEKLER[zaman_secim]
        query_end  = datetime.utcnow()
        query_start = query_end - timedelta(days=days)
        days_label  = zaman_secim

    refresh_s = st.selectbox("Otomatik Yenileme",
                              [60, 30,60,120,180,240,300],
                              format_func=lambda x: f"Her {x} saniye")

    st.markdown("---")
    harita_stil = st.selectbox("Harita Stili", ["Uydu", "Uydu+Yol", "Koyu", "Acik"], index=0)
    show_faults = st.checkbox("Fay Hatlarini Goster", value=True)

    st.markdown("---")
    st.markdown("**Veri Kaynakları** (tıkla aç/kapat)")
    SRC_LABELS = {
        "USGS":     "USGS (ABD)",
        "EMSC":     "EMSC (Avrupa)",
        "AFAD":     "AFAD API (Türkiye)",
        "AFAD-Web": "AFAD Web — son 100 + ML/MW tipi",
        "Kandilli": "Kandilli Rasathanesi",
        "GFZ":      "GFZ Potsdam",
        "IRIS":     "IRIS/SAGE (ABD)",
        "INGV":     "INGV (İtalya/Akdeniz)",
    }
    if "active_sources" not in st.session_state:
        st.session_state.active_sources = list(SRC_LABELS.keys())
    active_sources = []
    for src, label in SRC_LABELS.items():
        if st.checkbox(label, value=src in st.session_state.active_sources, key=f"src_{src}"):
            active_sources.append(src)
    st.session_state.active_sources = active_sources
    if not active_sources:
        st.warning("En az bir kaynak seç!")

# ─── Otomatik yenileme ──────────────────────────────────────────────────────
st_autorefresh(interval=refresh_s * 1000, key="eq_ref")

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="radar-header">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
    <div>
      <span style="font-size:1.5rem;font-weight:800">🌍 Erzincan Deprem Radari</span>
      <span style="margin-left:12px;font-size:0.8rem;opacity:0.55">
        {radius_km} km &nbsp;·&nbsp; M{min_mag}+ &nbsp;·&nbsp;
        {days_label} &nbsp;·&nbsp; {len(active_sources)} kaynak paralel
        &nbsp;·&nbsp; <span class="blink" style="color:#f44336">● CANLI</span>
      </span>
    </div>
    <span style="font-size:0.78rem;opacity:0.5">{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Veri ───────────────────────────────────────────────────────────────────
with st.spinner(f"{len(active_sources)} kaynak paralel sorgulanıyor..."):
    df, statuses = fetch_all(ERZ_LAT, ERZ_LON, radius_km, min_mag, query_start, query_end, tuple(active_sources))

# Kaynak pilleri
src_html = ""
for name, (status, cnt) in statuses.items():
    ok = status.startswith("OK")
    cls = "src-ok" if ok else "src-err"
    src_html += f'<span class="src-pill {cls}">{"✓" if ok else "✗"} {name} {cnt}</span>'
st.markdown(src_html, unsafe_allow_html=True)

if df.empty:
    st.error("Hicbir kaynaktan veri alinamadi.")
    st.stop()

now_utc = datetime.utcnow()
last1h  = df[df["zaman"] >= now_utc - timedelta(hours=1)]
last24h = df[df["zaman"] >= now_utc - timedelta(hours=24)]
big4    = df[df["buyukluk"] >= 4.0]

# ─── Metrikler ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
boxes = [
    (c1, len(df),                        "#90caf9", "Toplam"),
    (c2, len(last24h),                   "#ffb74d", "Son 24 Saat"),
    (c3, len(last1h),                    "#a5d6a7", "Son 1 Saat"),
    (c4, f"M{df['buyukluk'].max():.1f}", mag_color(df["buyukluk"].max()), "En Buyuk"),
    (c5, len(big4),                      "#ef9a9a", "M4.0+"),
    (c6, df["kaynak"].nunique(),         "#ce93d8", "Aktif Kaynak"),
]
for col, val, color, label in boxes:
    with col:
        st.markdown(
            f'<div class="stat-box">'
            f'<div style="font-size:1.7rem;font-weight:800;color:{color}">{val}</div>'
            f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
            f'</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Harita stili ───────────────────────────────────────────────────────────
ESRI_SAT    = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"

# Fay hatları: MTA Türkiye Diri Fay Haritası 2013 (resmi, 14.500+ segment)
# Kaynak: mta.gov.tr/v3.0/sayfalar/hizmetler/doc/DFY_GEO_WGS84.zip
def fault_color(kayma):
    k = (kayma or "").upper()
    if k.startswith("SAD"): return "#ff3333"   # sağ-yanal (KAF tipi) — kırmızı
    if k.startswith("SOD"): return "#ff8800"   # sol-yanal (DAF tipi) — turuncu
    if k.startswith("T"):   return "#00bbff"   # ters — mavi
    if k.startswith("AÇ"):  return "#aa66ff"   # açılma çatlağı — mor
    if k.startswith("N"):   return "#ffdd00"   # normal — sarı
    if "SAD" in k:          return "#ff5577"
    if "SOD" in k:          return "#ffaa44"
    return "#cccccc"

@st.cache_data(show_spinner=False)
def load_fault_lines():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turkey_faults.geojson")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    lines = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        if gtype == "LineString":
            segments = [geom.get("coordinates") or []]
        elif gtype == "MultiLineString":
            segments = geom.get("coordinates") or []
        else:
            continue
        props = feat.get("properties") or {}
        kayma = props.get("kayma_turu") or ""
        color = fault_color(kayma)
        for coords in segments:
            if len(coords) < 2:
                continue
            lines.append({
                "fay_adi":   props.get("fay_adi") or "Adlandırılmamış",
                "segment":   props.get("segment") or "",
                "kayma":     props.get("kayma_aciklama") or "Bilinmiyor",
                "uzunluk":   props.get("uzunluk_km") or 0,
                "color":     color,
                "lats":      [c[1] for c in coords],
                "lons":      [c[0] for c in coords],
            })
    return lines

FAULT_LINES = load_fault_lines()

def make_mapbox_layout(stil):
    # Uydu: saf uydu + yer adlari katmani (labels below traces)
    if stil == "Uydu":
        return dict(
            style="white-bg",
            layers=[
                {"below": "traces", "sourcetype": "raster",
                 "source": [ESRI_SAT], "sourceattribution": "ESRI World Imagery"},
                {"below": "traces", "sourcetype": "raster",
                 "source": [ESRI_LABELS], "opacity": 0.85},
            ],
        )
    elif stil == "Uydu+Yol":
        return dict(
            style="white-bg",
            layers=[
                {"below": "traces", "sourcetype": "raster", "source": [ESRI_SAT]},
                {"below": "traces", "sourcetype": "raster",
                 "source": [ESRI_LABELS], "opacity": 0.9},
            ],
        )
    elif stil == "Koyu":
        return dict(style="carto-darkmatter")
    else:
        return dict(style="carto-positron")

# ─── Harita + Kayan Liste ───────────────────────────────────────────────────
col_map, col_list = st.columns([2.8, 1])

with col_map:
    st.markdown('<div class="chart-title">🗺️ Deprem Haritasi</div>', unsafe_allow_html=True)

    fig_map = go.Figure()
    bands = [
        ("M < 2",  df[df["buyukluk"] < 2],                                        "#43A047"),
        ("M 2-3",  df[(df["buyukluk"] >= 2) & (df["buyukluk"] < 3)],              "#F9A825"),
        ("M 3-4",  df[(df["buyukluk"] >= 3) & (df["buyukluk"] < 4)],              "#FB8C00"),
        ("M 4-5",  df[(df["buyukluk"] >= 4) & (df["buyukluk"] < 5)],              "#E53935"),
        ("M 5-6",  df[(df["buyukluk"] >= 5) & (df["buyukluk"] < 6)],              "#7B1FA2"),
        ("M 6-7",  df[(df["buyukluk"] >= 6) & (df["buyukluk"] < 7)],              "#4A148C"),
        ("M 7+",   df[df["buyukluk"] >= 7],                                        "#B71C1C"),
    ]
    for label, sub, color in bands:
        if sub.empty: continue
        sizes = sub["buyukluk"].apply(lambda m: max(8, m * 6))
        hover_text = sub.apply(lambda r:
            f"<b>M{r['buyukluk']:.1f} — {r['sinif']}</b><br>"
            f"Derinlik: {r['derinlik']:.1f} km<br>"
            f"Zaman: {r['zaman_str']}<br>"
            f"Konum: {str(r['konum'])[:55]}<br>"
            f"Erzincan'a: {r['uzaklik_km']} km<br>"
            f"Kaynak: {r['kaynak']}", axis=1)

        # Siyah dis hat (biraz daha buyuk, tamamen siyah)
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["lat"], lon=sub["lon"],
            mode="markers", name=label, showlegend=False,
            marker=dict(size=sizes + 3, color="rgba(0,0,0,0.75)"),
            hoverinfo="skip",
        ))
        # Renkli ic daire
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["lat"], lon=sub["lon"],
            mode="markers", name=label,
            marker=dict(size=sizes, color=color, opacity=0.92),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ))
    # Fay hatları (MTA Diri Fay Haritası — kayma türüne göre renklendirilmiş)
    if show_faults and FAULT_LINES:
        # Erzincan + (yarıçap × 1.6) bbox dışındakileri filtrele (perf için)
        deg = 1.0 / 111.0
        margin = max(radius_km * 1.6, 250) * deg
        lat_min, lat_max = ERZ_LAT - margin, ERZ_LAT + margin
        lon_min, lon_max = ERZ_LON - margin / math.cos(math.radians(ERZ_LAT)), \
                           ERZ_LON + margin / math.cos(math.radians(ERZ_LAT))

        def in_view(fault):
            return any(lat_min <= la <= lat_max for la in fault["lats"]) and \
                   any(lon_min <= lo <= lon_max for lo in fault["lons"])

        visible = [f for f in FAULT_LINES if in_view(f)]

        # Renge göre gruplayıp tek trace'e topla (None separator ile)
        by_color = {}
        for fault in visible:
            color = fault["color"]
            entry = by_color.setdefault(color, {"lats": [], "lons": [], "labels": []})
            entry["lats"].extend(fault["lats"] + [None])
            entry["lons"].extend(fault["lons"] + [None])
            seg = fault["segment"]
            label = f"{fault['fay_adi']} — {seg}" if seg else fault["fay_adi"]
            label = f"{label}<br>Kayma: {fault['kayma']}"
            if fault["uzunluk"]:
                label += f" · Uzunluk: {fault['uzunluk']} km"
            entry["labels"].extend([label] * len(fault["lats"]) + [None])

        for color, data in by_color.items():
            # Gölge (siyah, alt katman)
            fig_map.add_trace(go.Scattermapbox(
                lat=data["lats"], lon=data["lons"], mode="lines",
                showlegend=False, hoverinfo="skip",
                line=dict(color="rgba(0,0,0,0.55)", width=3.5),
            ))
            # Renkli üst çizgi
            fig_map.add_trace(go.Scattermapbox(
                lat=data["lats"], lon=data["lons"], mode="lines",
                name="Fay hattı", showlegend=False,
                line=dict(color=color, width=1.8),
                text=data["labels"],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

    # Erzincan pin
    pin_color = "#ffffff" if (harita_stil in ["Uydu", "Uydu+Yol", "Koyu"]) else "#1a2a3a"
    fig_map.add_trace(go.Scattermapbox(
        lat=[ERZ_LAT], lon=[ERZ_LON], mode="markers+text",
        name="Erzincan",
        marker=dict(size=16, color=pin_color, symbol="circle"),
        text=["📍 Erzincan"], textposition="top right",
        textfont=dict(color=pin_color, size=12, family="Arial Bold"),
        hoverinfo="skip",
    ))

    mapbox_cfg = make_mapbox_layout(harita_stil)
    mapbox_cfg.update({"center": dict(lat=ERZ_LAT, lon=ERZ_LON), "zoom": 6})

    fig_map.update_layout(
        mapbox=mapbox_cfg,
        margin=dict(t=0, b=0, l=0, r=0),
        height=780,
        legend=dict(
            bgcolor="rgba(0,0,0,0.65)" if DARK else "rgba(255,255,255,0.92)",
            font=dict(color="white" if DARK else "#1a2a3a", size=10),
            x=0.01, y=0.99,
            bordercolor="rgba(255,255,255,0.2)" if DARK else "rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
        paper_bgcolor=BG,
    )
    st.plotly_chart(fig_map, use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True,
                            "modeBarButtonsToRemove": ["toImage"],
                            "displaylogo": False})

    # Fay hattı renk lejantı
    if show_faults:
        st.markdown(f"""
        <div style="
            background:{BG2}; border:1px solid {BORDER}; border-radius:8px;
            padding:0.55rem 0.9rem; margin-top:-0.4rem;
            display:flex; flex-wrap:wrap; gap:0.9rem; align-items:center;
            font-size:0.78rem; color:{SUBTEXT};">
          <span style="font-weight:600; color:{TEXT};">Fay Hattı Türü:</span>
          <span><span style="display:inline-block;width:18px;height:3px;background:#ff3333;
                vertical-align:middle;margin-right:5px;border-radius:2px;"></span>Sağ-yanal (KAF tipi)</span>
          <span><span style="display:inline-block;width:18px;height:3px;background:#ff8800;
                vertical-align:middle;margin-right:5px;border-radius:2px;"></span>Sol-yanal (DAF tipi)</span>
          <span><span style="display:inline-block;width:18px;height:3px;background:#ffdd00;
                vertical-align:middle;margin-right:5px;border-radius:2px;"></span>Normal</span>
          <span><span style="display:inline-block;width:18px;height:3px;background:#00bbff;
                vertical-align:middle;margin-right:5px;border-radius:2px;"></span>Ters</span>
          <span><span style="display:inline-block;width:18px;height:3px;background:#aa66ff;
                vertical-align:middle;margin-right:5px;border-radius:2px;"></span>Açılma çatlağı</span>
          <span style="margin-left:auto;font-size:0.72rem;opacity:0.7;">Kaynak: MTA Diri Fay Haritası 2013</span>
        </div>
        """, unsafe_allow_html=True)

with col_list:
    st.markdown('<div class="chart-title">⚡ Son Depremler</div>', unsafe_allow_html=True)

    lf_col1, lf_col2 = st.columns(2)
    with lf_col1:
        list_mag = st.selectbox("Min büyüklük", [0, 1, 2, 3, 4, 5, 6, 7],
                                format_func=lambda x: "Tümü" if x == 0 else f"M{x}+",
                                key="list_mag")
    with lf_col2:
        list_time = st.selectbox("Zaman",
                                 ["Tümü", "Son 1 Saat", "Son 6 Saat", "Son 24 Saat"],
                                 key="list_time")

    def render_scrollable(data, limit=300):
        if data.empty:
            st.caption("Bu filtre için deprem yok.")
            return
        cards = ""
        for _, row in data.head(limit).iterrows():
            c   = mag_color(row["buyukluk"])
            e   = mag_emoji(row["buyukluk"])
            loc = str(row["konum"])[:44] if row["konum"] else "—"
            sub = "rgba(200,215,230,0.7)" if DARK else "rgba(40,60,80,0.65)"
            cards += (
                f'<div class="eq-card" style="border-left-color:{c}">'
                f'{e} <b style="color:{c}">M{row["buyukluk"]:.1f}</b>'
                f' <span style="color:{TEXT}">&nbsp;·&nbsp; {row["derinlik"]:.0f} km'
                f' &nbsp;·&nbsp; {row["uzaklik_km"]} km</span><br>'
                f'<span style="font-size:0.74rem;color:{sub}">'
                f'{row["zaman_str"]} &nbsp;·&nbsp; {loc}'
                f' &nbsp;·&nbsp; <i>{row["kaynak"]}</i></span>'
                f'</div>'
            )
        st.markdown(f'<div class="eq-scroll-container">{cards}</div>',
                    unsafe_allow_html=True)

    df_list = df.copy()
    if list_mag > 0:
        df_list = df_list[df_list["buyukluk"] >= list_mag]
    if list_time == "Son 1 Saat":
        df_list = df_list[df_list["zaman"] >= now_utc - timedelta(hours=1)]
    elif list_time == "Son 6 Saat":
        df_list = df_list[df_list["zaman"] >= now_utc - timedelta(hours=6)]
    elif list_time == "Son 24 Saat":
        df_list = df_list[df_list["zaman"] >= now_utc - timedelta(hours=24)]

    st.caption(f"{len(df_list)} deprem gösteriliyor")
    render_scrollable(df_list)

# ════════════════════════════════════════════════════════════════
# DERINLIK – ZAMAN – BUYUKLUK  (tam genislik)
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="chart-title">🔬 Derinlik · Zaman · Buyukluk — Her nokta bir deprem</div>',
            unsafe_allow_html=True)
st.caption(
    "Dikey eksen: zemin yüzeyi en üstte (0 km), aşağı doğru derinlik artar. "
    "Yatay eksen: zaman. Nokta boyutu: büyüklük. Renk: büyüklük sınıfı."
)

df_plot = df.dropna(subset=["derinlik"]).copy()
depth_max = min(df_plot["derinlik"].quantile(0.98), 200)

sinif_bands = [
    ("M < 2",  df_plot[df_plot["buyukluk"] < 2],                                        "#43A047"),
    ("M 2-3",  df_plot[(df_plot["buyukluk"] >= 2) & (df_plot["buyukluk"] < 3)],         "#F9A825"),
    ("M 3-4",  df_plot[(df_plot["buyukluk"] >= 3) & (df_plot["buyukluk"] < 4)],         "#FB8C00"),
    ("M 4-5",  df_plot[(df_plot["buyukluk"] >= 4) & (df_plot["buyukluk"] < 5)],         "#E53935"),
    ("M 5-6",  df_plot[(df_plot["buyukluk"] >= 5) & (df_plot["buyukluk"] < 6)],         "#7B1FA2"),
    ("M 6-7",  df_plot[(df_plot["buyukluk"] >= 6) & (df_plot["buyukluk"] < 7)],         "#4A148C"),
    ("M 7+",   df_plot[df_plot["buyukluk"] >= 7],                                        "#B71C1C"),
]

fig_depth = go.Figure()

# Derinlik zon bantlari (arka plan)
zones = [
    (0,  10,  "rgba(67,160,71,0.13)"  if DARK else "rgba(67,160,71,0.18)",  "Yuzeysel  0–10 km"),
    (10, 35,  "rgba(251,140,0,0.11)"  if DARK else "rgba(251,140,0,0.15)",  "Kabuk  10–35 km"),
    (35, max(depth_max + 10, 60),
              "rgba(229,57,53,0.09)"  if DARK else "rgba(229,57,53,0.12)",  "Mantle  35+ km"),
]
for y0, y1, fill, label in zones:
    fig_depth.add_hrect(y0=y0, y1=y1, fillcolor=fill, layer="below", line_width=0)
    if y0 < depth_max:
        fig_depth.add_annotation(
            x=df_plot["zaman"].min(), y=(y0 + min(y1, depth_max)) / 2,
            text=f"  {label}", showarrow=False,
            font=dict(size=9, color=ANNOT),
            xanchor="left", yanchor="middle",
        )

# Zemin yüzeyi referans çizgisi
zline_color = "rgba(80,200,100,0.7)" if DARK else "rgba(20,120,40,0.8)"
fig_depth.add_hline(y=0, line=dict(color=zline_color, width=2, dash="dot"))
fig_depth.add_annotation(
    x=df_plot["zaman"].max(), y=0,
    text="Zemin Yuzeyi (0 km)", showarrow=False,
    font=dict(size=9, color="#2e7d32" if not DARK else "#66bb6a",
              family="Arial Bold"),
    xanchor="right", yanchor="bottom",
)

for label, sub, color in sinif_bands:
    if sub.empty: continue
    fig_depth.add_trace(go.Scatter(
        x=sub["zaman"], y=sub["derinlik"],
        mode="markers", name=label,
        marker=dict(
            size=sub["buyukluk"].apply(lambda m: max(8, float(m) ** 2.2)),
            color=color, opacity=0.82,
            line=dict(width=0.8, color="rgba(255,255,255,0.25)" if DARK else "rgba(0,0,0,0.18)"),
        ),
        text=sub.apply(lambda r:
            f"<b>M{r['buyukluk']:.1f}</b><br>"
            f"Derinlik: <b>{r['derinlik']:.1f} km</b> (zeminden asagiya)<br>"
            f"Zaman: {r['zaman_str']}<br>"
            f"Konum: {str(r['konum'])[:55]}<br>"
            f"Erzincan'a: {r['uzaklik_km']} km &nbsp;|&nbsp; {r['kaynak']}", axis=1),
        hovertemplate="%{text}<extra></extra>",
    ))

# M4+ etiketler
big_ann = df_plot[df_plot["buyukluk"] >= 4.0]
if not big_ann.empty:
    fig_depth.add_trace(go.Scatter(
        x=big_ann["zaman"], y=big_ann["derinlik"],
        mode="text", showlegend=False,
        text=big_ann["buyukluk"].apply(lambda m: f"M{m:.1f}"),
        textposition="top center",
        textfont=dict(color="#b71c1c" if not DARK else "#ffcdd2", size=11, family="Arial Black"),
    ))

fig_depth.update_layout(
    height=520,
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font=dict(color=TEXT, size=12, family="Arial"),
    yaxis=dict(
        title=dict(text="Derinlik (km)  —  asagi dogru artar",
                   font=dict(size=12, color=TEXT)),
        autorange="reversed",
        range=[depth_max, -2],
        gridcolor=GRID, gridwidth=1,
        zeroline=True,
        zerolinecolor="#2e7d32" if not DARK else "#2a7a40",
        zerolinewidth=2,
        ticksuffix=" km",
        dtick=10,
        tickfont=dict(color=TEXT),
    ),
    xaxis=dict(
        title=dict(text="Zaman", font=dict(color=TEXT)),
        gridcolor=GRID, gridwidth=1,
        tickfont=dict(color=TEXT),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0.55)" if DARK else "rgba(255,255,255,0.92)",
        font=dict(color="#e0e8f0" if DARK else "#1a2a3a", size=11),
        x=1.01, y=1, xanchor="left",
        title=dict(text="Buyukluk Sinifi",
                   font=dict(size=10, color="#e0e8f0" if DARK else "#1a2a3a")),
        bordercolor=BORDER, borderwidth=1,
    ),
    margin=dict(t=10, b=50, l=70, r=140),
    hovermode="closest",
)
st.plotly_chart(fig_depth, use_container_width=True,
                config={"displayModeBar": True, "displaylogo": False})

# ════════════════════════════════════════════════════════════════
# AKTIVITE GRAFİKLERİ
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="chart-title">📊 Deprem Aktivitesi</div>', unsafe_allow_html=True)

view_mode = st.radio("Gorunum:", ["Gunluk", "Saatlik", "Kumulatif"],
                     horizontal=True, key="act_mode")

common_layout = dict(
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font=dict(color=TEXT, size=12, family="Arial"),
    xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
    yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
    legend=dict(
        bgcolor="rgba(0,0,0,0.55)" if DARK else "rgba(255,255,255,0.95)",
        font=dict(color=TEXT, size=11),
        orientation="h", x=0, y=1.08,
        bordercolor=BORDER, borderwidth=1,
    ),
    margin=dict(t=40, b=40, l=60, r=70),
    hovermode="x unified",
    height=420,
)

SINIF_COLORS = {
    "Hafif":    "#43A047",
    "Kucuk":    "#F9A825",
    "Orta":     "#FB8C00",
    "Buyuk":    "#E53935",
    "Cok Buyuk":"#7B1FA2",
    "Siddetli": "#4A148C",
    "Yikici":   "#B71C1C",
}

if view_mode == "Gunluk":
    df["gun"] = df["zaman"].dt.date
    grouped   = df.groupby(["gun", "sinif"]).size().reset_index(name="sayi")
    daily_tot = df.groupby("gun").size().reset_index(name="toplam").sort_values("gun")
    daily_tot["kumulatif"] = daily_tot["toplam"].cumsum()

    fig_act = make_subplots(specs=[[{"secondary_y": True}]])
    for sinif, color in SINIF_COLORS.items():
        sub = grouped[grouped["sinif"] == sinif]
        if sub.empty: continue
        fig_act.add_trace(go.Bar(x=sub["gun"], y=sub["sayi"],
                                  name=sinif, marker_color=color, opacity=0.85),
                           secondary_y=False)
    fig_act.add_trace(go.Scatter(
        x=daily_tot["gun"], y=daily_tot["kumulatif"],
        name="Kumulatif", mode="lines+markers",
        line=dict(color="#90caf9", width=2, dash="dot"),
        marker=dict(size=5, color="#90caf9"),
    ), secondary_y=True)
    fig_act.update_layout(barmode="stack", **common_layout)
    fig_act.update_yaxes(title_text="Gunluk Sayi", gridcolor=GRID, secondary_y=False)
    fig_act.update_yaxes(title_text="Kumulatif Toplam", secondary_y=True,
                          gridcolor="rgba(0,0,0,0)")

elif view_mode == "Saatlik":
    df["saat"] = df["zaman"].dt.floor("h")
    hourly = df.groupby("saat").agg(
        sayi=("buyukluk", "count"),
        max_mag=("buyukluk", "max"),
        avg_mag=("buyukluk", "mean"),
    ).reset_index()
    fig_act = make_subplots(specs=[[{"secondary_y": True}]])
    fig_act.add_trace(go.Bar(
        x=hourly["saat"], y=hourly["sayi"], name="Saatlik Sayi",
        marker=dict(color=hourly["max_mag"], colorscale="YlOrRd", showscale=True,
                    colorbar=dict(title="Max M", thickness=12, len=0.7)),
        hovertemplate="<b>%{x}</b><br>Deprem: %{y}<extra></extra>",
    ), secondary_y=False)
    fig_act.add_trace(go.Scatter(
        x=hourly["saat"], y=hourly["avg_mag"], name="Ort. Buyukluk",
        mode="lines+markers", line=dict(color="#ce93d8", width=2),
        marker=dict(size=5),
    ), secondary_y=True)
    fig_act.update_layout(**common_layout)
    fig_act.update_yaxes(title_text="Saatlik Sayi", gridcolor=GRID, secondary_y=False)
    fig_act.update_yaxes(title_text="Ort. Buyukluk", secondary_y=True,
                          gridcolor="rgba(0,0,0,0)", range=[0, 6])

else:  # Kumulatif
    df_s = df.sort_values("zaman").copy()
    fig_act = go.Figure()
    for sinif, color in SINIF_COLORS.items():
        sub = df_s[df_s["sinif"] == sinif].copy()
        if sub.empty: continue
        sub["cum"] = range(1, len(sub) + 1)
        fig_act.add_trace(go.Scatter(
            x=sub["zaman"], y=sub["cum"], mode="lines",
            name=sinif, line=dict(color=color, width=2), stackgroup="one",
        ))
    fig_act.update_layout(**common_layout)
    fig_act.update_yaxes(title_text="Kumulatif Sayi", gridcolor=GRID)

st.plotly_chart(fig_act, use_container_width=True,
                config={"displayModeBar": False, "displaylogo": False})

# ─── Buyukluk + Derinlik dagilimi ───────────────────────────────────────────
col_h1, col_h2 = st.columns(2)
hist_layout = dict(
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font=dict(color=TEXT, size=11, family="Arial"),
    margin=dict(t=10, b=30, l=45, r=20), height=260,
    xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
    yaxis=dict(gridcolor=GRID, title=dict(text="Deprem Sayisi", font=dict(color=TEXT)),
               tickfont=dict(color=TEXT)),
)

with col_h1:
    st.markdown('<div class="chart-title">📉 Buyukluk Dagilimi</div>', unsafe_allow_html=True)
    fig_h1 = px.histogram(df, x="buyukluk", nbins=30,
                           color_discrete_sequence=["#1a73e8"],
                           labels={"buyukluk": "Buyukluk (M)"})
    fig_h1.update_traces(marker_line_width=0.5,
                          marker_line_color="rgba(255,255,255,0.2)")
    fig_h1.update_layout(**hist_layout)
    st.plotly_chart(fig_h1, use_container_width=True,
                    config={"displayModeBar": False, "displaylogo": False})

with col_h2:
    st.markdown('<div class="chart-title">🏔️ Derinlik Dagilimi</div>', unsafe_allow_html=True)
    fig_h2 = px.histogram(df[df["derinlik"] <= 200], x="derinlik", nbins=30,
                           color_discrete_sequence=["#FB8C00"],
                           labels={"derinlik": "Derinlik (km)"})
    fig_h2.update_traces(marker_line_width=0.5,
                          marker_line_color="rgba(255,255,255,0.2)")
    fig_h2.update_layout(**hist_layout)
    st.plotly_chart(fig_h2, use_container_width=True,
                    config={"displayModeBar": False, "displaylogo": False})

# ════════════════════════════════════════════════════════════════
# KORELASYON MATRİSİ — En büyük deprem öncesi öncü örüntüler
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="chart-title">🔬 Öncü Deprem Korelasyon Analizi</div>', unsafe_allow_html=True)
st.caption(
    "Seçilen dönemdeki en büyük deprem baz alınır. "
    "O depremin öncesinde aynı bölgede yaşanan depremlerin özellikleri arasındaki korelasyon gösterilir. "
    "Negatif 'gün_önce' = ana depremi geç takip eden artçılar. "
    "Güçlü korelasyonlar (|r| > 0.5) potansiyel öncü örüntülere işaret edebilir."
)

if len(df) >= 5:
    # En büyük depremi bul
    idx_max = df["buyukluk"].idxmax()
    main_eq = df.loc[idx_max]

    korr_pencere_gun = st.slider(
        "Analiz penceresi (ana deprem öncesi kaç gün)",
        min_value=3, max_value=90, value=30, step=3,
        key="korr_pencere",
    )
    korr_radius = st.slider(
        "Etki yarıçapı (km)",
        min_value=30, max_value=300, value=100, step=10,
        key="korr_radius",
    )

    pencere_baslangic = main_eq["zaman"] - timedelta(days=korr_pencere_gun)

    # Öncü adaylar: ana depremi geçmeden önce, belirtilen yarıçapta
    precursors = df[
        (df["zaman"] < main_eq["zaman"]) &
        (df["zaman"] >= pencere_baslangic) &
        (df.apply(lambda r: haversine(main_eq["lat"], main_eq["lon"], r["lat"], r["lon"]), axis=1) <= korr_radius) &
        (df.index != idx_max)
    ].copy()

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown(f"""
        <div class="stat-box">
          <div style="font-size:0.75rem;color:{SUBTEXT}">Ana Deprem</div>
          <div style="font-size:1.4rem;font-weight:800;color:#E53935">M{main_eq['buyukluk']:.1f}</div>
          <div style="font-size:0.72rem;color:{SUBTEXT}">{main_eq['zaman_str']}</div>
        </div>""", unsafe_allow_html=True)
    with col_info2:
        st.markdown(f"""
        <div class="stat-box">
          <div style="font-size:0.75rem;color:{SUBTEXT}">Konum</div>
          <div style="font-size:0.9rem;font-weight:700">{str(main_eq['konum'])[:40]}</div>
          <div style="font-size:0.72rem;color:{SUBTEXT}">{main_eq['lat']:.3f}N, {main_eq['lon']:.3f}E · {main_eq['derinlik']:.0f} km</div>
        </div>""", unsafe_allow_html=True)
    with col_info3:
        st.markdown(f"""
        <div class="stat-box">
          <div style="font-size:0.75rem;color:{SUBTEXT}">Öncü Aday Sayısı</div>
          <div style="font-size:1.4rem;font-weight:800;color:#1a73e8">{len(precursors)}</div>
          <div style="font-size:0.72rem;color:{SUBTEXT}">{korr_pencere_gun} gün · {korr_radius} km yarıçap</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if len(precursors) >= 4:
        # Özellik matrisi oluştur
        precursors["gun_once"] = (main_eq["zaman"] - precursors["zaman"]).dt.total_seconds() / 86400
        precursors["uzaklik_ana"] = precursors.apply(
            lambda r: haversine(main_eq["lat"], main_eq["lon"], r["lat"], r["lon"]), axis=1
        )

        feat_cols = {
            "gun_once":   "Gün Önce",
            "uzaklik_ana": "Uzaklık (km)",
            "derinlik":   "Derinlik (km)",
            "buyukluk":   "Büyüklük (M)",
            "lat":        "Enlem",
            "lon":        "Boylam",
        }
        corr_df = precursors[list(feat_cols.keys())].dropna()
        corr_df.columns = list(feat_cols.values())
        corr_matrix = corr_df.corr()

        col_hm, col_sc = st.columns([1, 1.2])

        with col_hm:
            st.markdown('<div class="chart-title">🟥 Korelasyon Matrisi</div>', unsafe_allow_html=True)
            # Renk: kırmızı=pozitif, mavi=negatif korelasyon
            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                texttemplate="%{text}",
                textfont=dict(size=11, color=TEXT),
                hovertemplate="<b>%{y} ↔ %{x}</b><br>r = %{z:.3f}<extra></extra>",
                showscale=True,
                colorbar=dict(
                    title=dict(text="r", font=dict(color=TEXT)),
                    tickfont=dict(color=TEXT),
                    thickness=14, len=0.85,
                ),
            ))
            fig_corr.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2,
                font=dict(color=TEXT, size=11, family="Arial"),
                height=370,
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(tickfont=dict(color=TEXT), tickangle=-30),
                yaxis=dict(tickfont=dict(color=TEXT)),
            )
            st.plotly_chart(fig_corr, use_container_width=True,
                            config={"displayModeBar": False, "displaylogo": False})

        with col_sc:
            st.markdown('<div class="chart-title">📍 Öncü Adaylar — Zaman & Büyüklük</div>', unsafe_allow_html=True)
            fig_pre = go.Figure()

            # Öncü depremler — boyut=büyüklük, renk=derinlik
            fig_pre.add_trace(go.Scatter(
                x=precursors["gun_once"],
                y=precursors["buyukluk"],
                mode="markers",
                name="Öncü aday",
                marker=dict(
                    size=precursors["buyukluk"].apply(lambda m: max(8, m * 7)),
                    color=precursors["derinlik"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Derinlik km", font=dict(color=TEXT, size=10)),
                        tickfont=dict(color=TEXT, size=9),
                        thickness=12, len=0.7, x=1.02,
                    ),
                    line=dict(width=1, color="rgba(255,255,255,0.4)" if DARK else "rgba(0,0,0,0.2)"),
                    opacity=0.85,
                ),
                text=precursors.apply(lambda r:
                    f"<b>M{r['buyukluk']:.1f}</b><br>"
                    f"Ana depremi {r['gun_once']:.1f} gün önce<br>"
                    f"Uzaklık: {r['uzaklik_ana']:.1f} km<br>"
                    f"Derinlik: {r['derinlik']:.1f} km<br>"
                    f"Zaman: {r['zaman_str']}", axis=1),
                hovertemplate="%{text}<extra></extra>",
            ))

            # Ana deprem işareti
            fig_pre.add_vline(x=0, line=dict(color="#E53935", width=2, dash="dot"))
            fig_pre.add_annotation(
                x=0, y=main_eq["buyukluk"],
                text=f"  ← Ana M{main_eq['buyukluk']:.1f}",
                showarrow=False,
                font=dict(color="#E53935", size=11, family="Arial Bold"),
                xanchor="left",
            )

            fig_pre.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2,
                font=dict(color=TEXT, size=11, family="Arial"),
                height=370,
                margin=dict(t=10, b=40, l=55, r=80),
                xaxis=dict(
                    title=dict(text="Ana Depremden Kaç Gün Önce", font=dict(color=TEXT)),
                    gridcolor=GRID, tickfont=dict(color=TEXT),
                    autorange="reversed",
                ),
                yaxis=dict(
                    title=dict(text="Büyüklük (M)", font=dict(color=TEXT)),
                    gridcolor=GRID, tickfont=dict(color=TEXT),
                ),
                legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
                hovermode="closest",
            )
            st.plotly_chart(fig_pre, use_container_width=True,
                            config={"displayModeBar": False, "displaylogo": False})

        # Önemli korelasyon tespitleri
        strong = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if abs(r) >= 0.4:
                    col_a = corr_matrix.columns[i]
                    col_b = corr_matrix.columns[j]
                    yön = "pozitif ↑" if r > 0 else "negatif ↓"
                    kuvvet = "güçlü" if abs(r) >= 0.65 else "orta"
                    strong.append(f"**{col_a}** ↔ **{col_b}**: r={r:.2f} ({kuvvet} {yön})")
        if strong:
            st.markdown("**Dikkat çeken korelasyonlar (|r| ≥ 0.40):**")
            for s in strong:
                st.markdown(f"- {s}")
        else:
            st.info("Bu dönemde belirgin bir korelasyon örüntüsü tespit edilmedi (|r| < 0.40).")
    else:
        st.info(f"Korelasyon analizi için en az 4 öncü aday gerekli. "
                f"Mevcut: {len(precursors)}. Yarıçapı veya pencereyi genişletin.")
else:
    st.info("Korelasyon analizi için yeterli veri yok.")

# ════════════════════════════════════════════════════════════════
# BİLİMSEL ANALİZ: b-değeri · Benioff Zorlanması · Epimerkez Göçü
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="chart-title">🧪 Bilimsel Sismoloji Analizleri</div>', unsafe_allow_html=True)
st.caption(
    "b-değeri (Gutenberg-Richter) · Benioff kümülatif zorlanması · Epimerkez göç analizi. "
    "Bu üç gösterge akademik çalışmalarda öncü örüntü araştırmalarında kullanılır."
)

sci_df = df.dropna(subset=["buyukluk", "derinlik", "lat", "lon"]).copy()
sci_df = sci_df.sort_values("zaman").reset_index(drop=True)

if len(sci_df) >= 20:
    col_b, col_ben = st.columns(2)

    # ── b-değeri kayan pencere ──────────────────────────────────
    with col_b:
        st.markdown('<div class="chart-title">📐 b-Değeri Zaman Serisi</div>', unsafe_allow_html=True)
        st.caption("Kayan pencerede Gutenberg-Richter b-değeri. Büyük deprem öncesi düşüş öncü sinyal olabilir.")

        WINDOW = max(20, len(sci_df) // 8)
        Mc = float(sci_df["buyukluk"].quantile(0.15))  # yaklaşık tamamlanma büyüklüğü

        b_vals, b_times, b_counts = [], [], []
        for i in range(WINDOW, len(sci_df) + 1, max(1, WINDOW // 4)):
            chunk = sci_df.iloc[i - WINDOW:i]
            above = chunk[chunk["buyukluk"] >= Mc]
            if len(above) < 10:
                continue
            mean_m = above["buyukluk"].mean()
            if mean_m <= Mc:
                continue
            b = math.log10(math.e) / (mean_m - Mc)
            b_vals.append(round(b, 3))
            b_times.append(chunk["zaman"].iloc[-1])
            b_counts.append(len(above))

        if len(b_vals) >= 3:
            b_mean = sum(b_vals) / len(b_vals)
            fig_b = go.Figure()
            fig_b.add_hline(y=b_mean,
                            line=dict(color="#90caf9", width=1, dash="dot"))
            fig_b.add_annotation(
                x=b_times[-1], y=b_mean,
                text=f"  Ortalama b={b_mean:.2f}",
                showarrow=False, font=dict(color="#90caf9", size=9), xanchor="left",
            )
            colors_b = [mag_color(4.5 - b * 0.8) for b in b_vals]
            fig_b.add_trace(go.Scatter(
                x=b_times, y=b_vals, mode="lines+markers",
                name="b-değeri",
                line=dict(color="#64b5f6", width=2),
                marker=dict(size=7, color=colors_b,
                            line=dict(width=1, color="rgba(255,255,255,0.3)" if DARK else "rgba(0,0,0,0.2)")),
                hovertemplate="<b>b = %{y:.3f}</b><br>%{x}<extra></extra>",
            ))
            fig_b.add_hrect(y0=0, y1=0.7,
                            fillcolor="rgba(229,57,53,0.08)", layer="below", line_width=0)
            fig_b.add_annotation(
                x=b_times[0], y=0.35,
                text="  b < 0.7: Yüksek stres bölgesi",
                showarrow=False, font=dict(size=8, color="rgba(229,57,53,0.8)"), xanchor="left",
            )
            fig_b.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2,
                font=dict(color=TEXT, size=11),
                height=300, margin=dict(t=10, b=40, l=55, r=20),
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                yaxis=dict(
                    title=dict(text="b-değeri", font=dict(color=TEXT)),
                    gridcolor=GRID, tickfont=dict(color=TEXT), range=[0, 2.5],
                ),
                legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_b, use_container_width=True,
                            config={"displayModeBar": False, "displaylogo": False})
            st.caption(f"Pencere: {WINDOW} deprem | Mc ≈ M{Mc:.1f} | Veri noktası: {len(b_vals)}")
        else:
            st.info("b-değeri için yeterli veri yok. Zaman aralığını genişletin.")

    # ── Kümülatif Benioff Zorlanması ────────────────────────────
    with col_ben:
        st.markdown('<div class="chart-title">⚡ Benioff Kümülatif Zorlanması</div>', unsafe_allow_html=True)
        st.caption("√Enerji toplamı. İvmelenen eğri (concave up) büyük deprem öncesi kritik nokta işareti olabilir.")

        # Enerji ∝ 10^(1.5M) → Benioff strain = Σ√(10^(1.5M))
        sci_df["benioff"] = sci_df["buyukluk"].apply(lambda m: math.sqrt(10 ** (1.5 * m)))
        sci_df["cum_benioff"] = sci_df["benioff"].cumsum()
        # Normalize 0-100
        b_max = sci_df["cum_benioff"].max()
        sci_df["cum_norm"] = sci_df["cum_benioff"] / b_max * 100 if b_max > 0 else 0

        fig_ben = go.Figure()
        fig_ben.add_trace(go.Scatter(
            x=sci_df["zaman"], y=sci_df["cum_norm"],
            mode="lines", name="Benioff Zorlanması",
            line=dict(color="#ffb74d", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(255,183,77,0.12)" if DARK else "rgba(255,183,77,0.20)",
            hovertemplate="<b>%{y:.1f}</b><br>%{x}<extra></extra>",
        ))
        # M4+ olayları kırmızı çizgi ile işaretle
        big_events = sci_df[sci_df["buyukluk"] >= 4.0]
        for _, ev in big_events.iterrows():
            fig_ben.add_vline(
                x=ev["zaman"].timestamp() * 1000,
                line=dict(color="#E53935", width=1.2, dash="dot"),
            )
        if not big_events.empty:
            fig_ben.add_trace(go.Scatter(
                x=big_events["zaman"],
                y=sci_df.loc[big_events.index, "cum_norm"],
                mode="markers", name="M4+ olaylar",
                marker=dict(size=9, color="#E53935", symbol="triangle-up",
                            line=dict(width=1, color="rgba(255,255,255,0.5)")),
                hovertemplate="<b>M%{text}</b><br>%{x}<extra></extra>",
                text=big_events["buyukluk"].apply(lambda m: f"{m:.1f}"),
            ))
        fig_ben.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TEXT, size=11),
            height=300, margin=dict(t=10, b=40, l=55, r=20),
            xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
            yaxis=dict(
                title=dict(text="Kümülatif Benioff (normalize %)", font=dict(color=TEXT)),
                gridcolor=GRID, tickfont=dict(color=TEXT),
            ),
            legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)",
                        orientation="h", x=0, y=1.08),
        )
        st.plotly_chart(fig_ben, use_container_width=True,
                        config={"displayModeBar": False, "displaylogo": False})

    # ── Epimerkez Göç Analizi ───────────────────────────────────
    st.markdown("---")
    col_mig1, col_mig2 = st.columns(2)

    with col_mig1:
        st.markdown('<div class="chart-title">🧭 Epimerkez Göç Haritası</div>', unsafe_allow_html=True)
        st.caption("Depremlerin zamansal sıralaması. Mor=eski, sarı=yeni. Fay segmentine doğru göç öncü işaret olabilir.")

        n_pts = len(sci_df)
        time_idx = list(range(n_pts))
        fig_mig = go.Figure()
        fig_mig.add_trace(go.Scatter(
            x=sci_df["lon"], y=sci_df["lat"],
            mode="markers",
            marker=dict(
                size=sci_df["buyukluk"].apply(lambda m: max(5, m * 5)),
                color=time_idx,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Zaman →", font=dict(color=TEXT, size=10)),
                    tickfont=dict(color=TEXT, size=9),
                    tickvals=[0, n_pts - 1],
                    ticktext=["Eski", "Yeni"],
                    thickness=12, len=0.7,
                ),
                opacity=0.85,
                line=dict(width=0.8, color="rgba(255,255,255,0.3)" if DARK else "rgba(0,0,0,0.2)"),
            ),
            text=sci_df.apply(lambda r:
                f"<b>M{r['buyukluk']:.1f}</b><br>{r['zaman_str']}<br>{str(r['konum'])[:40]}", axis=1),
            hovertemplate="%{text}<extra></extra>",
        ))
        # Erzincan merkezi
        fig_mig.add_trace(go.Scatter(
            x=[ERZ_LON], y=[ERZ_LAT], mode="markers+text",
            marker=dict(size=14, color="#ff3333", symbol="star"),
            text=["ERZ"], textposition="top right",
            textfont=dict(color="#ff3333", size=10),
            name="Erzincan", showlegend=False,
            hoverinfo="skip",
        ))
        fig_mig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TEXT, size=11),
            height=340, margin=dict(t=10, b=40, l=55, r=80),
            xaxis=dict(title=dict(text="Boylam", font=dict(color=TEXT)),
                       gridcolor=GRID, tickfont=dict(color=TEXT)),
            yaxis=dict(title=dict(text="Enlem", font=dict(color=TEXT)),
                       gridcolor=GRID, tickfont=dict(color=TEXT),
                       scaleanchor="x", scaleratio=1),
            hovermode="closest",
        )
        st.plotly_chart(fig_mig, use_container_width=True,
                        config={"displayModeBar": False, "displaylogo": False})

    with col_mig2:
        st.markdown('<div class="chart-title">📉 Derinlik Göçü (Zaman)</div>', unsafe_allow_html=True)
        st.caption("Derinlik zamanla azalıyorsa (yukarı göç) stres/sıvı yükselimi olabilir — öncü örüntü.")

        sci_df_dep = sci_df.dropna(subset=["derinlik"]).copy()
        fig_dep_mig = go.Figure()
        fig_dep_mig.add_trace(go.Scatter(
            x=sci_df_dep["zaman"], y=sci_df_dep["derinlik"],
            mode="markers",
            marker=dict(
                size=sci_df_dep["buyukluk"].apply(lambda m: max(5, m * 4.5)),
                color=sci_df_dep["buyukluk"],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(
                    title=dict(text="M", font=dict(color=TEXT, size=10)),
                    tickfont=dict(color=TEXT, size=9),
                    thickness=12, len=0.65,
                ),
                opacity=0.85,
                line=dict(width=0.8, color="rgba(255,255,255,0.3)" if DARK else "rgba(0,0,0,0.2)"),
            ),
            hovertemplate="<b>%{y:.1f} km</b><br>%{x}<extra></extra>",
        ))
        # Trend çizgisi
        if len(sci_df_dep) >= 10:
            x_num = (sci_df_dep["zaman"] - sci_df_dep["zaman"].min()).dt.total_seconds()
            coeffs = np.polyfit(x_num, sci_df_dep["derinlik"], 1)
            trend_y = coeffs[0] * x_num + coeffs[1]
            yön = "▼ Derinleşiyor" if coeffs[0] > 0 else "▲ Yüzeye yaklaşıyor"
            fig_dep_mig.add_trace(go.Scatter(
                x=sci_df_dep["zaman"], y=trend_y,
                mode="lines", name=f"Trend ({yön})",
                line=dict(color="#ef5350" if coeffs[0] < 0 else "#66bb6a",
                          width=2.5, dash="dash"),
            ))
        fig_dep_mig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TEXT, size=11),
            height=340, margin=dict(t=10, b=40, l=60, r=80),
            xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
            yaxis=dict(
                title=dict(text="Derinlik (km) — aşağı artar", font=dict(color=TEXT)),
                gridcolor=GRID, tickfont=dict(color=TEXT),
                autorange="reversed",
            ),
            legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)",
                        x=0, y=1.08, orientation="h"),
            hovermode="closest",
        )
        st.plotly_chart(fig_dep_mig, use_container_width=True,
                        config={"displayModeBar": False, "displaylogo": False})
else:
    st.info("Bilimsel analiz için en az 20 deprem gerekli. Zaman aralığını veya yarıçapı genişletin.")

# ════════════════════════════════════════════════════════════════
# DENEYSEL AKADEMİK ANALİZLER
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="chart-title">🔭 Deneysel Akademik Analizler</div>', unsafe_allow_html=True)
st.caption(
    "Zaliapin-Ben-Zion η-değeri (2013) · Sobolev-Tyupkin RTL Sessizlik (1997) · "
    "Bowman AMR Güç Yasası (1998) · Uzamsal b-Değeri Haritası. "
    "Bu analizler peer-reviewed seismoloji literatüründen alınan yöntemlerdir."
)

exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
    "η Kümeleme (Zaliapin-BZ)", "RTL Sessizlik (Sobolev)", "AMR Güç Yasası (Bowman)", "Uzamsal b-Haritası"
])

exp_df = df.dropna(subset=["buyukluk","lat","lon","derinlik"]).copy()
exp_df = exp_df.sort_values("zaman").reset_index(drop=True)

# ── ORTAK: b-değeri (MLE) ───────────────────────────────────────
def calc_b_mle(magnitudes, Mc=None):
    mags = np.array(magnitudes, dtype=float)
    if Mc is None:
        Mc = float(np.percentile(mags, 15))
    above = mags[mags >= Mc]
    if len(above) < 5:
        return 1.0, Mc
    mean_m = above.mean()
    if mean_m <= Mc:
        return 1.0, Mc
    return math.log10(math.e) / (mean_m - Mc), Mc

# ─────────────────────────────────────────────────────────────────
# TAB 1 — η (Zaliapin & Ben-Zion 2013)
# Normalize uzay-zaman nearest-neighbor mesafesi
# η_ij = t_ij × r_ij^(d/b) × 10^(-b×m_i/2)
# ─────────────────────────────────────────────────────────────────
with exp_tab1:
    st.markdown("**Zaliapin & Ben-Zion (2013) — Deprem Kümeleme Analizi**")
    st.markdown(
        "Her deprem için en yakın 'ebeveyn' deprem hesaplanır: "
        "`η = Δt × r^(d/b) × 10^(−b·m/2)`. "
        "Log(η) histogramı bimodal olduğunda — sol tepe = artçı/öncü kümeler, "
        "sağ tepe = bağımsız depremler. Bu ayrım klasik yöntemlerden çok daha hassastır."
    )

    if len(exp_df) >= 15:
        b_eta, Mc_eta = calc_b_mle(exp_df["buyukluk"].tolist())
        d_frac = 1.6  # Türkiye için tipik fraktal boyut

        eta_col1, eta_col2 = st.columns([1, 1])
        with eta_col1:
            st.metric("Hesaplanan b", f"{b_eta:.3f}")
            st.metric("Mc (tamamlanma)", f"M{Mc_eta:.2f}")
            st.metric("Fraktal boyut d", f"{d_frac}")

        with st.spinner("η değerleri hesaplanıyor..."):
            eta_list, log_t_list, log_r_list = [], [], []
            n = min(len(exp_df), 400)  # performans sınırı
            sub = exp_df.iloc[:n].reset_index(drop=True)
            for j in range(1, n):
                t_j = sub["zaman"].iloc[j]
                min_eta = np.inf
                best_t, best_r = np.nan, np.nan
                for i in range(j):
                    m_i = sub["buyukluk"].iloc[i]
                    dt_yr = (t_j - sub["zaman"].iloc[i]).total_seconds() / (365.25*86400)
                    if dt_yr <= 0:
                        continue
                    dr = haversine(sub["lat"].iloc[i], sub["lon"].iloc[i],
                                   sub["lat"].iloc[j], sub["lon"].iloc[j])
                    dr = max(dr, 0.1)
                    eta = dt_yr * (dr**(d_frac/b_eta)) * (10**(-b_eta*m_i/2))
                    if eta < min_eta:
                        min_eta, best_t, best_r = eta, dt_yr, dr
                if np.isfinite(min_eta) and min_eta > 0:
                    eta_list.append(math.log10(min_eta))
                    log_t_list.append(math.log10(max(best_t, 1e-10)))
                    log_r_list.append(math.log10(max(best_r, 0.1)))

        if eta_list:
            eta_arr = np.array(eta_list)
            # Eşik: histogram çukuru (yaklaşık medyan - 0.5 std)
            eta_thresh = float(np.percentile(eta_arr, 35))

            col_eh, col_es = st.columns(2)
            with col_eh:
                fig_eta_h = go.Figure(go.Histogram(
                    x=eta_arr, nbinsx=50,
                    marker_color="#64b5f6", opacity=0.8,
                    name="log(η)",
                ))
                fig_eta_h.add_vline(x=eta_thresh,
                    line=dict(color="#E53935", width=2, dash="dash"))
                fig_eta_h.add_annotation(
                    x=eta_thresh, y=0.95,
                    text=f"  Eşik η={eta_thresh:.1f}",
                    showarrow=False, yref="paper",
                    font=dict(color="#E53935", size=10),
                )
                fig_eta_h.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=BG2,
                    font=dict(color=TEXT, size=11), height=300,
                    margin=dict(t=30, b=40, l=55, r=20),
                    title=dict(text="log(η) Dağılımı — Bimodal = iki popülasyon",
                               font=dict(color=TEXT, size=11)),
                    xaxis=dict(title=dict(text="log₁₀(η)", font=dict(color=TEXT)),
                               gridcolor=GRID, tickfont=dict(color=TEXT)),
                    yaxis=dict(title=dict(text="Sayı", font=dict(color=TEXT)),
                               gridcolor=GRID, tickfont=dict(color=TEXT)),
                )
                st.plotly_chart(fig_eta_h, use_container_width=True,
                                config={"displayModeBar": False, "displaylogo": False})

            with col_es:
                clustered = eta_arr < eta_thresh
                colors_eta = np.where(clustered,
                    "#E53935" if DARK else "#c62828",
                    "#64b5f6" if DARK else "#1565c0")
                fig_eta_s = go.Figure()
                for label, mask, col in [
                    ("Tetiklenmiş (küme)", clustered, "#E53935"),
                    ("Bağımsız (arka plan)", ~clustered, "#64b5f6"),
                ]:
                    fig_eta_s.add_trace(go.Scatter(
                        x=np.array(log_t_list)[mask],
                        y=np.array(log_r_list)[mask],
                        mode="markers", name=label,
                        marker=dict(size=5, color=col, opacity=0.7),
                    ))
                fig_eta_s.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=BG2,
                    font=dict(color=TEXT, size=11), height=300,
                    margin=dict(t=30, b=40, l=55, r=20),
                    title=dict(text="log(Δt) – log(r) Uzayı",
                               font=dict(color=TEXT, size=11)),
                    xaxis=dict(title=dict(text="log₁₀(Δt [yıl])", font=dict(color=TEXT)),
                               gridcolor=GRID, tickfont=dict(color=TEXT)),
                    yaxis=dict(title=dict(text="log₁₀(r [km])", font=dict(color=TEXT)),
                               gridcolor=GRID, tickfont=dict(color=TEXT)),
                    legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)",
                                x=0, y=1.08, orientation="h"),
                )
                st.plotly_chart(fig_eta_s, use_container_width=True,
                                config={"displayModeBar": False, "displaylogo": False})

            n_clust = int(clustered.sum())
            n_bg = int((~clustered).sum())
            st.info(
                f"**Tetiklenmiş depremler:** {n_clust} (%{100*n_clust/len(eta_arr):.0f})  |  "
                f"**Bağımsız arka plan:** {n_bg} (%{100*n_bg/len(eta_arr):.0f})  |  "
                f"Eşik: log(η) = {eta_thresh:.2f}"
            )
        else:
            st.warning("η hesabı için yeterli event çifti oluşturulamadı.")
    else:
        st.info("η analizi için en az 15 deprem gerekli.")

# ─────────────────────────────────────────────────────────────────
# TAB 2 — RTL (Sobolev & Tyupkin 1997)
# Sismik sessizlik anomali tespiti
# ─────────────────────────────────────────────────────────────────
with exp_tab2:
    st.markdown("**Sobolev & Tyupkin (1997) — RTL Sismik Sessizlik Algoritması**")
    st.markdown(
        "Bölge-Zaman-Uzunluk ağırlıklı sismisiyet oranı hesaplanır. "
        "Normalize Z-skoru **–2 altına** düştüğünde istatistiksel sessizlik anlamına gelir. "
        "Büyük depremlerin %60–80'i öncesinde RTL < –2 gözlemlendi (literatür)."
    )

    if len(exp_df) >= 20:
        rtl_r0 = st.slider("r₀ (km) — mekansal ağırlık uzunluğu",
                           30, 300, 100, 10, key="rtl_r0")
        rtl_t0 = st.slider("t₀ (gün) — zamansal ağırlık uzunluğu",
                           30, 730, 180, 30, key="rtl_t0")

        with st.spinner("RTL hesaplanıyor..."):
            # Rupture uzunluğu: Wells & Coppersmith 1994
            exp_df["L_km"] = exp_df["buyukluk"].apply(
                lambda m: max(0.1, 10**(-2.44 + 0.59*m))
            )
            rtl_times, rtl_scores = [], []
            step = max(1, len(exp_df) // 80)  # max 80 nokta
            for idx in range(10, len(exp_df), step):
                t_ref = exp_df["zaman"].iloc[idx]
                past = exp_df.iloc[:idx]
                score = 0.0
                for _, ev in past.iterrows():
                    r = haversine(ERZ_LAT, ERZ_LON, ev["lat"], ev["lon"])
                    dt_days = (t_ref - ev["zaman"]).total_seconds() / 86400
                    if dt_days <= 0:
                        continue
                    score += (math.exp(-r / rtl_r0) *
                              math.exp(-dt_days / rtl_t0) /
                              ev["L_km"])
                rtl_times.append(t_ref)
                rtl_scores.append(score)

        if len(rtl_scores) >= 5:
            arr = np.array(rtl_scores)
            mu, sigma = arr.mean(), arr.std()
            rtl_z = ((arr - mu) / sigma).tolist() if sigma > 0 else (arr - mu).tolist()

            colors_rtl = [
                "#E53935" if z < -2 else
                "#FB8C00" if z < -1 else
                "#64b5f6"
                for z in rtl_z
            ]
            fig_rtl = go.Figure()
            fig_rtl.add_hrect(y0=-2, y1=min(rtl_z)-0.5,
                              fillcolor="rgba(229,57,53,0.10)", layer="below", line_width=0)
            fig_rtl.add_hline(y=-2, line=dict(color="#E53935", width=1.5, dash="dash"))
            fig_rtl.add_hline(y=-1, line=dict(color="#FB8C00", width=1, dash="dot"))
            fig_rtl.add_annotation(
                x=rtl_times[-1], y=-2,
                text="  RTL < –2: Sessizlik Anomalisi",
                showarrow=False, font=dict(color="#E53935", size=10), xanchor="left",
            )
            fig_rtl.add_trace(go.Scatter(
                x=rtl_times, y=rtl_z,
                mode="lines+markers",
                line=dict(color="#90caf9", width=2),
                marker=dict(size=6, color=colors_rtl,
                            line=dict(width=1, color="rgba(255,255,255,0.3)" if DARK else "rgba(0,0,0,0.2)")),
                hovertemplate="<b>RTL Z = %{y:.2f}</b><br>%{x}<extra></extra>",
                name="RTL Z-skoru",
            ))
            # M4+ olayları işaretle
            big_ev = exp_df[exp_df["buyukluk"] >= 4.0]
            for _, ev in big_ev.iterrows():
                fig_rtl.add_vline(x=ev["zaman"],
                    line=dict(color="#7B1FA2", width=1.5, dash="dot"))
            if not big_ev.empty:
                fig_rtl.add_trace(go.Scatter(
                    x=big_ev["zaman"],
                    y=[min(rtl_z)] * len(big_ev),
                    mode="markers", name="M4+ olaylar",
                    marker=dict(size=10, color="#7B1FA2", symbol="triangle-up"),
                    hovertemplate="M%{text}<extra></extra>",
                    text=big_ev["buyukluk"].apply(lambda m: f"{m:.1f}"),
                ))
            fig_rtl.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2,
                font=dict(color=TEXT, size=12), height=420,
                margin=dict(t=20, b=50, l=60, r=30),
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                yaxis=dict(
                    title=dict(text="RTL Z-skoru (σ)", font=dict(color=TEXT)),
                    gridcolor=GRID, tickfont=dict(color=TEXT),
                ),
                legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)",
                            x=0, y=1.06, orientation="h"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_rtl, use_container_width=True,
                            config={"displayModeBar": False, "displaylogo": False})

            anomaly_periods = sum(1 for z in rtl_z if z < -2)
            st.info(
                f"**Sessizlik anomalisi (RTL < –2):** {anomaly_periods} nokta / {len(rtl_z)} toplam  |  "
                f"r₀={rtl_r0} km · t₀={rtl_t0} gün  |  "
                f"Mor dikey çizgiler = M4+ olaylar"
            )
        else:
            st.warning("RTL için yeterli zaman noktası oluşturulamadı.")
    else:
        st.info("RTL için en az 20 deprem gerekli.")

# ─────────────────────────────────────────────────────────────────
# TAB 3 — AMR (Bowman et al. 1998)
# Accelerating Moment Release — güç yasası fit
# C(t) = A + B·(tf − t)^m  →  m < 1 ivcelenme
# ─────────────────────────────────────────────────────────────────
with exp_tab3:
    st.markdown("**Bowman et al. (1998) — Accelerating Moment Release (AMR)**")
    st.markdown(
        "Kümülatif Benioff zorlanmasına `C(t) = A + B·(tₓ − t)^m` güç yasası fit edilir. "
        "**m < 1** → ivcelenen yayılım, büyük deprem yakın. "
        "**m ≈ 1** → lineer (stabil). **m > 1** → yavaşlama. "
        "tₓ = tahmini kritik zaman (potansiyel kırılma anı)."
    )

    if len(exp_df) >= 20:
        amr_df = exp_df.sort_values("zaman").copy()
        amr_df["benioff"] = amr_df["buyukluk"].apply(
            lambda m: math.sqrt(10**(1.5*m))
        )
        amr_df["cum_ben"] = amr_df["benioff"].cumsum()
        C_max = amr_df["cum_ben"].max()
        if C_max > 0:
            amr_df["C_norm"] = amr_df["cum_ben"] / C_max
        else:
            amr_df["C_norm"] = amr_df["cum_ben"]

        t0_amr = amr_df["zaman"].iloc[0]
        t_days = ((amr_df["zaman"] - t0_amr)
                  .dt.total_seconds() / 86400).values
        C = amr_df["C_norm"].values
        T_obs = t_days[-1]

        # Grid search: tf ∈ [T_obs*1.05, T_obs*5], m ∈ [0.1, 2.0]
        best_rmse, best_m, best_tf, best_fitted = np.inf, 1.0, T_obs*1.5, C
        for tf_mult in np.linspace(1.05, 4.0, 25):
            tf = T_obs * tf_mult
            X_vals = (tf - t_days)
            X_vals = np.maximum(X_vals, 1e-6)
            for m_try in np.linspace(0.1, 1.9, 30):
                X = X_vals ** m_try
                # Least squares: C = A + B*X  →  normal equations
                mat = np.column_stack([np.ones_like(X), X])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(mat, C, rcond=None)
                    A_fit, B_fit = coeffs
                    fitted = A_fit + B_fit * X
                    rmse = np.sqrt(np.mean((C - fitted)**2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_m = m_try
                        best_tf = tf
                        best_fitted = fitted.copy()
                except Exception:
                    pass

        tf_date = t0_amr + timedelta(days=float(best_tf))
        m_interp = ("🔴 İvceleniyor — kritik noktaya yaklaşım" if best_m < 0.8
                    else "🟡 Lineer yayılım — stabil" if best_m < 1.2
                    else "🟢 Yavaşlıyor — enerji dağılıyor")

        col_amr1, col_amr2, col_amr3 = st.columns(3)
        col_amr1.metric("m (güç yasası üssü)", f"{best_m:.3f}")
        col_amr2.metric("tₓ (tahmini kritik)", tf_date.strftime("%d.%m.%Y") if best_tf < T_obs * 20 else "Belirsiz")
        col_amr3.metric("RMSE", f"{best_rmse:.4f}")
        st.info(m_interp)

        fig_amr = go.Figure()
        fig_amr.add_trace(go.Scatter(
            x=amr_df["zaman"], y=C,
            mode="lines", name="Gözlenen Benioff",
            line=dict(color="#ffb74d", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(255,183,77,0.10)" if DARK else "rgba(255,183,77,0.18)",
        ))
        fig_amr.add_trace(go.Scatter(
            x=amr_df["zaman"], y=best_fitted,
            mode="lines", name=f"AMR fit (m={best_m:.2f})",
            line=dict(
                color="#E53935" if best_m < 0.8 else "#66bb6a",
                width=2, dash="dash"
            ),
        ))
        # Kritik zaman çizgisi
        if tf_date > amr_df["zaman"].max():
            fig_amr.add_vline(x=tf_date,
                line=dict(color="#E53935", width=1.5, dash="dot"))
            fig_amr.add_annotation(
                x=tf_date, y=1.0,
                text=f"  tₓ={tf_date.strftime('%d.%m')}",
                showarrow=False, yref="paper",
                font=dict(color="#E53935", size=10), xanchor="left",
            )
        fig_amr.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TEXT, size=12), height=400,
            margin=dict(t=20, b=50, l=65, r=30),
            xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
            yaxis=dict(
                title=dict(text="Kümülatif Benioff (normalize)", font=dict(color=TEXT)),
                gridcolor=GRID, tickfont=dict(color=TEXT),
            ),
            legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)",
                        x=0, y=1.06, orientation="h"),
        )
        st.plotly_chart(fig_amr, use_container_width=True,
                        config={"displayModeBar": False, "displaylogo": False})
        st.caption(
            "⚠️ AMR tₓ tahmini istatistiksel bir fit olup kesin deprem tahmini değildir. "
            "Akademik referans: Bowman et al., JGR 1998."
        )
    else:
        st.info("AMR için en az 20 deprem gerekli.")

# ─────────────────────────────────────────────────────────────────
# TAB 4 — Uzamsal b-Değeri Haritası
# ─────────────────────────────────────────────────────────────────
with exp_tab4:
    st.markdown("**Uzamsal b-Değeri Haritası — Stres Zonları**")
    st.markdown(
        "Bölge grid'e bölünür, her hücrede Gutenberg-Richter b-değeri MLE ile hesaplanır. "
        "**Düşük b (kırmızı) = yüksek gerilme alanı**, kırılma potansiyeli yüksek. "
        "**Yüksek b (mavi) = heterojen veya düşük gerilme** ortamı."
    )

    if len(exp_df) >= 25:
        bg_n = st.slider("Grid çözünürlüğü (NxN)", 8, 20, 12, 1, key="bg_n")
        bg_sr = st.slider("Hücre arama yarıçapı (km)", 20, 150, 60, 10, key="bg_sr")
        bg_min = st.slider("Min olay sayısı/hücre", 5, 20, 8, 1, key="bg_min")

        with st.spinner("Uzamsal b-değerleri hesaplanıyor..."):
            deg = 1 / 111
            margin_deg = radius_km * deg * 0.75
            lats_g = np.linspace(ERZ_LAT - margin_deg, ERZ_LAT + margin_deg, bg_n)
            lons_g = np.linspace(ERZ_LON - margin_deg * 1.3, ERZ_LON + margin_deg * 1.3, bg_n)

            Mc_g = float(exp_df["buyukluk"].quantile(0.15))
            df_mc = exp_df[exp_df["buyukluk"] >= Mc_g]

            b_grid  = np.full((bg_n, bg_n), np.nan)
            n_grid  = np.zeros((bg_n, bg_n), dtype=int)

            for i, lat_g in enumerate(lats_g):
                for j, lon_g in enumerate(lons_g):
                    dists = df_mc.apply(
                        lambda r: haversine(lat_g, lon_g, r["lat"], r["lon"]), axis=1
                    )
                    sub_g = df_mc[dists <= bg_sr]
                    if len(sub_g) < bg_min:
                        continue
                    mean_m = sub_g["buyukluk"].mean()
                    if mean_m <= Mc_g:
                        continue
                    b_val = math.log10(math.e) / (mean_m - Mc_g)
                    b_grid[i, j] = np.clip(b_val, 0.3, 3.0)
                    n_grid[i, j] = len(sub_g)

        if not np.all(np.isnan(b_grid)):
            fig_bmap = go.Figure()

            # b-değeri heatmap (interpolated)
            fig_bmap.add_trace(go.Heatmap(
                x=lons_g, y=lats_g, z=b_grid,
                colorscale="RdBu",  # Kırmızı=düşük b, Mavi=yüksek b
                zmin=0.5, zmax=2.0,
                reversescale=False,
                opacity=0.75,
                colorbar=dict(
                    title=dict(text="b-değeri", font=dict(color=TEXT)),
                    tickfont=dict(color=TEXT),
                    thickness=14,
                ),
                hovertemplate="lon=%{x:.2f} lat=%{y:.2f}<br>b=%{z:.2f}<extra></extra>",
            ))

            # Deprem noktaları üzerine
            fig_bmap.add_trace(go.Scatter(
                x=exp_df["lon"], y=exp_df["lat"],
                mode="markers", name="Depremler",
                marker=dict(
                    size=exp_df["buyukluk"].apply(lambda m: max(4, m*4)),
                    color=exp_df["buyukluk"].apply(mag_color),
                    opacity=0.6,
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                ),
                hovertemplate="M%{text}<extra></extra>",
                text=exp_df["buyukluk"].apply(lambda m: f"{m:.1f}"),
            ))

            # Erzincan
            fig_bmap.add_trace(go.Scatter(
                x=[ERZ_LON], y=[ERZ_LAT],
                mode="markers+text", name="Erzincan",
                marker=dict(size=14, color="#ff3333", symbol="star"),
                text=["ERZ"], textposition="top right",
                textfont=dict(color="#ff3333", size=10),
                hoverinfo="skip",
            ))

            fig_bmap.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2,
                font=dict(color=TEXT, size=11), height=500,
                margin=dict(t=10, b=40, l=60, r=20),
                xaxis=dict(title=dict(text="Boylam", font=dict(color=TEXT)),
                           gridcolor=GRID, tickfont=dict(color=TEXT),
                           scaleanchor="y", scaleratio=1),
                yaxis=dict(title=dict(text="Enlem", font=dict(color=TEXT)),
                           gridcolor=GRID, tickfont=dict(color=TEXT)),
                legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)",
                            x=0, y=1.06, orientation="h"),
                hovermode="closest",
            )
            st.plotly_chart(fig_bmap, use_container_width=True,
                            config={"displayModeBar": False, "displaylogo": False})

            valid_b = b_grid[~np.isnan(b_grid)]
            low_b_pct = float(np.mean(valid_b < 0.8) * 100)
            st.info(
                f"Mc ≈ M{Mc_g:.1f} | Hücre arama r={bg_sr} km | "
                f"Dolu hücre: {(~np.isnan(b_grid)).sum()}/{bg_n*bg_n} | "
                f"Düşük-b (< 0.8) bölge: %{low_b_pct:.0f} — yüksek gerilme zonu"
            )
        else:
            st.warning("Yeterli veri yok. Grid yarıçapını veya min olay eşiğini düşürün.")
    else:
        st.info("Uzamsal b-haritası için en az 25 deprem gerekli.")

# ─── Tam tablo ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander(f"📋 Tum Depremler — {len(df)} Kayit"):
    show = df[["zaman_str","buyukluk","sinif","derinlik","uzaklik_km","konum","kaynak"]].copy()
    show.columns = ["Zaman","Buyukluk","Sinif","Derinlik (km)","Uzaklik (km)","Konum","Kaynak"]
    st.dataframe(show, use_container_width=True, hide_index=True,
                 column_config={"Buyukluk": st.column_config.NumberColumn(format="M%.1f")})
    st.download_button("Indir (CSV)",
                       data=show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"erzincan_deprem_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:{SUBTEXT};font-size:0.7rem;
            margin-top:0.8rem;padding:0.5rem;background:{BG3};
            border-radius:8px;border:1px solid {BORDER}">
  USGS · EMSC · AFAD · Kandilli · GFZ Potsdam · IRIS/SAGE &nbsp;|&nbsp;
  Her {refresh_s} saniyede otomatik yenileme &nbsp;|&nbsp;
  Uydu haritasi: ESRI World Imagery &nbsp;|&nbsp;
  {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
</div>
""", unsafe_allow_html=True)
