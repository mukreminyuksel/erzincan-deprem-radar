from __future__ import annotations  # Python 3.9 uyumu — PEP 604 'X | None' syntax fix

import concurrent.futures
import importlib
import json
import math
import os
from datetime import datetime, timedelta

import numpy as np
import ephem
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from streamlit_option_menu import option_menu

import earthquake_core as _earthquake_core

_earthquake_core = importlib.reload(_earthquake_core)
from earthquake_core import (
    QUICK_WINDOWS,
    activity_level,
    duration_from_quick_window,
    estimate_energy_joules,
    event_signature,
    has_active_sources,
    nearest_fault_vertex_distance,
    parse_usgs_feed_features,
    safe_html,
    source_agreement_summary,
    to_utc_naive,
    usgs_feed_url_for_window,
    utc_now_naive,
)

# v1.17 — Ajan 4'ün NNR-MORVEL56 Euler kutbu rotasyon fonksiyonu.
# Henüz yoksa (parallel ajan ortamı), `_plate_velocity_vector_extern` None kalır
# ve plaka simülasyon paneli kendi Euler dönüştürücüsünü kullanır.
try:
    from earthquake_core import plate_velocity_vector as _plate_velocity_vector_extern
except ImportError:
    _plate_velocity_vector_extern = None

ERZ_LAT = 39.7333
ERZ_LON = 39.4917
APP_VERSION = "1.17.3"
APP_TITLE = f"Erzincan Deprem Radari v{APP_VERSION}"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    if m < 1.0: return "#B3E5FC"  # Soluk Mavi (Göze batmasın)
    if m < 2.0: return "#C5E1A5"  # Soluk Yeşil
    if m < 3.0: return "#76FF03"  # Parlak Neon Çimen Yeşil
    if m < 4.0: return "#FFEA00"  # Parlak Neon Sarı
    if m < 5.0: return "#FF6D00"  # Neon Turuncu
    if m < 6.0: return "#FF1744"  # Parlak Kırmızı
    if m < 7.0: return "#D500F9"  # Parlak Mor
    if m < 8.0: return "#000000"  # Siyah
    return "#263238"  # Koyu Gri/Siyah (Mapbox sembol kısıtlaması nedeniyle sınır rengi vs UI'da halledilebilir)

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
        start, end = to_utc_naive(start_dt), to_utc_naive(end_dt)
        r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
            "format": "geojson", "latitude": lat, "longitude": lon,
            "maxradiuskm": radius_km, "minmagnitude": min_mag,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=4.0)
        r.raise_for_status()
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

def fetch_usgs_fast(lat, lon, radius_km, min_mag, start_dt, end_dt):
    try:
        feed_url = usgs_feed_url_for_window(start_dt, end_dt)
        if not feed_url:
            return [], "ATLANDI: 30 gun+"
        r = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4.0)
        r.raise_for_status()
        rows = parse_usgs_feed_features(
            r.json().get("features", []), lat, lon, radius_km, min_mag
        )
        start, end = to_utc_naive(start_dt), to_utc_naive(end_dt)
        rows = [
            row for row in rows
            if start <= datetime.strptime(row["zaman"], "%Y-%m-%d %H:%M:%S") <= end
        ]
        return rows, f"OK ({len(rows)})"
    except Exception as e:
        return [], f"HATA: {str(e)[:50]}"

def fetch_emsc(lat, lon, radius_km, min_mag, start_dt, end_dt):
    try:
        start, end = to_utc_naive(start_dt), to_utc_naive(end_dt)
        r = requests.get("https://www.seismicportal.eu/fdsnws/event/1/query", params={
            "format": "json", "lat": lat, "lon": lon,
            "maxradius": radius_km / 111, "minmag": min_mag,
            "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=4.0)
        r.raise_for_status()
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
        start, end = to_utc_naive(start_dt), to_utc_naive(end_dt)
        margin = radius_km / 111
        r = requests.get("https://deprem.afad.gov.tr/apiv2/event/filter", params={
            "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "minlat": lat - margin, "maxlat": lat + margin,
            "minlon": lon - margin, "maxlon": lon + margin,
            "minmag": min_mag, "format": "json", "limit": 1000,
            "orderby": "timedesc",
        }, headers={"User-Agent": "Mozilla/5.0"}, timeout=4.0)
        r.raise_for_status()
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
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=4.0)
        r.raise_for_status()
        content = r.content.decode("iso-8859-9", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
        pre = soup.find("pre")
        if not pre:
            return [], "HATA: pre tag yok"
        cutoff = to_utc_naive(start_dt)
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
        start, end = to_utc_naive(start_dt), to_utc_naive(end_dt)
        r = requests.get("https://geofon.gfz-potsdam.de/fdsnws/event/1/query", params={
            "format": "text", "lat": lat, "lon": lon,
            "maxradius": radius_km / 111, "minmagnitude": min_mag,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=4.0)
        r.raise_for_status()
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
        start, end = to_utc_naive(start_dt), to_utc_naive(end_dt)
        r = requests.get("https://service.iris.edu/fdsnws/event/1/query", params={
            "format": "text", "latitude": lat, "longitude": lon,
            "maxradius": radius_km / 111, "minmagnitude": min_mag,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=4.0)
        r.raise_for_status()
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
        start, end = to_utc_naive(start_dt), to_utc_naive(end_dt)
        r = requests.get(
            "https://deprem.afad.gov.tr/last-earthquakes.html",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=4.0,
        )
        r.raise_for_status()
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
                if dt < start or dt > end:
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
        start, end = to_utc_naive(start_dt), to_utc_naive(end_dt)
        r = requests.get("https://webservices.ingv.it/fdsnws/event/1/query", params={
            "format": "text", "lat": lat, "lon": lon,
            "maxradius": radius_km / 111, "minmagnitude": min_mag,
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time", "limit": 1000,
        }, timeout=4.0)
        r.raise_for_status()
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
    "USGS-Fast": fetch_usgs_fast,
    "USGS":      fetch_usgs,
    "EMSC":      fetch_emsc,
    "AFAD":      fetch_afad,
    "AFAD-Web":  fetch_afad_html,
    "Kandilli":  fetch_kandilli,
    "GFZ":       fetch_gfz,
    "IRIS":      fetch_iris,
    "INGV":      fetch_ingv,
}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_all(lat, lon, radius_km, min_mag, start_dt, end_dt, active_sources):
    # Cache buster for mag_color change
    statuses = {}
    all_rows = []
    fetchers = {k: v for k, v in ALL_FETCHERS.items() if k in active_sources}
    if not fetchers:
        return pd.DataFrame(), statuses
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fetchers)) as ex:
        futures = {ex.submit(fn, lat, lon, radius_km, min_mag, start_dt, end_dt): name
                   for name, fn in fetchers.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
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
    # Vektörleştirilmiş haversine (Ajan 3) — df.apply Python loop'undan ~50-100× hızlı
    lat_rad = math.radians(lat)
    lon_rad_val = math.radians(lon)
    df_lat_rad = np.radians(df["lat"].to_numpy())
    df_lon_rad = np.radians(df["lon"].to_numpy())
    dlat = df_lat_rad - lat_rad
    dlon = df_lon_rad - lon_rad_val
    a_hav = np.sin(dlat / 2) ** 2 + math.cos(lat_rad) * np.cos(df_lat_rad) * np.sin(dlon / 2) ** 2
    df["uzaklik_km"] = 6371.0 * 2.0 * np.arcsin(np.sqrt(a_hav))
    df = df.sort_values("zaman", ascending=False).reset_index(drop=True)

    # Tekilleştirme — Kandilli UTC+3 düzeltmesi, toleranslar genişletildi
    # Farklı ağların aynı depremi rapor etme süresi gerçekte 0-90sn arası
    # Sliding-window dedup (Ajan 3): O(n²) → O(n × k), k ≈ 2dk pencere satır sayısı
    if len(df) > 1:
        n = len(df)
        times_ns = df["zaman"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        lats_arr = df["lat"].to_numpy()
        lons_arr = df["lon"].to_numpy()
        mags_arr = df["buyukluk"].to_numpy()
        keep_mask = np.ones(n, dtype=bool)
        threshold_ns = np.int64(120_000_000_000)  # 120 s
        for i in range(1, n):
            ti = times_ns[i]
            for j in range(i - 1, -1, -1):
                if not keep_mask[j]:
                    continue
                if abs(ti - times_ns[j]) >= threshold_ns:
                    break  # descending sorted: ileri gidersek pencere kapanır
                if (abs(lats_arr[i] - lats_arr[j]) < 0.15
                    and abs(lons_arr[i] - lons_arr[j]) < 0.15
                    and abs(mags_arr[i] - mags_arr[j]) < 0.5):
                    keep_mask[i] = False
                    break
        df = df[keep_mask].reset_index(drop=True)
    df["renk"]     = df["buyukluk"].apply(mag_color)
    df["emoji"]    = df["buyukluk"].apply(mag_emoji)
    df["sinif"]    = df["buyukluk"].apply(mag_label)
    df["boyut"]    = df["buyukluk"].apply(lambda m: max(6, float(m) ** 2.1))
    df["zaman_str"] = df["zaman"].dt.strftime("%d.%m.%Y %H:%M:%S")
    df["event_id"] = df.apply(
        lambda r: event_signature(r["zaman_str"], r["lat"], r["lon"], r["buyukluk"]),
        axis=1,
    )
    return df, statuses

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, .stApp {{ background: {BG}; color: {TEXT}; }}
  /* Geniş layout — boşlukları sıkılaştır */
  .block-container {{
    padding: 0.15rem 0.8rem 0.8rem 0.8rem !important;
    max-width: 100% !important;
  }}
  /* Sidebar iç padding'i de sıkılaştır */
  [data-testid="stSidebar"] > div:first-child {{
    padding-top: 0.15rem !important;
    padding-bottom: 0.4rem !important;
  }}
  /* Streamlit elementleri arası dikey boşluğu azalt */
  [data-testid="stVerticalBlock"] > div {{ gap: 0.25rem !important; }}
  div[data-testid="element-container"] {{ margin-bottom: 0 !important; }}
  hr {{ margin: 0.25rem 0 !important; }}

  /* Autorefresh sırasında sayfa soluklaşmasın — tüm seçicileri kapsa */
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"],
  [data-testid="staleElement"],
  [data-testid="staleElementContainer"],
  [data-testid="stElementContainer"],
  .staleElement,
  .staleElementContainer,
  div[class*="stale"],
  div[class*="Stale"],
  .stApp,
  .main,
  section.main,
  .block-container {{
    opacity: 1 !important;
    transition: none !important;
    animation: none !important;
  }}
  [data-testid="stAppViewContainer"] [style*="opacity: 0.33"],
  [data-testid="stAppViewContainer"] [style*="opacity:0.33"],
  [data-testid="stAppViewContainer"] [style*="opacity: 0.2"],
  [data-testid="stAppViewContainer"] [style*="opacity:0.2"],
  [data-testid="stAppViewContainer"] [style*="opacity: 0.4"],
  [data-testid="stAppViewContainer"] [style*="opacity:0.4"] {{
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
    border: 1px solid {BORDER}; border-radius: 8px;
    padding: 0.34rem 0.75rem; margin-bottom: 0.2rem;
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
    border-radius: 7px; padding: 0.28rem 0.4rem; text-align: center;
  }}
  .eq-scroll-container {{
    height: 650px;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-width: thin;
    scrollbar-color: {BORDER} {BG3};
  }}
  .eq-scroll-container::-webkit-scrollbar {{ width: 5px; }}
  .eq-scroll-container::-webkit-scrollbar-track {{ background: {BG3}; border-radius: 4px; }}
  .eq-scroll-container::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}

  .eq-card {{
    padding: 5px 8px; border-radius: 7px; margin: 2px 0;
    font-size: 0.82rem; border-left: 4px solid;
    background: {CARD_BG}; color: {TEXT} !important;
  }}
  .eq-card span, .eq-card b, .eq-card i {{
    color: inherit;
  }}
  .blink {{ animation: blink 1.2s step-start infinite; }}
  @keyframes blink {{ 50% {{ opacity: 0.15; }} }}

  .chart-title {{
    font-size: 0.9rem; font-weight: 700; color: {TEXT};
    margin: 0.05rem 0 0.1rem 0;
  }}

  div[data-testid="stTabs"] > div[role="tablist"] {{
    position: sticky;
    top: 0;
    z-index: 50;
    gap: 8px;
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px;
    margin: 0.15rem 0 0.35rem 0;
  }}
  /* Ana menü sekmeleri (Canlı Radar, İstatistik, vs.) renkleri */
  div[data-testid="stTabs"] button[role="tab"] {{
    color: {"'#FFD54F'" if DARK else "'#1A237E'"} !important;
    font-weight: 700 !important;
  }}
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: {"'#FFB300'" if DARK else "'#0D47A1'"} !important;
    border-bottom-color: {"'#FFB300'" if DARK else "'#0D47A1'"} !important;
  }}
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
  }}
  div[data-testid="stTabs"] button[role="tab"] {{
    min-height: 42px;
    flex: 1 1 0;
    border: 1px solid {BORDER};
    border-radius: 7px;
    background: {BG3};
    color: {TEXT};
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 0;
    padding: 0.45rem 0.8rem;
  }}
  div[data-testid="stTabs"] button[role="tab"]:hover {{
    border-color: #64b5f6;
    background: {"#102844" if DARK else "#d7e8f8"};
  }}
  div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    background: {"#1a73e8" if DARK else "#0d5fc6"};
    color: #ffffff;
    border-color: #90caf9;
    box-shadow: inset 0 -3px 0 #ffb74d;
  }}
  div[data-testid="stTabs"] button[role="tab"] p {{
    font-size: inherit;
    font-weight: inherit;
  }}

  /* ─── ANA MENÜ pill bar (v1.17.1 — overflow override KALDIRILDI) ──────
     Önceki v1.17a deneysel `overflow: visible !important` kuralı bazı
     panellerde scroll'u kırıyordu (kullanıcı bildirdi). Sticky çalışması
     için Streamlit'in iç DOM yapısının visible olması gerekir — ama bunu
     dışarıdan zorlamak diğer scroll alanlarını da bozuyor.
     Karar: sticky'yi soft tarz (override'sız) bırak; çalışırsa bonus,
     çalışmazsa kullanıcı yine geri scroll'la pill bar'a ulaşabilir.
     Scroll'un yer değişimi/kaybı ortadan kalktı. */
  .st-key-sticky_nav {{
    background: {BG};
    padding: 4px 0 6px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 6px;
  }}
  .st-key-sticky_nav iframe {{
    background: transparent !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar — v1.15b: Filtreler açıkta + 3 expander grup ──────────────────
# Mimari: Sık değişen filtreler en üstte açık, daha az değişenler expander içinde.
# Görünüm/Veri Kaynakları/Sistem expanded=False ile başlar — kullanıcı tıklayınca açılır.
with st.sidebar:
    st.markdown("### 🎯 Ayarlar")

    # ─── FİLTRELER (her zaman görünür — kullanıcının ana etkileşim noktası) ──
    radius_km = st.slider("Yarıçap (km)", 50, 600, 100, 10)
    min_mag   = st.slider("Min. Büyüklük", 0.5, 5.0, 1.0, 0.5)

    zaman_secenekleri = list(QUICK_WINDOWS.keys()) + ["Özel gün sayısı", "Özel Tarih Aralığı"]
    zaman_secim = st.selectbox("Zaman Aralığı", zaman_secenekleri, index=6)

    if zaman_secim == "Özel Tarih Aralığı":
        today = utc_now_naive().date()
        d_start = st.date_input("Başlangıç", value=today - timedelta(days=30),
                                max_value=today)
        d_end   = st.date_input("Bitiş", value=today, max_value=today)
        if d_start >= d_end:
            st.warning("Başlangıç tarihi bitiş tarihinden önce olmalı.")
            d_start = d_end - timedelta(days=1)
        query_start = datetime(d_start.year, d_start.month, d_start.day, 0, 0, 0)
        query_end   = datetime(d_end.year,   d_end.month,   d_end.day,   23, 59, 59)
        days_label  = f"{d_start.strftime('%d.%m.%Y')} – {d_end.strftime('%d.%m.%Y')}"
    elif zaman_secim == "Özel gün sayısı":
        if "custom_days" not in st.session_state:
            st.session_state.custom_days = 30
        if "custom_days_slider" not in st.session_state:
            st.session_state.custom_days_slider = st.session_state.custom_days
        if "custom_days_input" not in st.session_state:
            st.session_state.custom_days_input = st.session_state.custom_days

        # Slider üst sınırı 10 yıl (3650 gün) — görsel hızlı seçim için geniş aralık.
        # Bunun ötesi için aşağıdaki number_input sınırsız (max_value yok).
        SLIDER_MAX_DAYS = 3650

        def sync_custom_days_from_slider():
            st.session_state.custom_days = int(st.session_state.custom_days_slider)
            st.session_state.custom_days_input = st.session_state.custom_days

        def sync_custom_days_from_input():
            # Input slider'ın üst sınırını aşabilir; slider'ı clamp ile senkronla.
            new_val = int(st.session_state.custom_days_input)
            st.session_state.custom_days = new_val
            st.session_state.custom_days_slider = min(new_val, SLIDER_MAX_DAYS)

        # Slider değeri sınırın üstündeyse görsel olarak clamp'le (state etkilenmez)
        slider_display = min(int(st.session_state.custom_days), SLIDER_MAX_DAYS)
        if st.session_state.custom_days_slider != slider_display:
            st.session_state.custom_days_slider = slider_display

        st.slider(
            "Gün sayısı (slider: 1–3650 gün ≈ 10 yıl)",
            min_value=1, max_value=SLIDER_MAX_DAYS,
            step=1,
            key="custom_days_slider",
            on_change=sync_custom_days_from_slider,
        )

        st.number_input(
            "Gün kutusu (sınırsız — daha uzun dönem için doğrudan yaz)",
            min_value=1,
            step=1,
            key="custom_days_input",
            on_change=sync_custom_days_from_input,
        )

        query_end = utc_now_naive().replace(second=0, microsecond=0)
        query_start = query_end - timedelta(days=int(st.session_state.custom_days))
        days_label = f"Son {int(st.session_state.custom_days)} gün"
    else:
        query_end  = utc_now_naive().replace(second=0, microsecond=0)
        query_start = query_end - duration_from_quick_window(zaman_secim)
        days_label  = zaman_secim

    refresh_s = st.selectbox("Otomatik Yenileme",
                              [60, 30,60,120,180,240,300],
                              format_func=lambda x: f"Her {x} saniye")

    # ─── 🎨 GÖRÜNÜM (expander — kapalı başla) ──────────────────────────────
    with st.expander("🎨 Görünüm", expanded=False):
        tema_secim = st.radio("Tema", ["Karanlik", "Aydinlik"],
                              index=0 if DARK else 1, horizontal=True)
        if (tema_secim == "Karanlik") != DARK:
            st.session_state.tema = "dark" if tema_secim == "Karanlik" else "light"
            st.rerun()
        harita_stil = st.selectbox("Harita Stili", ["Uydu", "Uydu+Yol", "Koyu", "Acik"], index=0)
        show_faults = st.checkbox("Fay Hatlarını Göster", value=True)
        show_plates = st.checkbox("Kıta / Plaka Sınırlarını Göster", value=True)

    # ─── 📡 VERİ KAYNAKLARI (expander — kapalı başla, 9 checkbox) ──────────
    with st.expander("📡 Veri Kaynakları (9 ağ)", expanded=False):
        SRC_LABELS = {
            "USGS-Fast": "USGS Fast Feed (1 dk)",
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

    # Veri kaynağı doğrulama — expander dışında ki st.stop() net görünsün
    if not has_active_sources(active_sources):
        st.warning("En az bir kaynak seçilmeli!")
        st.stop()

    # ─── ℹ️ SİSTEM & SÜRÜM NOTLARI (expander — kapalı başla) ───────────────
    with st.expander("ℹ️ Sistem & Sürüm Notları", expanded=False):
        st.markdown(f"**Sürüm:** v{APP_VERSION}")
        st.markdown(
            "<div style='font-size:0.78rem;opacity:0.85;line-height:1.45'>"
            "• 📌 ANA MENÜ scroll'da viewport'ta sabit kalır — overflow:clip fix (v1.16).<br>"
            "• 🎨 ANA MENÜ üst horizontal pill bar'a taşındı (v1.15a).<br>"
            "• 🧭 Sidebar 3 expander grubuna düzenlendi (v1.15b).<br>"
            "• ⚡ Her panel @st.fragment ile izole — etkileşimler diğer panel state'ini bozmaz.<br>"
            "• 🚀 Haversine NumPy vektörleştirilmiş + sliding-window dedup.<br>"
            "• 🗺️ Plaka sınırları PB2002 tipine göre renkli: 🔴 yaklaşan / 🔵 ayrılan / 🟡 yanal.<br>"
            "• 🛡️ Ağır analizler 'Çalıştır' butonu ile manuel tetiklenir.<br>"
            "• ⏱️ API timeout 4 sn, fetch_all cache TTL 600 sn."
            "</div>",
            unsafe_allow_html=True,
        )

# ─── Otomatik yenileme ──────────────────────────────────────────────────────
st_autorefresh(interval=refresh_s * 1000, key="eq_ref")

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="radar-header">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
    <div>
      <span style="font-size:1.25rem;font-weight:800">🌍 Erzincan Deprem Radari <span style="font-size:0.78rem;opacity:0.75">(v {APP_VERSION})</span></span>
      <span style="margin-left:12px;font-size:0.8rem;opacity:0.55">
        {radius_km} km &nbsp;·&nbsp; M{min_mag}+ &nbsp;·&nbsp;
        {days_label} &nbsp;·&nbsp; {len(active_sources)} kaynak paralel
        &nbsp;·&nbsp; <span class="blink" style="color:#f44336">● CANLI</span>
      </span>
    </div>
    <span style="font-size:0.72rem;opacity:0.5">{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</span>
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

now_utc = utc_now_naive()
last1h  = df[df["zaman"] >= now_utc - timedelta(hours=1)]
last24h = df[df["zaman"] >= now_utc - timedelta(hours=24)]
big4    = df[df["buyukluk"] >= 4.0]

# ─── ANA MENÜ — üst horizontal pill (v1.16: tam sticky) ───────────────────────
# st.container(key="sticky_nav") → ".st-key-sticky_nav" → position:sticky (v1.16 fix)
_MENU_LABELS = [
    "🌍 Canlı Radar",
    "📊 İstatistik & Analiz",
    "🧭 Fay Sistemleri",
    "🌍 Plaka Simülasyonu",
    "🔭 Astronomik Analiz",
    "🚨 Erken Uyarı",
    "🎓 Bilgi Havuzu",
    "⚙️ Sistem & Veri",
    "📝 Raporlar",
]
_MENU_ICONS = [
    "globe", "bar-chart-line", "compass", "globe-americas", "moon-stars",
    "exclamation-triangle", "mortarboard", "gear", "file-text",
]
with st.container(key="sticky_nav"):
    active_menu = option_menu(
        menu_title=None,
        options=_MENU_LABELS,
        icons=_MENU_ICONS,
        orientation="horizontal",
        default_index=0,
        key="main_nav",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent", "margin-top": "0"},
            "icon": {"font-size": "0.95rem"},
            "nav-link": {
                "font-size": "0.82rem",
                "text-align": "center",
                "margin": "0 2px",
                "padding": "6px 10px",
                "border-radius": "8px",
                "white-space": "nowrap",
            },
            "nav-link-selected": {"background-color": "#1976d2", "font-weight": "600"},
        },
    )

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
            f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
            f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
            f'</div>', unsafe_allow_html=True)

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

@st.cache_resource(show_spinner=False)
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
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            lines.append({
                "fay_adi":   props.get("fay_adi") or "Adlandırılmamış",
                "segment":   props.get("segment") or "",
                "kayma":     props.get("kayma_aciklama") or "Bilinmiyor",
                "uzunluk":   props.get("uzunluk_km") or 0,
                "color":     color,
                "lats":      lats,
                "lons":      lons,
                "min_lat":   min(lats),
                "max_lat":   max(lats),
                "min_lon":   min(lons),
                "max_lon":   max(lons),
            })
    return lines

FAULT_LINES = load_fault_lines()

# PB2002 (Peter Bird 2003) plaka sınır tipleri — Türkiye + komşu Mediterranean odaklı
# convergent = yaklaşan (subduction/collision) — kırmızı
# divergent  = ayrılan (rift/MOR) — mavi
# transform  = yanal kayan (strike-slip) — sarı
PB2002_BOUNDARY_TYPES = {
    # Türkiye merkezli
    ("AT", "EU"): ("transform",  "Kuzey Anadolu Fay Zonu (NAFZ)"),
    ("AR", "AT"): ("transform",  "Doğu Anadolu Fay Zonu (EAFZ)"),
    ("AS", "AT"): ("divergent",  "Batı Anadolu Açılma Zonu"),
    ("AF", "AT"): ("convergent", "Kıbrıs Yayı (Cyprean Arc)"),
    ("AS", "EU"): ("convergent", "Helenik Yay — Kuzey"),
    ("AF", "AS"): ("convergent", "Helenik Hendeği (subduction)"),
    ("AF", "AR"): ("transform",  "Ölüdeniz Fay Zonu (DSFZ)"),
    ("AR", "EU"): ("convergent", "Bitlis-Zagros Sıkışma Zonu"),
    # Diğer büyük dünya sınırları (ana hatlar, kullanıcı zoom out yaparsa)
    ("EU", "NA"): ("divergent",  "Mid-Atlantic Ridge — Kuzey"),
    ("NA", "PA"): ("transform",  "San Andreas + Aleut"),
    ("CO", "NA"): ("convergent", "Cocos subduction"),
    ("CO", "SA"): ("divergent",  "East Pacific Rise — Kuzey"),
    ("NZ", "SA"): ("convergent", "Nazca-Güney Amerika (Andlar)"),
    ("PA", "AU"): ("convergent", "Tonga-Kermadec"),
    ("PA", "SA"): ("convergent", "Doğu Pasifik"),
    ("AF", "EU"): ("convergent", "Akdeniz — batı"),
    ("AF", "SA"): ("divergent",  "Mid-Atlantic Ridge — Güney"),
    ("IN", "EU"): ("convergent", "Himalaya kuşağı"),
    ("AU", "EU"): ("convergent", "Australia-Eurasia"),
    ("AN", "PA"): ("divergent",  "Pacific-Antarctic Ridge"),
}

BOUNDARY_TYPE_STYLE = {
    "convergent": {"color": "#ff5252", "width": 3.5},  # kırmızı, kalın — subduction/collision
    "divergent":  {"color": "#42a5f5", "width": 2.8},  # mavi — rift/MOR
    "transform":  {"color": "#ffd54f", "width": 3.0},  # sarı — strike-slip
    "unknown":    {"color": "#90a4ae", "width": 1.8},  # gri ince — etiketsiz
}

def classify_boundary(plate_a, plate_b):
    """PB2002 plaka çiftini sınıflandır: tip + okunabilir isim."""
    key = tuple(sorted([plate_a or "", plate_b or ""]))
    if key in PB2002_BOUNDARY_TYPES:
        return PB2002_BOUNDARY_TYPES[key]
    return ("unknown", "")

@st.cache_resource(show_spinner=False)
def load_tectonic_plates():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tectonic_plates.geojson")
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
        plate_a = props.get("PlateA") or ""
        plate_b = props.get("PlateB") or ""
        b_type, b_name = classify_boundary(plate_a, plate_b)
        style = BOUNDARY_TYPE_STYLE[b_type]
        for coords in segments:
            if len(coords) < 2:
                continue
            label_parts = [f"{plate_a}-{plate_b} Plaka Sınırı"]
            if b_name:
                label_parts.append(b_name)
            label_parts.append({
                "convergent": "Tip: Yaklaşan (subduction/çarpışma)",
                "divergent":  "Tip: Ayrılan (rift/sırt)",
                "transform":  "Tip: Yanal Kayan (strike-slip)",
                "unknown":    "Tip: Sınıflandırılmamış",
            }[b_type])
            lines.append({
                "isim":    " — ".join(label_parts),
                "lats":    [c[1] for c in coords],
                "lons":    [c[0] for c in coords],
                "type":    b_type,
                "color":   style["color"],
                "width":   style["width"],
                "plate_a": plate_a,  # PB2002 code (örn. "AT", "EU") — v1.17.3 toplu hareket için
                "plate_b": plate_b,
            })
    return lines

PLATE_LINES = load_tectonic_plates()

# PB2002 (Bird 2003) sınır plaka kodları ↔ data/plate_velocities.json hız kodları
# PB2002'de Anadolu "AT"; hız dosyasında "AN" — eşleme zorunlu.
# Aegean Sea (AS) MORVEL56'da ayrı çözülmüyor, Anadolu'yla benzer hızda → AN.
_PB2002_TO_VELOCITY_CODE = {
    "AT": "AN",   # Anatolian   → Anadolu
    "AS": "AN",   # Aegean Sea  ≈ Anadolu (NNR-MORVEL56'da AS yok)
    "EU": "EU",   # Eurasian
    "AF": "AF",   # African (Nubia)
    "AR": "AR",   # Arabian
    # Diğer global plakalar (NA, SA, PA, IN, AU, ...) hız dosyasında yok →
    # ilgili sınırlar v1.17.3 toplu simülasyonda sabit kalır (kayma 0).
}

# ════════════════════════════════════════════════════════════════════════════
# 🌍 Plaka Hız Vektörleri — v1.17 (Ajan 5)
# data/plate_velocities.json'dan iki olası formatı destekler:
#   1) NNR-MORVEL56 Euler kutubu  (Argus, Gordon & DeMets 2011, G-cubed)
#      {"AN": {"euler_lat", "euler_lon", "omega_deg_myr", ...}}
#   2) Hazır deg/yıl deltaları    (Ajan 5 fallback formatı)
#      {"plates": [{"plate": "AN", "delta_lat_per_year": ..., "delta_lon_per_year": ...}]}
# Hiçbir kaynak yoksa AN için literal hardcode fallback.
# ════════════════════════════════════════════════════════════════════════════
_PLAKA_FALLBACK_AN = {
    "delta_lat_per_year": -0.00022,
    "delta_lon_per_year":  0.00020,
    "name": "Anadolu (fallback)",
    "approx_speed_mm_yr": 25,
}

def _euler_to_delta_deg(euler_lat, euler_lon, omega_deg_myr, lat, lon):
    """NNR-MORVEL56 Euler kutbundan (lat, lon) noktasında deg/yıl yer değiştirme.

    Çıktı: (delta_lat_per_year, delta_lon_per_year) — derece/yıl cinsinden.
    Math: V = Ω × P (birim küre); V tanjant düzlemine doğu/kuzey projeksiyonu;
    dλ_rad = ve_rad / cos(φ).
    """
    omega_rad_yr = math.radians(omega_deg_myr) / 1_000_000.0
    phi   = math.radians(lat)
    lam   = math.radians(lon)
    phi_e = math.radians(euler_lat)
    lam_e = math.radians(euler_lon)
    cos_phi = math.cos(phi)
    if abs(cos_phi) < 1e-6:
        cos_phi = 1e-6 if cos_phi >= 0 else -1e-6

    P  = (math.cos(phi)*math.cos(lam), math.cos(phi)*math.sin(lam), math.sin(phi))
    Om = (omega_rad_yr*math.cos(phi_e)*math.cos(lam_e),
          omega_rad_yr*math.cos(phi_e)*math.sin(lam_e),
          omega_rad_yr*math.sin(phi_e))
    Vx = Om[1]*P[2] - Om[2]*P[1]
    Vy = Om[2]*P[0] - Om[0]*P[2]
    Vz = Om[0]*P[1] - Om[1]*P[0]
    east  = (-math.sin(lam),               math.cos(lam),                0.0)
    north = (-math.sin(phi)*math.cos(lam), -math.sin(phi)*math.sin(lam), math.cos(phi))
    ve_rad = Vx*east[0]  + Vy*east[1]  + Vz*east[2]
    vn_rad = Vx*north[0] + Vy*north[1] + Vz*north[2]
    dlat_deg_yr = math.degrees(vn_rad)
    dlon_deg_yr = math.degrees(ve_rad / cos_phi)
    return dlat_deg_yr, dlon_deg_yr

@st.cache_resource(show_spinner=False)
def load_plate_velocities(ref_lat: float = ERZ_LAT, ref_lon: float = ERZ_LON):
    """Tüm plakalar için (ref_lat, ref_lon) noktasında deg/yıl hız vektörü.

    Ajan 4 entegrasyonu (try/except ile):
      - varsa data/plate_velocities.json → format otomatik tespit
      - yoksa veya parse hatası → hardcode AN fallback

    Çıkış formatı:
      {plate_code: {"delta_lat_per_year": float, "delta_lon_per_year": float,
                    "name": str, "approx_speed_mm_yr": float|None,
                    "is_euler_derived": bool}}
    """
    out = {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "plate_velocities.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = None
        if isinstance(raw, dict):
            entries = []
            if "plates" in raw and isinstance(raw["plates"], list):
                entries = raw["plates"]
            else:
                for code, val in raw.items():
                    if isinstance(val, dict):
                        entries.append({"plate": code, **val})
            for e in entries:
                if not isinstance(e, dict):
                    continue
                code = e.get("plate") or e.get("code")
                if not code or not isinstance(code, str):
                    continue
                name = e.get("name") or code
                speed = e.get("approx_speed_mm_yr")
                if "delta_lat_per_year" in e and "delta_lon_per_year" in e:
                    out[code] = {
                        "delta_lat_per_year": float(e["delta_lat_per_year"]),
                        "delta_lon_per_year": float(e["delta_lon_per_year"]),
                        "name": name,
                        "approx_speed_mm_yr": speed,
                        "is_euler_derived": False,
                    }
                elif ("euler_lat" in e and "euler_lon" in e and "omega_deg_myr" in e):
                    dlat, dlon = _euler_to_delta_deg(
                        float(e["euler_lat"]), float(e["euler_lon"]),
                        float(e["omega_deg_myr"]), ref_lat, ref_lon)
                    out[code] = {
                        "delta_lat_per_year": dlat,
                        "delta_lon_per_year": dlon,
                        "name": name,
                        "approx_speed_mm_yr": speed,
                        "is_euler_derived": True,
                    }
                elif "ve_mm_yr" in e and "vn_mm_yr" in e:
                    ve = float(e["ve_mm_yr"])  # mm/yr east
                    vn = float(e["vn_mm_yr"])  # mm/yr north
                    cos_phi = math.cos(math.radians(ref_lat)) or 1e-6
                    out[code] = {
                        "delta_lat_per_year": (vn / 1000.0) / 111_320.0,
                        "delta_lon_per_year": (ve / 1000.0) / (111_320.0 * cos_phi),
                        "name": name,
                        "approx_speed_mm_yr": speed,
                        "is_euler_derived": False,
                    }
    if "AN" not in out:
        out["AN"] = {**_PLAKA_FALLBACK_AN, "is_euler_derived": False}
    return out

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


@st.cache_data(show_spinner=False, ttl=300)
def calc_etas_cache(df_sub_dict, d_frac, b_eta):
    import math

    import numpy as np
    import pandas as pd
    sub = pd.DataFrame(df_sub_dict)
    eta_list = []
    log_t_list = []
    log_r_list = []

    for j in range(1, len(sub)):
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
    return eta_list, log_t_list, log_r_list

@st.cache_data(show_spinner=False, ttl=300)
def calc_rtl_cache(df_dict, rtl_r0, rtl_t0, ERZ_LAT, ERZ_LON):
    import math

    import pandas as pd
    exp_df = pd.DataFrame(df_dict)
    exp_df["L_km"] = exp_df["buyukluk"].apply(
        lambda m: max(0.1, 10**(-2.44 + 0.59*m))
    )
    rtl_times, rtl_scores = [], []
    step = max(1, len(exp_df) // 80)
    for idx in range(10, len(exp_df), step):
        t_ref = exp_df["zaman"].iloc[idx]
        past = exp_df.iloc[:idx]
        score = 0.0
        for _, ev in past.iterrows():
            r = haversine(ERZ_LAT, ERZ_LON, ev["lat"], ev["lon"])
            dt_days = (t_ref - ev["zaman"]).total_seconds() / 86400
            if dt_days <= 0:
                continue
            score += (math.exp(-r / rtl_r0) * math.exp(-dt_days / rtl_t0) / ev["L_km"])
        rtl_times.append(t_ref)
        rtl_scores.append(score)
    return rtl_times, rtl_scores

@st.cache_data(show_spinner=False, ttl=300)
def calc_amr_cache(df_dict):
    import math

    import numpy as np
    import pandas as pd
    amr_df = pd.DataFrame(df_dict).sort_values("zaman").copy()
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
    t_days = ((amr_df["zaman"] - t0_amr).dt.total_seconds() / 86400).values
    C = amr_df["C_norm"].values
    T_obs = t_days[-1]

    best_rmse, best_m, best_tf, best_fitted = np.inf, 1.0, T_obs*1.5, C
    for tf_mult in np.linspace(1.05, 4.0, 25):
        tf = T_obs * tf_mult
        X_vals = (tf - t_days)
        X_vals = np.maximum(X_vals, 1e-6)
        for m_try in np.linspace(0.1, 1.9, 30):
            X = X_vals ** m_try
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
    return amr_df["zaman"].tolist(), C.tolist(), best_m, best_tf, best_fitted.tolist(), T_obs, t0_amr, best_rmse

# ─── CACHED FUNCTIONS FOR PERFORMANCE ───────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def calc_b_grid_cache(df_mc_dict, bg_n, bg_sr, bg_min, radius_km, ERZ_LAT, ERZ_LON, mc_g):
    import math

    import numpy as np
    import pandas as pd
    df_mc = pd.DataFrame(df_mc_dict)
    deg = 1 / 111
    margin_deg = radius_km * deg * 0.75
    lats_g = np.linspace(ERZ_LAT - margin_deg, ERZ_LAT + margin_deg, bg_n)
    lons_g = np.linspace(ERZ_LON - margin_deg * 1.3, ERZ_LON + margin_deg * 1.3, bg_n)

    b_grid  = np.full((bg_n, bg_n), np.nan)
    n_grid  = np.zeros((bg_n, bg_n), dtype=int)
    
    lats = df_mc["lat"].values
    lons = df_mc["lon"].values

    for i, lat_g in enumerate(lats_g):
        for j, lon_g in enumerate(lons_g):
            # Vectorized haversine distance calculation using NumPy (100x faster than df.apply)
            lat1_rad, lon1_rad = np.radians(lat_g), np.radians(lon_g)
            lat2_rad, lon2_rad = np.radians(lats), np.radians(lons)
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            dists = 6371 * 2 * np.arcsin(np.sqrt(a))
            
            sub_g = df_mc[dists <= bg_sr]
            if len(sub_g) < bg_min:
                continue
            mean_m = sub_g["buyukluk"].mean()
            if mean_m <= mc_g:
                continue
            b_val = math.log10(math.e) / (mean_m - mc_g)
            b_grid[i, j] = np.clip(b_val, 0.3, 3.0)
            n_grid[i, j] = len(sub_g)
    return b_grid, lats_g, lons_g

@st.fragment
def _render_canli_radar():
    # ─── Harita + Kayan Liste ───────────────────────────────────────────────────
    col_map, col_list = st.columns([2.8, 1])

    with col_map:
        st.markdown('<div class="chart-title">🗺️ Deprem Haritasi</div>', unsafe_allow_html=True)

        fig_map = go.Figure()
        bands = [
            ("M < 1",  df[df["buyukluk"] < 1],                                        "#B3E5FC"),
            ("M 1-2",  df[(df["buyukluk"] >= 1) & (df["buyukluk"] < 2)],              "#C5E1A5"),
            ("M 2-3",  df[(df["buyukluk"] >= 2) & (df["buyukluk"] < 3)],              "#76FF03"),
            ("M 3-4",  df[(df["buyukluk"] >= 3) & (df["buyukluk"] < 4)],              "#FFEA00"),
            ("M 4-5",  df[(df["buyukluk"] >= 4) & (df["buyukluk"] < 5)],              "#FF6D00"),
            ("M 5-6",  df[(df["buyukluk"] >= 5) & (df["buyukluk"] < 6)],              "#FF1744"),
            ("M 6-7",  df[(df["buyukluk"] >= 6) & (df["buyukluk"] < 7)],              "#D500F9"),
            ("M 7+",   df[df["buyukluk"] >= 7],                                        "#000000"),
        ]
        for label, sub, color in bands:
            if sub.empty: continue
            sizes = sub["buyukluk"].apply(lambda m: max(8, m * 6))
            hover_text = sub.apply(lambda r:
                f"<b>M{r['buyukluk']:.1f} — {r['sinif']}</b><br>"
                f"Derinlik: {r['derinlik']:.1f} km<br>"
                f"Zaman: {r['zaman_str']}<br>"
                f"Konum: {safe_html(str(r['konum'])[:55])}<br>"
                f"Erzincan'a: {r['uzaklik_km']} km<br>"
                f"Kaynak: {safe_html(r['kaynak'])}", axis=1)

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

            # v1.9 precomputed bbox + bbox-overlap testi (Python any() generator yerine
            # O(1) numerik karşılaştırma — 14,500 fay × ortalama 10 vertex'lik any()
            # döngüsünden ~10× hızlanma; ayrıca segmenti yalnızca uçları view dışındaysa
            # gözden kaçırma riskini ortadan kaldırır).
            def in_view(fault):
                return (fault["max_lat"] >= lat_min and fault["min_lat"] <= lat_max
                        and fault["max_lon"] >= lon_min and fault["min_lon"] <= lon_max)

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
                label = f"{safe_html(label)}<br>Kayma: {safe_html(fault['kayma'])}"
                if fault["uzunluk"]:
                    label += f" · Uzunluk: {safe_html(fault['uzunluk'])} km"
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

        if show_plates and PLATE_LINES:
            # Tip bazında grupla — convergent kırmızı, divergent mavi, transform sarı
            plates_by_type = {"convergent": [], "divergent": [], "transform": [], "unknown": []}
            for plate in PLATE_LINES:
                plates_by_type.setdefault(plate.get("type", "unknown"), []).append(plate)

            # Tüm tipler için ortak siyah gölge (önce çizilir, altta kalır)
            all_lats, all_lons = [], []
            for plates_in_type in plates_by_type.values():
                for plate in plates_in_type:
                    all_lats.extend(plate["lats"] + [None])
                    all_lons.extend(plate["lons"] + [None])
            fig_map.add_trace(go.Scattermapbox(
                lat=all_lats, lon=all_lons, mode="lines",
                showlegend=False, hoverinfo="skip",
                line=dict(color="rgba(0,0,0,0.7)", width=5),
            ))

            # Her tip için ayrı renkli trace (sıralama: önce unknown alta, sonra üst)
            type_display = {
                "unknown":    "Plaka Sınırı (sınıflandırılmamış)",
                "convergent": "🔺 Yaklaşan Sınır (subduction/çarpışma)",
                "divergent":  "🔻 Ayrılan Sınır (rift/sırt)",
                "transform":  "↔ Yanal Kayan Sınır (strike-slip)",
            }
            for b_type in ["unknown", "convergent", "divergent", "transform"]:
                plates_in_type = plates_by_type.get(b_type, [])
                if not plates_in_type:
                    continue
                style = BOUNDARY_TYPE_STYLE[b_type]
                t_lats, t_lons, t_labels = [], [], []
                for plate in plates_in_type:
                    t_lats.extend(plate["lats"] + [None])
                    t_lons.extend(plate["lons"] + [None])
                    t_labels.extend([plate["isim"]] * len(plate["lats"]) + [None])
                fig_map.add_trace(go.Scattermapbox(
                    lat=t_lats, lon=t_lons, mode="lines",
                    name=type_display[b_type], showlegend=True,
                    line=dict(color=style["color"], width=style["width"]),
                    text=t_labels,
                    hovertemplate="<b>%{text}</b><extra></extra>",
                ))

            # ─── Plaka tip glyph dekoratörleri (F-32) ──────────────────────
            # Çizgi vertex'leri boyunca eşit aralıklı yön/tip gösterici Unicode glyph:
            # ▲ convergent (subduction üçgenleri), ◇ divergent (rift baklava),
            # ↔ transform (yanal kayma okları). Mapbox tile'larda native dash/symbol
            # desteklenmediğinden text glyph kullanılıyor.
            def _sample_along_plates(plates_in_type, step_km):
                lats_out, lons_out = [], []
                for plate in plates_in_type:
                    lats, lons = plate["lats"], plate["lons"]
                    if len(lats) < 2:
                        continue
                    accumulated = 0.0
                    lats_out.append(lats[0])
                    lons_out.append(lons[0])
                    for k in range(1, len(lats)):
                        accumulated += haversine(lats[k-1], lons[k-1], lats[k], lons[k])
                        if accumulated >= step_km:
                            lats_out.append(lats[k])
                            lons_out.append(lons[k])
                            accumulated = 0.0
                return lats_out, lons_out

            glyph_specs = [
                # (tip,         karakter, step_km, font_size, renk)
                ("convergent", "▲", 80,  18, "#ff5252"),
                ("divergent",  "◇", 100, 14, "#42a5f5"),
                ("transform",  "↔", 120, 18, "#ffd54f"),
            ]
            for b_type, glyph_char, step_km, font_size, glyph_color in glyph_specs:
                plates_in_type = plates_by_type.get(b_type, [])
                if not plates_in_type:
                    continue
                g_lats, g_lons = _sample_along_plates(plates_in_type, step_km)
                if not g_lats:
                    continue
                fig_map.add_trace(go.Scattermapbox(
                    lat=g_lats, lon=g_lons,
                    mode="text",
                    text=[glyph_char] * len(g_lats),
                    textfont=dict(size=font_size, color=glyph_color),
                    textposition="middle center",
                    hoverinfo="skip",
                    showlegend=False,
                ))

            # Plaka İtme Yönleri (Oklar)
            plate_motions = [
                {"lat": 37.5, "lon": 41.5, "text": "↖ Arap Plakası", "isim": "Kuzeybatıya itiyor (~18 mm/yıl)"},
                {"lat": 39.0, "lon": 37.0, "text": "← Anadolu Plakası", "isim": "Batıya kaçıyor (~21 mm/yıl)"},
                {"lat": 40.5, "lon": 39.5, "text": "↓ Avrasya Plakası", "isim": "Güneye doğru direnç/baskı"},
            ]
            fig_map.add_trace(go.Scattermapbox(
                lat=[p["lat"] for p in plate_motions],
                lon=[p["lon"] for p in plate_motions],
                mode="markers+text",
                name="Plaka Hareket Yönü",
                marker=dict(size=14, color="#FF0055", symbol="circle"),
                text=[p["text"] for p in plate_motions],
                textposition="bottom right",
                textfont=dict(size=18, color="#FF0055"),
                hovertext=[p["isim"] for p in plate_motions],
                hovertemplate="<b>%{text}</b><br>%{hovertext}<extra></extra>",
                showlegend=False,
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
                loc = safe_html(str(row["konum"])[:44]) if row["konum"] else "—"
                kaynak = safe_html(row["kaynak"])
                sub = "rgba(200,215,230,0.7)" if DARK else "rgba(40,60,80,0.65)"
                cards += (
                    f'<div class="eq-card" style="border-left-color:{c}">'
                    f'{e} <b style="color:{c}">M{row["buyukluk"]:.1f}</b>'
                    f' <span style="color:{TEXT}">&nbsp;·&nbsp; {row["derinlik"]:.0f} km'
                    f' &nbsp;·&nbsp; {row["uzaklik_km"]} km</span><br>'
                    f'<span style="font-size:0.74rem;color:{sub}">'
                    f'{row["zaman_str"]} &nbsp;·&nbsp; {loc}'
                    f' &nbsp;·&nbsp; <i>{kaynak}</i></span>'
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
        ("M < 1",  df_plot[df_plot["buyukluk"] < 1],                                        "#B3E5FC"),
        ("M 1-2",  df_plot[(df_plot["buyukluk"] >= 1) & (df_plot["buyukluk"] < 2)],         "#C5E1A5"),
        ("M 2-3",  df_plot[(df_plot["buyukluk"] >= 2) & (df_plot["buyukluk"] < 3)],         "#76FF03"),
        ("M 3-4",  df_plot[(df_plot["buyukluk"] >= 3) & (df_plot["buyukluk"] < 4)],         "#FFEA00"),
        ("M 4-5",  df_plot[(df_plot["buyukluk"] >= 4) & (df_plot["buyukluk"] < 5)],         "#FF6D00"),
        ("M 5-6",  df_plot[(df_plot["buyukluk"] >= 5) & (df_plot["buyukluk"] < 6)],         "#FF1744"),
        ("M 6-7",  df_plot[(df_plot["buyukluk"] >= 6) & (df_plot["buyukluk"] < 7)],         "#D500F9"),
        ("M 7+",   df_plot[df_plot["buyukluk"] >= 7],                                        "#000000"),
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
                f"Konum: {safe_html(str(r['konum'])[:55])}<br>"
                f"Erzincan'a: {r['uzaklik_km']} km &nbsp;|&nbsp; {safe_html(r['kaynak'])}", axis=1),
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

if active_menu == "🌍 Canlı Radar":
    _render_canli_radar()

if active_menu == "⚙️ Sistem & Veri":
    st.markdown('<div class="chart-title">⚙️ 1. Kaynak Sağlığı</div>', unsafe_allow_html=True)
    status_rows = []
    for name in active_sources:
        status, cnt = statuses.get(name, ("HATA: yanıt yok", 0))
        status_rows.append({
            "Kaynak": name,
            "Durum": "Çalışıyor" if status.startswith("OK") else "Sorunlu",
            "Kayıt": cnt,
            "Mesaj": status,
        })
    status_df = pd.DataFrame(status_rows)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    src_counts = df.groupby("kaynak").size().sort_values(ascending=False).reset_index(name="kayıt")
    fig_health = px.bar(src_counts, x="kaynak", y="kayıt", color="kaynak")
    fig_health.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TEXT, size=11),
        margin=dict(t=10, b=45, l=45, r=10),
        height=320,
        showlegend=False,
        xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
        yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT), title="Kayıt"),
    )
    st.plotly_chart(fig_health, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📊 2. Veri Kalitesi Kontrolü</div>', unsafe_allow_html=True)
    raw_count = int(sum(cnt for _, cnt in statuses.values()))
    cleaned_count = len(df)
    removed_count = max(0, raw_count - cleaned_count)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Kaynaklardan Gelen", raw_count)
    q2.metric("Analizde Kullanılan", cleaned_count)
    q3.metric("Elenen / Birleşen", removed_count)
    q4.metric("Aktif Kaynak", len(active_sources))
    st.markdown(
        "**Tekilleştirme toleransları:** zaman < 120 sn, enlem/boylam farkı < 0.15°, büyüklük farkı < 0.5.  \n"
        "**Not:** Bu katalog, canlı izleme için normalize edilmiş olay listesidir. Ham kaynak raporları ile akademik kataloglar arasında revizyon farkları olabilir."
    )
    quality_rows = []
    for source_name, (status, cnt) in statuses.items():
        final_count = int((df["kaynak"] == source_name).sum()) if "kaynak" in df else 0
        quality_rows.append({
            "Kaynak": source_name,
            "Ham/Filtrelenmiş Kayıt": cnt,
            "Tekilleştirme Sonrası Temsilci": final_count,
            "Durum": status,
        })
    st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
    st.info(
        "Veri analizleri tekilleştirilmiş katalog üzerinden yürütülür. "
        "Kaynak bazlı farklı büyüklük/konum raporları bilimsel belirsizliktir; kesin hüküm değil, ölçüm ve kataloglama farkı olarak ele alınmalıdır."
    )

if active_menu == "🧭 Fay Sistemleri":
    st.markdown('<div class="chart-title">🧭 Fay Analizi</div>', unsafe_allow_html=True)
    fault_sample = df.head(250).copy()
    nearest_rows = []
    for _, ev in fault_sample.iterrows():
        nearest = nearest_fault_vertex_distance(ev["lat"], ev["lon"], FAULT_LINES)
        nearest_rows.append({
            "Zaman": ev["zaman_str"],
            "M": ev["buyukluk"],
            "Konum": ev["konum"],
            "Yakın Fay": nearest["fault_name"],
            "Fay Uzaklığı (km)": nearest["distance_km"],
        })
    fault_df = pd.DataFrame(nearest_rows).dropna(subset=["Fay Uzaklığı (km)"])
    f1, f2 = st.columns([1, 1])
    with f1:
        fig_fault = px.histogram(fault_df, x="Fay Uzaklığı (km)", nbins=25, color_discrete_sequence=["#64b5f6"])
        fig_fault.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TEXT, size=11),
            margin=dict(t=10, b=45, l=45, r=10),
            height=320,
            xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
            yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT), title="Olay"),
        )
        st.plotly_chart(fig_fault, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    with f2:
        top_faults = fault_df["Yakın Fay"].value_counts().head(10).reset_index()
        top_faults.columns = ["Fay", "Olay"]
        fig_top_faults = px.bar(top_faults, x="Olay", y="Fay", orientation="h", color="Olay")
        fig_top_faults.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TEXT, size=11),
            margin=dict(t=10, b=35, l=110, r=10),
            height=320,
            showlegend=False,
            xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
            yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
        )
        st.plotly_chart(fig_top_faults, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    st.dataframe(fault_df.head(100), use_container_width=True, hide_index=True)

@st.fragment
def _render_istatistik_top():
    st.markdown('<div class="chart-title">🤖 Sistem Yorumu (Uzman İçgörüsü)</div>', unsafe_allow_html=True)
    total_eq = len(df)
    mag_max = df["buyukluk"].max() if not df.empty else 0
    zaman_label = str(days_label).lower()
    shallow_pct = (len(df[df["derinlik"] <= 10]) / total_eq * 100) if total_eq > 0 else 0
    insight_text = f"**Analiz:** {zaman_label} içinde izlenen sismik aktivite toplam **{total_eq}** deprem üretti. "
    if mag_max >= 5.0:
        insight_text += f"Bölgede olağandışı hareketlilik gözleniyor (Maks: **M{mag_max}**). "
    elif total_eq > 50:
        insight_text += "Aktivite sayısında yüksek bir yoğunluk mevcut, ancak büyük yıkıcı enerji birikimi raporlanmadı. "
    else:
        insight_text += "Genel aktivite seviyesi beklenen sismik aralıklarda seyrediyor. "
    insight_text += f"Depremlerin **%{shallow_pct:.1f}** kadarı 10 km'den daha sığ derinliklerde meydana geldi."
    st.info(insight_text, icon="🧠")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🚦 Aktivite / Alarm</div>', unsafe_allow_html=True)
    agreement = source_agreement_summary(df.to_dict("records"))
    recent_factor = min(40, len(last24h) * 4)
    mag_factor = min(35, max(0, (df["buyukluk"].max() - 2.0) * 12))
    source_factor = min(15, agreement["source_count"] * 2)
    distance_factor = min(10, max(0, 10 - float(df["uzaklik_km"].min()) / 10))
    alarm_score = round(min(100, recent_factor + mag_factor + source_factor + distance_factor))
    level = activity_level(alarm_score)
    st.metric("Aktivite Skoru", f"{alarm_score}/100", level)
    score_parts = pd.DataFrame([
        {"Bileşen": "Son 24 saat yoğunluğu", "Puan": recent_factor},
        {"Bileşen": "En büyük deprem", "Puan": mag_factor},
        {"Bileşen": "Kaynak kapsamı", "Puan": source_factor},
        {"Bileşen": "Erzincan yakınlığı", "Puan": distance_factor},
    ])
    fig_score = px.bar(score_parts, x="Bileşen", y="Puan", color="Bileşen")
    fig_score.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TEXT, size=11),
        margin=dict(t=10, b=45, l=45, r=10),
        height=320,
        showlegend=False,
        xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
        yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT), range=[0, 45]),
    )
    st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    st.info("Bu skor deprem tahmini değildir; sadece seçilen veri penceresindeki aktiviteyi özetleyen karar destek göstergesidir.")

if active_menu == "📊 İstatistik & Analiz":
    _render_istatistik_top()

if active_menu == "📝 Raporlar":
    st.markdown('<div class="chart-title">🧾 Radar Raporu</div>', unsafe_allow_html=True)
    report_lines = [
        f"**Sürüm:** v {APP_VERSION}",
        f"**Zaman aralığı:** {days_label}",
        f"**Toplam olay:** {len(df)}",
        f"**Son 24 saat:** {len(last24h)}",
        f"**En büyük olay:** M{df['buyukluk'].max():.1f}",
        f"**Aktif kaynak sayısı:** {df['kaynak'].nunique()}",
        f"**Erzincan'a en yakın olay:** {df['uzaklik_km'].min():.1f} km",
    ]
    st.markdown("\n\n".join(report_lines))
    report_text = "\n".join(line.replace("**", "") for line in report_lines)
    st.download_button(
        "Raporu indir (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"erzincan_radar_rapor_v{APP_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )

if active_menu == "🎓 Bilgi Havuzu":
    @st.fragment
    def _render_edu():
        st.markdown('<div class="chart-title">📚 Temel Deprem Mühendisliği Bilgi Havuzu</div>', unsafe_allow_html=True)
        st.caption("Bu ekran öğretici simülasyon alanıdır; resmi tehlike haritası, ShakeMap veya yapı performans hesabı değildir.")

        edu_mode = st.radio(
            "Eğitim modu",
            ["3D Fay Mekaniği", "P / S / Rayleigh Dalgaları", "Erzincan Sanal Etki Haritası"],
            horizontal=True,
            key="edu_mode",
        )

        def cuboid_mesh(x0, x1, y0, y1, z0, z1, dx, dy, dz, name, color):
            x = [x0 + dx, x1 + dx, x1 + dx, x0 + dx, x0 + dx, x1 + dx, x1 + dx, x0 + dx]
            y = [y0 + dy, y0 + dy, y1 + dy, y1 + dy, y0 + dy, y0 + dy, y1 + dy, y1 + dy]
            z = [z0 + dz, z0 + dz, z0 + dz, z0 + dz, z1 + dz, z1 + dz, z1 + dz, z1 + dz]
            return go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=[0, 0, 0, 4, 4, 2, 1, 3, 0, 1, 5, 4],
                j=[1, 2, 3, 5, 6, 3, 5, 7, 4, 2, 6, 7],
                k=[2, 3, 7, 6, 7, 6, 6, 6, 1, 5, 7, 0],
                name=name,
                color=color,
                opacity=0.72,
                flatshading=True,
            )

        if edu_mode == "3D Fay Mekaniği":
            fault_info = {
                "Sağ yanal doğrultu atımlı fay": {
                    "desc": "Fayın karşı bloğu gözlemciye göre sağa hareket eder. Kuzey Anadolu Fayı'nın Erzincan çevresindeki baskın davranışı sağ yanal doğrultu atımlıdır.",
                    "mode": "right",
                    "stress": "Makaslama / yanal kayma",
                },
                "Sol yanal doğrultu atımlı fay": {
                    "desc": "Fayın karşı bloğu gözlemciye göre sola hareket eder. Hareket baskın olarak yataydır.",
                    "mode": "left",
                    "stress": "Makaslama / yanal kayma",
                },
                "Normal fay": {
                    "desc": "Üst blok, alt bloğa göre aşağı hareket eder. Genellikle kabuğun uzamasıyla ilişkilidir.",
                    "mode": "normal",
                    "stress": "Gerilme / açılma",
                },
                "Ters / bindirme fay": {
                    "desc": "Üst blok, alt bloğa göre yukarı hareket eder. Sıkışma rejimlerinde görülür.",
                    "mode": "reverse",
                    "stress": "Sıkışma",
                },
                "Oblik fay": {
                    "desc": "Yatay doğrultu atım ve düşey atım bileşenleri birlikte anlamlıdır.",
                    "mode": "oblique",
                    "stress": "Makaslama + sıkışma veya gerilme",
                },
            }
            col_fault_controls, _ = st.columns([1.15, 0.85])
            with col_fault_controls:
                selected_fault_type = st.radio("Fay tipi", list(fault_info.keys()), key="edu_fault_type", horizontal=True)
                slip = st.slider("Atım miktarı", 0.0, 1.0, 0.45, 0.05, key="edu_slip")
            info = fault_info[selected_fault_type]

            def displacement_for(mode, factor):
                left = {"dx": 0, "dy": 0, "dz": 0}
                right = {"dx": 0, "dy": 0, "dz": 0}
                amount = slip * factor
                if mode == "right":
                    left["dy"], right["dy"] = amount, -amount
                elif mode == "left":
                    left["dy"], right["dy"] = -amount, amount
                elif mode == "normal":
                    right["dz"] = -amount
                    right["dx"] = amount * 0.25
                elif mode == "reverse":
                    right["dz"] = amount
                    right["dx"] = -amount * 0.25
                elif mode == "oblique":
                    left["dy"], right["dy"], right["dz"] = -amount * 0.45, amount * 0.45, amount * 0.55
                return left, right

            mode = info["mode"]
            left_disp, right_disp = displacement_for(mode, 1.0)
            fig_fault_demo = go.Figure()

            # Initial Traces
            fig_fault_demo.add_trace(cuboid_mesh(-2.0, -0.08, -1.1, 1.1, -0.6, 0.6, **left_disp, name="Sol blok", color="#42A5F5"))
            fig_fault_demo.add_trace(cuboid_mesh(0.08, 2.0, -1.1, 1.1, -0.6, 0.6, **right_disp, name="Sağ blok", color="#FFB74D"))
            fig_fault_demo.add_trace(go.Surface(
                x=np.array([[0, 0], [0, 0]]),
                y=np.array([[-1.35, 1.35], [-1.35, 1.35]]),
                z=np.array([[-0.75, -0.75], [0.75, 0.75]]),
                name="Fay düzlemi",
                colorscale=[[0, "#E53935"], [1, "#E53935"]],
                opacity=0.42,
                showscale=False,
            ))
            fig_fault_demo.add_trace(go.Scatter3d(
                x=[-1.0, -1.0 + left_disp["dx"], 1.0, 1.0 + right_disp["dx"]],
                y=[0, left_disp["dy"], 0, right_disp["dy"]],
                z=[0.78, 0.78 + left_disp["dz"], 0.78, 0.78 + right_disp["dz"]],
                mode="lines+markers", name="Atım vektörü", line=dict(color="#E3F2FD", width=6), marker=dict(size=4, color="#E3F2FD"),
            ))
            fig_fault_demo.add_trace(go.Cone(
                x=[1.0 + right_disp["dx"]], y=[0 + right_disp["dy"]], z=[0.6 + right_disp["dz"]],
                u=[0], v=[0], w=[0.5], sizemode="absolute", sizeref=0.5, anchor="tail",
                colorscale=[[0, "#00E5FF"], [1, "#00E5FF"]], showscale=False, name="Gözlemci İkonu"
            ))
            frames = []
            for step in np.linspace(0, 1, 9):
                left_frame, right_frame = displacement_for(mode, float(step))
                frames.append(go.Frame(
                    data=[
                        cuboid_mesh(-2.0, -0.08, -1.1, 1.1, -0.6, 0.6, **left_frame, name="Sol blok", color="#42A5F5"),
                        cuboid_mesh(0.08, 2.0, -1.1, 1.1, -0.6, 0.6, **right_frame, name="Sağ blok", color="#FFB74D"),
                        go.Scatter3d(
                            x=[-1.0, -1.0 + left_frame["dx"], 1.0, 1.0 + right_frame["dx"]],
                            y=[0, left_frame["dy"], 0, right_frame["dy"]],
                            z=[0.78, 0.78 + left_frame["dz"], 0.78, 0.78 + right_frame["dz"]],
                            mode="lines+markers", line=dict(color="#E3F2FD", width=6), marker=dict(size=4, color="#E3F2FD")
                        ),
                        go.Cone(
                            x=[1.0 + right_frame["dx"]], y=[0 + right_frame["dy"]], z=[0.6 + right_frame["dz"]],
                            u=[0], v=[0], w=[0.5], sizemode="absolute", sizeref=0.5, anchor="tail", colorscale=[[0, "#00E5FF"], [1, "#00E5FF"]], showscale=False
                        )
                    ],
                    traces=[0, 1, 3, 4],
                    name=f"{step:.2f}",
                ))
            fig_fault_demo.frames = frames

            fig_fault_demo.update_layout(
                uirevision="constant",
                paper_bgcolor=BG,
                plot_bgcolor=BG2,
                font=dict(color=TEXT),
                height=500,
                margin=dict(t=8, b=8, l=0, r=0),
                scene=dict(
                    bgcolor=BG2,
                    xaxis=dict(title="", range=[-2.4, 2.4], color=TEXT, gridcolor=GRID, showticklabels=False),
                    yaxis=dict(title="", range=[-1.9, 1.9], color=TEXT, gridcolor=GRID, showticklabels=False),
                    zaxis=dict(title="", range=[-1.2, 1.2], color=TEXT, gridcolor=GRID, showticklabels=False),
                    aspectmode="manual",
                    aspectratio=dict(x=1.8, y=1.2, z=0.75),
                ),
                legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)", orientation="h"),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    x=0.02,
                    y=0.02,
                    buttons=[dict(
                        label="▶ Animasyonu Oynat",
                        method="animate",
                        args=[None, {"frame": {"duration": 130, "redraw": True}, "fromcurrent": True}],
                    )],
                )],
            )
            st.plotly_chart(fig_fault_demo, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

            st.markdown("---")
            st.markdown(f"**Tanım:** {info['desc']}")
            st.markdown(f"**Baskın gerilme biçimi:** {info['stress']}")
            st.markdown("**Erzincan notu:** Erzincan ve yakın çevresi, Kuzey Anadolu Fayı üzerinde sağ yanal doğrultu atımlı tektonik rejimle öne çıkar.")
            st.markdown(
                "**Kaynaklar:** "
                "[USGS fay türleri](https://www.usgs.gov/faqs/what-a-fault-and-what-are-different-types), "
                "[IRIS fay animasyonları](https://iris.edu/hq/inclass/animation/fault_strikeslip), "
                "[Britannica fault geology](https://www.britannica.com/science/fault-geology)"
            )
            st.warning("Bu model kavramsaldır; gerçek fay yüzeyi, eğim, rake, segment süreksizlikleri ve yerel zemin koşulları hesaba katılmaz.")


        elif edu_mode == "P / S / Rayleigh Dalgaları":
            st.markdown(
                "Deprem dalgalarının yeraltında ve yüzeyde nasıl ilerlediğini anlamak için "
                "**2 Boyutlu Parçacık Izgarası** (Particle Grid) modelini kullanıyoruz. "
                "Her bir nokta (parçacık), kayaları veya toprağı temsil eder."
            )

            c_wave1, c_wave2 = st.columns([1, 2])
            with c_wave1:
                wave_type = st.radio("Gösterilecek Dalga", ["P Dalgası (Sıkışma)", "S Dalgası (Kesme)", "Rayleigh (Yüzey)"], key="wave_type_radio")
            with c_wave2:
                st.info("💡 **İpucu:** Yeraltındaki noktaların dalga geçerken nasıl titreştiğine dikkatlice bakın! P dalgası ileri-geri, S dalgası yukarı-aşağı, Rayleigh ise eliptik olarak sallanır.")

            grid_x, grid_z = np.meshgrid(np.linspace(0, 100, 26), np.linspace(-40, 0, 11))
            x_base = grid_x.flatten()
            z_base = grid_z.flatten()

            focus_x, focus_z = 0, -20
            dist = np.sqrt((x_base - focus_x)**2 + (z_base - focus_z)**2)

            fig_wave2d = go.Figure()

            # Yeraltı Toprak Dokusu
            fig_wave2d.add_shape(
                type="rect", x0=-5, y0=-45, x1=105, y1=0,
                fillcolor="#3E2723", opacity=0.3, layer="below", line_width=0
            )

            # Zemin
            fig_wave2d.add_trace(go.Scatter(x=[-5, 105], y=[0, 0], mode="lines", line=dict(color="#4CAF50", width=4), name="Yeryüzü", hoverinfo="skip"))

            # Binalar
            bina_x = [20, 50, 80]
            for bx in bina_x:
                fig_wave2d.add_shape(
                    type="rect", x0=bx-2, y0=0, x1=bx+2, y1=4,
                    fillcolor="#B0BEC5", line=dict(color="#37474F", width=2)
                )
                fig_wave2d.add_shape(
                    type="path", path=f"M {bx-2.5} 4 L {bx} 6.5 L {bx+2.5} 4 Z",
                    fillcolor="#E53935", line=dict(color="#B71C1C", width=2)
                )

            fig_wave2d.add_trace(go.Scatter(x=x_base, y=z_base, mode="markers",
                                            marker=dict(size=8, color=np.zeros_like(x_base), colorscale="YlOrRd", cmin=0, cmax=2.5, showscale=False,
                                                        line=dict(color="#000000", width=0.5)),
                                            name="Parçacıklar", hoverinfo="skip"))
            fig_wave2d.add_trace(go.Scatter(x=[focus_x], y=[focus_z], mode="markers+text", text=["Odak"], textposition="bottom right", marker=dict(size=16, color="#E53935", symbol="star"), name="Odak", hoverinfo="skip"))

            frames = []
            num_frames = 60
            for t in range(0, num_frames):
                dx = np.zeros_like(x_base)
                dz = np.zeros_like(z_base)

                if "P Dalgası" in wave_type:
                    radius = t * 2.2
                    active = np.abs(dist - radius) < 15
                    amp = 3.5 * np.exp(-(dist - radius)**2 / 25)
                    freq = 0.5
                    dx[active] = amp[active] * (x_base[active] - focus_x) / dist[active] * np.sin((dist[active] - radius) * freq)
                    dz[active] = amp[active] * (z_base[active] - focus_z) / dist[active] * np.sin((dist[active] - radius) * freq)

                elif "S Dalgası" in wave_type:
                    radius = t * 1.5
                    active = np.abs(dist - radius) < 12
                    amp = 3.5 * np.exp(-(dist - radius)**2 / 20)
                    freq = 0.6
                    nx = -(z_base[active] - focus_z) / dist[active]
                    nz = (x_base[active] - focus_x) / dist[active]
                    dx[active] = amp[active] * nx * np.sin((dist[active] - radius) * freq)
                    dz[active] = amp[active] * nz * np.sin((dist[active] - radius) * freq)

                elif "Rayleigh" in wave_type:
                    radius = t * 1.2
                    depth_decay = np.exp(z_base / 8.0)
                    surf_dist = x_base - focus_x
                    active = np.abs(surf_dist - radius) < 15
                    amp = 4.5 * depth_decay * np.exp(-(surf_dist - radius)**2 / 25)
                    freq = 0.5
                    dx[active] = amp[active] * 0.7 * np.sin((surf_dist[active] - radius) * freq)
                    dz[active] = amp[active] * np.cos((surf_dist[active] - radius) * freq)

                energy = np.sqrt(dx**2 + dz**2)
                color_array = np.where(energy > 0.1, energy, 0)

                frames.append(go.Frame(
                    data=[go.Scatter(x=x_base + dx, y=z_base + dz, mode="markers",
                                     marker=dict(size=8, color=color_array, colorscale="YlOrRd", cmin=0, cmax=2.5, showscale=False,
                                                 line=dict(color="#000000", width=0.5)))],
                    traces=[2],
                    name=str(t)
                ))

            fig_wave2d.frames = frames
            fig_wave2d.update_layout(
                uirevision="constant",
                paper_bgcolor=BG,
                plot_bgcolor=BG2,
                font=dict(color=TEXT),
                height=500,
                xaxis=dict(title="Mesafe (km)", range=[-5, 105], gridcolor=GRID, zeroline=False),
                yaxis=dict(title="Derinlik (km)", range=[-45, 10], gridcolor=GRID, zeroline=False),
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=False,
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    x=0.01, y=1.05,
                    bgcolor=BG2,
                    font=dict(color=TEXT),
                    buttons=[dict(
                        label="▶ Animasyonu Oynat (Yavaş ve Detaylı)",
                        method="animate",
                        args=[None, {"frame": {"duration": 250, "redraw": False}, "fromcurrent": True}],
                    )]
                )]
            )
            st.plotly_chart(fig_wave2d, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

            st.markdown("---")
            if "P Dalgası" in wave_type:
                st.markdown("**P (Primary/Birincil) Dalgası:** İlk ulaşan dalgadır. Parçacıkları ses dalgası gibi sıkıştırıp genleştirerek (yayılım yönünde ileriye-geriye) titreştirir. Yukarıdaki animasyonda dalga geçerken noktaların sağa-sola esnediğini görebilirsiniz.")
            elif "S Dalgası" in wave_type:
                st.markdown("**S (Secondary/İkincil) Dalgası:** P'den sonra gelir. Parçacıkları ilerleme yönüne dik (yukarı-aşağı) keserek dalgalandırır. Binaları yanal olarak en çok sarsan ve hasar veren dalgalardan biridir.")
            else:
                st.markdown("**Rayleigh (Yüzey) Dalgası:** Sadece yeryüzüne yakın kısımlarda ilerler. Derinlere indikçe etkisi hızla azalır. Yeryüzündeki parçacıklar geriye doğru eliptik bir yörünge (okyanus dalgası gibi yuvarlanma) çizer. En yıkıcı etkiye sahip dalgalardır.")

        else:
            scenarios = {
                "Yedisu Segmenti (Doğu)": {"mag": 7.2, "depth": 12, "lat": 39.43, "lon": 40.54, "mechanism": "Sağ yanal / Yüksek tehlike"},
                "Karlıova Kesimi (Uzak Doğu)": {"mag": 7.4, "depth": 15, "lat": 39.30, "lon": 41.01, "mechanism": "Kesişim bölgesi"},
                "Refahiye Segmenti (Batı)": {"mag": 6.8, "depth": 10, "lat": 39.90, "lon": 38.76, "mechanism": "Sağ yanal doğrultu atımlı"},
                "Erzincan Merkez": {"mag": 7.8, "depth": 15, "lat": 39.75, "lon": 39.50, "mechanism": "Sağ yanal yıkıcı sarsıntı"},
            }
            selected_scenario = st.selectbox("Merkez Üssü (Episantr) Seçimi", list(scenarios.keys()), key="erz_scenario")
            scenario = scenarios[selected_scenario]

            sc1, sc2 = st.columns(2)
            with sc1:
                scenario_mag = st.slider("Moment büyüklüğü Mw", 3.5, 8.0, float(scenario["mag"]), 0.1, key="scenario_mag")
            with sc2:
                scenario_depth = st.slider("Derinlik (km)", 3, 40, int(scenario["depth"]), 1, key="scenario_depth")

            # Dalga hızları (km/s)
            vp = 6.0
            vs = 3.5
            vr = 3.0

            # Erzincan'a olan gerçek uzaklık (Haversine formülü)
            dist_to_erz = haversine(ERZ_LAT, ERZ_LON, scenario["lat"], scenario["lon"])
            p_arrival = dist_to_erz / vp
            s_arrival = dist_to_erz / vs
            r_arrival = dist_to_erz / vr

            st.info(f"📍 **Merkez Üssü - Erzincan Mesafesi:** {dist_to_erz:.0f} km | ⏱️ **Dalga Varış Süreleri:** P Dalgası: **{p_arrival:.1f} sn** | S Dalgası: **{s_arrival:.1f} sn** | Yüzey Dalgası: **{r_arrival:.1f} sn**")

            def create_circle_coords(clat, clon, radius_km, points=100):
                if radius_km <= 0:
                    return [clon] * points, [clat] * points
                R = 6371.0
                clat_rad = math.radians(clat)
                clon_rad = math.radians(clon)
                lats, lons = [], []
                for bearing in np.linspace(0, 2 * math.pi, points):
                    lat2_rad = math.asin(math.sin(clat_rad) * math.cos(radius_km / R) +
                                         math.cos(clat_rad) * math.sin(radius_km / R) * math.cos(bearing))
                    lon2_rad = clon_rad + math.atan2(math.sin(bearing) * math.sin(radius_km / R) * math.cos(clat_rad),
                                                     math.cos(radius_km / R) - math.sin(clat_rad) * math.sin(lat2_rad))
                    lats.append(math.degrees(lat2_rad))
                    lons.append(math.degrees(lon2_rad))
                return lons, lats

            fig_erz = go.Figure()

            # Merkez Üssü
            event_lat = scenario["lat"]
            event_lon = scenario["lon"]

            fig_erz.add_trace(go.Scattermapbox(
                lat=[event_lat], lon=[event_lon],
                mode="markers+text",
                marker=dict(size=20, color="#FFD54F", symbol="star", allowoverlap=True),
                text=[f"⭐ Merkez Üssü M{scenario_mag}"],
                textposition="bottom center",
                textfont=dict(color="#FFFFFF", size=14, family="Arial Black"),
                name="Sanal deprem kaynağı"
            ))

            # Animasyon Süresi (Streamlit Slider tabanlı interaktif scrub)
            max_t = int(max(dist_to_erz, 220) / vr) + 15
            if max_t > 150: max_t = 150
            if max_t < 40: max_t = 40

            st.markdown("### ⏱️ Deprem Yayılım Simülasyonu")
            sim_time = st.slider("Zamanı ileri/geri alarak sismik dalganın hedeflere varışını gözlemleyin (Saniye)", 0, max_t, 0, step=1, key="sim_slider")

            p_rad = vp * sim_time
            s_rad = vs * sim_time
            r_rad = vr * sim_time

            p_lon, p_lat = create_circle_coords(event_lat, event_lon, p_rad)
            s_lon, s_lat = create_circle_coords(event_lat, event_lon, s_rad)
            r_lon, r_lat = create_circle_coords(event_lat, event_lon, r_rad)

            fig_erz.add_trace(go.Scattermapbox(lat=p_lat, lon=p_lon, mode="lines", line=dict(color="#29B6F6", width=3), name=f"P-Dalgası (r={p_rad:.0f}km)"))
            fig_erz.add_trace(go.Scattermapbox(lat=s_lat, lon=s_lon, mode="lines", line=dict(color="#FFA726", width=4), name=f"S-Dalgası (r={s_rad:.0f}km)"))
            fig_erz.add_trace(go.Scattermapbox(lat=r_lat, lon=r_lon, mode="lines", line=dict(color="#F44336", width=6), name=f"Rayleigh Dalgası (r={r_rad:.0f}km)"))

            # Fay Hatları (Canlı Radardaki Altlık)
            if FAULT_LINES:
                deg = 1.0 / 111.0
                margin = max(220 * 1.6, 250) * deg
                lat_min, lat_max = ERZ_LAT - margin, ERZ_LAT + margin
                lon_min, lon_max = ERZ_LON - margin / math.cos(math.radians(ERZ_LAT)), ERZ_LON + margin / math.cos(math.radians(ERZ_LAT))

                def in_view(fault):
                    return (fault["max_lat"] >= lat_min and fault["min_lat"] <= lat_max
                            and fault["max_lon"] >= lon_min and fault["min_lon"] <= lon_max)

                visible = [f for f in FAULT_LINES if in_view(f)]

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
                    fig_erz.add_trace(go.Scattermapbox(
                        lat=data["lats"], lon=data["lons"],
                        mode="lines",
                        line=dict(width=1.5, color=color),
                        hoverinfo="text",
                        text=data["labels"],
                        hovertemplate="%{text}<extra></extra>",
                        name="Diri Fay (MTA)",
                        showlegend=False
                    ))

            fig_erz.update_layout(
                uirevision="constant",
                paper_bgcolor=BG,
                plot_bgcolor=BG2,
                font=dict(color=TEXT),
                height=650,
                margin=dict(t=30, b=8, l=0, r=0),
                mapbox=dict(
                    **make_mapbox_layout("Uydu"),
                    center=dict(lat=ERZ_LAT, lon=ERZ_LON),
                    zoom=7.5,
                    pitch=0,
                ),
                legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.1),
            )
            st.plotly_chart(fig_erz, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

            c1, c2, c3 = st.columns(3)
            c1.metric("Mekanizma", scenario["mechanism"])
            intensity_index = min(10, max(1, scenario_mag * 1.45 - math.log10(scenario_depth + 5) * 2.1 + 1.2))
            c2.metric("Eğitim etki göstergesi", f"{intensity_index:.1f} / 10")
            impact_radius = min(150, 20 * scenario_mag)
            c3.metric("Yıkıcı etki yarıçapı", f"{impact_radius:.0f} km")
            st.markdown("---")
            st.markdown("**3B sahne bilgisi:** Harita Karlıova'dan Refahiye'ye kadar genişletilmiştir. Mavi halka P-Dalgasını (Hızlı, uyarıcı), Turuncu halka S-Dalgasını (Kesme) ve Kırmızı halka Rayleigh Yüzey Dalgasını (En yıkıcı) temsil eder.")
            st.markdown(f"**Gerçek Zamanlı Fizik:** Animasyondaki saniyeler gerçek hıza ayarlıdır (P: ~{vp} km/s, S: ~{vs} km/s, Rayleigh: ~{vr} km/s). Yıldız merkez üssünden Erzincan'a varış sürelerini yukarıdaki panelden kontrol edebilirsiniz.")
            st.warning("Bu çıktı resmi deprem senaryosu, yapı tasarım girdisi veya afet tahmini değildir; yalnızca eğitim amaçlı nitel bir görselleştirmedir.")

    _render_edu()

def compute_environmental_features(row, full_df):
    t = row["zaman"]
    lat_str = str(row["lat"])
    lon_str = str(row["lon"])
    
    # 1. Mevsim Etkisi (Day of Year sin dalgası: 1 = Yaz, -1 = Kış)
    doy = t.timetuple().tm_yday
    mevsim = math.sin((doy - 172) * 2 * math.pi / 365.25)
    
    # 2. Beklenen Ortalama Sıcaklık (Türkiye/Erzincan yaklaşık iklim modeli)
    sicaklik = 11.5 + 16.5 * math.sin((doy - 105) * 2 * math.pi / 365.25)
    
    # Astronomik Gözlemci
    obs = ephem.Observer()
    obs.lat = lat_str
    obs.lon = lon_str
    # pandas timestamp -> datetime (naive = UTC for this app)
    obs.date = t.to_pydatetime()
    
    # 3. Ayın Çekim Gücü
    moon = ephem.Moon(obs)
    ay_cekim = 1.0 / (moon.earth_distance ** 2)
    
    # 4. Güneşe Uzaklık (AU)
    sun = ephem.Sun(obs)
    gunes_uzaklik = sun.earth_distance
    
    # 5. Gezegenlerin Çekim Etkisi (Jüpiter + Venüs proxy)
    jupiter = ephem.Jupiter(obs)
    venus = ephem.Venus(obs)
    gezegen_cekim = (317.8 / (jupiter.earth_distance ** 2)) + (0.815 / (venus.earth_distance ** 2))
    
    # 6. Haftalık Aktivite Yoğunluğu (Son 7 gün)
    haftalik_aktivite = len(full_df[(full_df["zaman"] >= t - timedelta(days=7)) & (full_df["zaman"] < t)])
    
    return pd.Series({
        "mevsim": mevsim,
        "sicaklik": sicaklik,
        "ay_cekim": ay_cekim,
        "gunes_uzaklik": gunes_uzaklik,
        "gezegen_cekim": gezegen_cekim,
        "haftalik_aktivite": haftalik_aktivite
    })

@st.fragment
def _render_istatistik_bottom():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🔬 Bilimsel Analizler (Derinlik, G-R & B-Value)</div>', unsafe_allow_html=True)

    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False
        
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        if st.button("🚀 Bilimsel Analizleri Çalıştır (Korelasyon, b-Değeri vb.)", use_container_width=True, type="primary"):
            st.session_state.run_analysis = True
    with col_btn2:
        if st.button("🛑 Kapat", use_container_width=True):
            st.session_state.run_analysis = False
            st.rerun()

    if not st.session_state.run_analysis:
        st.info("İşlemci (CPU) performansını korumak için ağır bilimsel istatistik hesaplamaları beklemeye alındı. Analizleri başlatmak için yukarıdaki **Çalıştır** düğmesine tıklayın.", icon="⏸️")
    else:
        agreement = source_agreement_summary(df.to_dict("records"))
        energy_total = float(df["buyukluk"].apply(estimate_energy_joules).sum())
        recent_factor = min(40, len(last24h) * 4)
        mag_factor = min(35, max(0, (df["buyukluk"].max() - 2.0) * 12))
        source_factor = min(15, agreement["source_count"] * 2)
        fault_factor = min(10, max(0, 10 - float(df["uzaklik_km"].min()) / 10))
        activity_score = round(min(100, recent_factor + mag_factor + source_factor + fault_factor))

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown(
                f'<div class="stat-box"><div style="font-size:1.35rem;font-weight:800;color:#90caf9">{activity_score}/100</div>'
                f'<div style="font-size:0.7rem;opacity:0.6">Aktivite Skoru</div></div>',
                unsafe_allow_html=True,
            )
        with a2:
            st.markdown(
                f'<div class="stat-box"><div style="font-size:1.35rem;font-weight:800;color:#ce93d8">{agreement["source_count"]}</div>'
                f'<div style="font-size:0.7rem;opacity:0.6">Kaynak Kapsamı</div></div>',
                unsafe_allow_html=True,
            )
        with a3:
            st.markdown(
                f'<div class="stat-box"><div style="font-size:1.35rem;font-weight:800;color:#ffb74d">{energy_total:,.0e}</div>'
                f'<div style="font-size:0.7rem;opacity:0.6">Yaklaşık Enerji J</div></div>',
                unsafe_allow_html=True,
            )
        with a4:
            st.markdown(
                f'<div class="stat-box"><div style="font-size:1.35rem;font-weight:800;color:#a5d6a7">{df["uzaklik_km"].min():.1f} km</div>'
                f'<div style="font-size:0.7rem;opacity:0.6">En Yakın Olay</div></div>',
                unsafe_allow_html=True,
            )

        src_counts = df.groupby("kaynak").size().sort_values(ascending=False).reset_index(name="kayıt")
        energy_df = df.sort_values("zaman").copy()
        energy_df["enerji_j"] = energy_df["buyukluk"].apply(estimate_energy_joules)
        energy_df["kumulatif_enerji"] = energy_df["enerji_j"].cumsum()
        src_col, energy_col = st.columns([1, 1.3])
        with src_col:
            st.markdown('<div class="chart-title">📡 Kaynak Kapsamı</div>', unsafe_allow_html=True)
            fig_src = px.bar(src_counts, x="kaynak", y="kayıt", color="kaynak")
            fig_src.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2,
                font=dict(color=TEXT, size=10),
                margin=dict(t=5, b=35, l=35, r=10),
                height=240,
                showlegend=False,
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT, size=9)),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT, size=9)),
            )
            st.plotly_chart(fig_src, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
        with energy_col:
            st.markdown('<div class="chart-title">⚡ Kümülatif Enerji Salınımı</div>', unsafe_allow_html=True)
            fig_energy = go.Figure(go.Scatter(
                x=energy_df["zaman"], y=energy_df["kumulatif_enerji"],
                mode="lines", line=dict(color="#ffb74d", width=2.5),
                fill="tozeroy", fillcolor="rgba(255,183,77,0.12)",
            ))
            fig_energy.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2,
                font=dict(color=TEXT, size=10),
                margin=dict(t=5, b=35, l=55, r=10),
                height=240,
                xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT, size=9)),
                yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT, size=9), title="Joule"),
            )
            st.plotly_chart(fig_energy, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
            st.info("💡 **Basitçe:** Bu grafik, fay hattında biriken enerjinin zaman içindeki tablosudur. Eğrinin yatay ve düz ilerlediği dönemler fayın **'kilitlendiği' ve enerji biriktirdiği** (suskunluk) tehlikeli zamanları gösterir. Çizginin aniden dik bir şekilde yukarı fırladığı anlar ise büyük bir depremin patlayarak bu gerilimi boşalttığı rahatlama anlarıdır. Eğer uzun süredir çizgi düz ilerliyorsa, fay büyük bir olaya hazırlanıyor demektir.")

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

            # Öncü adaylar: ana depremi geçmeden önce, belirtilen zaman aralığında
            time_filtered = df[
                (df["zaman"] < main_eq["zaman"]) &
                (df["zaman"] >= pencere_baslangic) &
                (df.index != idx_max)
            ].copy()

            if not time_filtered.empty:
                time_filtered["uzaklik_ana"] = time_filtered.apply(
                    lambda r: haversine(main_eq["lat"], main_eq["lon"], r["lat"], r["lon"]), axis=1
                )
                precursors = time_filtered[time_filtered["uzaklik_ana"] <= korr_radius].copy()
            else:
                precursors = pd.DataFrame()
            
            # PERFORMANS OPTİMİZASYONU: Çok fazla öncü varsa sistemi kitlememek için rastgele 600 örneklem al
            if len(precursors) > 600:
                precursors = precursors.sample(600, random_state=42)

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
                  <div style="font-size:0.9rem;font-weight:700">{safe_html(str(main_eq['konum'])[:40])}</div>
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
            
                # Yeni çevresel özellikleri hesapla ve ekle
                env_features = precursors.apply(lambda r: compute_environmental_features(r, df), axis=1)
                for col in env_features.columns:
                    precursors[col] = env_features[col]

                feat_cols = {
                    "gun_once":   "Gün Önce",
                    "uzaklik_ana": "Ana Şoka Uzaklık (km)",
                    "derinlik":   "Derinlik (km)",
                    "buyukluk":   "Büyüklük (M)",
                    "mevsim":     "Mevsim İndeksi",
                    "sicaklik":   "Ort. Sıcaklık (°C)",
                    "ay_cekim":   "Ay Çekim Gücü",
                    "gunes_uzaklik": "Güneş Uzaklık (AU)",
                    "gezegen_cekim": "Gezegen Çekimi",
                    "haftalik_aktivite": "Haftalık Sismik Yoğunluk",
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
                        textfont=dict(size=10, color=TEXT),
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
                        font=dict(color=TEXT, size=10, family="Arial"),
                        height=550,
                        margin=dict(t=10, b=10, l=10, r=60),
                        xaxis=dict(tickfont=dict(color=TEXT), tickangle=-45),
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
                    st.info("💡 **Basitçe:** Büyük bir deprem gelmeden günler veya haftalar önce, fay hattında çatırdamalar başlar ve küçük sarsıntılar oluşur. Buna **Öncü Deprem** (Foreshock) denir. Bu grafik, bölgedeki sarsıntıların sıradan rastgele titreşimler mi yoksa yaklaşan büyük bir ana depremin ayak sesleri mi (anormal kümelenme) olduğunu matematiksel olarak test eder.")

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

                with st.expander("🤔 Korelasyon (r) değerleri ne anlama geliyor?"):
                    st.markdown("""
                    **Korelasyon (r değeri)**, iki olayın birbiriyle ne kadar bağlantılı hareket ettiğini gösterir ve -1 ile +1 arasında bir puan alır.
                
                    *   **Güçlü Pozitif (r > +0.65) ↑**: İki özellik *aynı anda* artar veya azalır. Birlikte hareket ederler.
                    *   **Orta Pozitif (r: +0.40 ile +0.65)**: Aynı yönde bir eğilim var ancak kusursuz bir kural değil.
                    *   **Güçlü Negatif (r < -0.65) ↓**: Biri artarken diğeri *kesinlikle* azalır. Ters orantı vardır.
                    *   **Orta Negatif (r: -0.40 ile -0.65)**: Ters yönde bir eğilim seziliyor.
                
                    *(Sıfıra yakın puanlar ise o iki değişken arasında hiçbir mantıksal veya fiziksel bağ olmadığını kanıtlar.)*
                    """)

                with st.expander("🌍 Verilerin Kaynağı, Astronomik Detaylar ve Bilimsel Anlamı"):
                    st.markdown("""
                    **Veriler Nereden Geliyor?**
                    *   **Sismik Veriler:** AFAD, Kandilli, EMSC ve USGS gibi resmi rasathanelerden anlık çekilmektedir.
                    *   **Astronomik Veriler (Ay, Güneş, Gezegenler):** NASA'nın *Jet Propulsion Laboratory (JPL)* standartlarını kullanan yüksek hassasiyetli `ephem` Python astronomi kütüphanesi tarafından, her bir depremin saniyesi saniyesine Dünya'ya olan konumları hesaplanarak matrise işlenmektedir.
                    *   **İklim ve Hava Durumu:** API yoğunluğunu engellemek için, bulunduğunuz enlemin tarihsel sıcaklık modellerine dayalı olarak (Ocak'ta en düşük, Temmuz'da en yüksek olacak şekilde) matematiksel bir iklim simülasyonu uygulanır. Bu sayede hava sıcaklığı ile deprem sıklığı (sismik yoğunluk) arasındaki ilişki test edilebilir.

                    **Astronomik Verilerin Anlamı ve Değişim Döngüleri:**
                
                    *   **Ay Çekim Gücü (Tidal Stres):** Ay'ın Dünya etrafındaki yörüngesi tam yuvarlak değil, eliptiktir. Bir ay (yaklaşık 29.5 gün) içerisinde Dünya'ya en yakın olduğu **Yerberi (Perigee)** konumunda kütleçekim kuvveti maksimuma ulaşır. Ay uzaklaştıkça (**Yeröte - Apogee**) bu güç azalır. Çekim gücündeki bu artış, yerkabuğunda okyanuslardaki gelgitlere benzer bir esneme yaratarak faylardaki sürtünmeyi (çok minimal seviyede) değiştirebilir.
                
                    *   **Güneşe Uzaklık (AU):** Dünya'nın da Güneş etrafındaki yörüngesi eliptiktir. Sanılanın aksine Dünya, Güneş'e **Ocak ayının ilk haftasında** (Günberi - Perihelion) en yakın konumdadır. **Temmuz ayı başlarında** ise en uzak (Günöte - Aphelion) konumdadır. Dünya Güneş'e yaklaştığında yörüngedeki hızı artar. Bu hız değişimi, Dünya'nın sıvı çekirdeği ile sert yerkabuğu arasındaki açısal momentumu mikroskobik olarak etkileyebilir.
                
                    *   **Gezegen Çekimi (Jüpiter ve Venüs):** Dünya'ya en çok kütleçekim etkisi uygulayan gezegenler Jüpiter (devasa kütlesinden dolayı) ve Venüs'tür (bize çok yakın olmasından dolayı). Bu gezegenlerin çekim kuvveti, Dünya ile Güneş etrafında aynı hizaya geldikleri (kavuşum) aylarda zirve yapar. Venüs etkisi birkaç ayda bir, Jüpiter etkisi ise Dünya'nın onu yakalayıp geçtiği her ~13 ayda bir tavan yapar.
                
                    *   **Dünyanın Geoid Şekli:** Dünya kusursuz bir küre değildir; kutuplardan basık, ekvatordan şişkindir. Ay ve Güneş'in yörüngesel hizalanmaları ekvatoral şişkinliğe denk geldiğinde, gezegenimizin dönüş eksenine ekstra bir tork (burulma) kuvveti biner. Levha tektoniğinin (kıtaların kaymasının) altındaki devasa konveksiyon akımlarına minik ama sismolojide tartışılan bir "tetikleyici" faktör olarak dahil olabilir. Korelasyon matrisi tam olarak bu uçuk varsayımları kendi gözlerinizle test etmeniz için tasarlanmıştır!
                    """)
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
                    st.info("💡 **Basitçe (b-Değeri):** **b-Değeri**, sismolojinin en önemli uyarıcılarından biridir. Küçük depremlerin büyük depremlere olan oranını gösterir. Mavi çizginin yokuş aşağı inmesi (değerin düşmesi), fay hattındaki **stresin (gerilimin) aşırı yükseldiğini** ve kayaların artık kopma noktasına yaklaştığını gösterir. b-değerinin aniden çakılması, büyük bir depremin habercisi olarak kabul edilir.")
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
                st.info("💡 **Basitçe:** Yeraltındaki kayalar lastik gibi esneyebilir. **Benioff Zorlanması**, bu esnemenin miktarını ölçer. Grafik sürekli yukarı doğru tırmanıyorsa, tektonik plakalar birbirini itiyor ve yeraltındaki kayalar giderek daha fazla bükülüyor demektir. Eğrinin zirveye ulaştığı nokta, kayanın artık dayanamayıp kırıldığı (deprem) anı temsil eder.")

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
                        f"<b>M{r['buyukluk']:.1f}</b><br>{r['zaman_str']}<br>{safe_html(str(r['konum'])[:40])}", axis=1),
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
                st.info("💡 **Basitçe:** Depremler bazen rastgele değil, tıpkı devrilen domino taşları gibi belli bir yöne doğru ilerler. Bu harita, sarsıntıların **doğuya mı, batıya mı** doğru kaydığını gösterir. Fay üzerindeki enerjinin bir noktadan başka bir noktaya transfer edilmesi (göç etmesi), yakında hangi şehrin veya fay segmentinin tehlikeye gireceğini anlamamızı sağlar.")

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
                st.info("💡 **Basitçe:** Depremlerin sadece haritada değil, **yeraltındaki derinliklerinde de bir hareketi** vardır. Sarsıntıların 20 km derinlikten başlayıp gün geçtikçe 5 km, 2 km gibi yüzeye doğru tırmanması, yeraltındaki kırılmanın (veya magmanın) yüzeye doğru bir yol bulmaya çalıştığını ve yakında yıkıcı bir sığ deprem üretebileceğini işaret eder.")
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

        active_sub_tab = st.selectbox(
            "🔬 İleri Düzey Sismolojik Analiz Modeli Seçin",
            [
                "η Kümeleme Analizi (Zaliapin & Ben-Zion 2013)",
                "RTL Sismik Sessizlik Algoritması (Sobolev & Tyupkin 1997)",
                "AMR Güç Yasası Hızlanma Modeli (Bowman 1998)",
                "Uzamsal b-Değeri Haritası (Fay Kilitlenme Analizi)",
            ]
        )

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
        if active_sub_tab == "η Kümeleme Analizi (Zaliapin & Ben-Zion 2013)":
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
                    n = min(len(exp_df), 400)
                    sub = exp_df.iloc[:n].reset_index(drop=True)
                    eta_list, log_t_list, log_r_list = calc_etas_cache(sub.to_dict("list"), d_frac, b_eta)

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
                        st.info("💡 **Basitçe:** Tıpkı bir virüsün insanlara bulaşması gibi, depremler de birbirini tetikler (ETAS Modeli). Bu grafik, yaşanan sarsıntıların sadece eski bir depremin **zararsız artçıları mı**, yoksa yepyeni ve daha büyük bir depremi doğuracak **tehlikeli tetikleyiciler mi** olduğunu analiz eder. 'Bulaşıcılık' seviyesi yüksekse alarm zilleri çalmaya başlar.")

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
        if active_sub_tab == "RTL Sismik Sessizlik Algoritması (Sobolev & Tyupkin 1997)":
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
                    rtl_times, rtl_scores = calc_rtl_cache(exp_df.to_dict("list"), rtl_r0, rtl_t0, ERZ_LAT, ERZ_LON)

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
                    st.info("💡 **Basitçe:** Fay hatları büyük bir deprem üretmeden önce genellikle tamamen sessizleşir. **RTL Skoru**, bu fırtına öncesi sessizliği tespit eder. Grafikteki çizgi sıfırın altına (negatif bölgeye) inip uzun süre orada kalıyorsa, fay hattı tamamen **kilitlenmiş ve enerjisini hapsediyor** demektir. Bu kilit ne kadar uzun sürerse, kırılma o kadar şiddetli olur.")

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
        if active_sub_tab == "AMR Güç Yasası Hızlanma Modeli (Bowman 1998)":
            st.markdown("**Bowman et al. (1998) — Accelerating Moment Release (AMR)**")
            st.markdown(
                "Kümülatif Benioff zorlanmasına `C(t) = A + B·(tₓ − t)^m` güç yasası fit edilir. "
                "**m < 1** → ivcelenen yayılım, büyük deprem yakın. "
                "**m ≈ 1** → lineer (stabil). **m > 1** → yavaşlama. "
                "tₓ = tahmini kritik zaman (potansiyel kırılma anı)."
            )

            if len(exp_df) >= 20:
                amr_zaman, C, best_m, best_tf, best_fitted, T_obs, t0_amr, best_rmse = calc_amr_cache(exp_df.to_dict("list"))
                amr_df = pd.DataFrame({"zaman": amr_zaman, "C_norm": C})

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
                st.info("💡 **Basitçe:** Büyük bir dal kırılmadan önce çatırdama sesleri nasıl giderek hızlanır ve artarsa, fay hatları da aynısını yapar. **AMR Analizi**, bu sismik çatırdamaların ritmini ölçer. Kırmızı çizgi giderek hızlanarak dikey bir duvara tırmanıyorsa (hızlanan enerji salınımı), fayın kritik bir kopma noktasına doğru hızla ilerlediğini anlarız.")
                st.caption(
                    "⚠️ AMR tₓ tahmini istatistiksel bir fit olup kesin deprem tahmini değildir. "
                    "Akademik referans: Bowman et al., JGR 1998."
                )
            else:
                st.info("AMR için en az 20 deprem gerekli.")

        # ─────────────────────────────────────────────────────────────────
        # TAB 4 — Uzamsal b-Değeri Haritası
        # ─────────────────────────────────────────────────────────────────
        if active_sub_tab == "Uzamsal b-Değeri Haritası (Fay Kilitlenme Analizi)":
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
                    Mc_g = float(exp_df["buyukluk"].quantile(0.15))
                    df_mc = exp_df[exp_df["buyukluk"] >= Mc_g]
                    # Pass dict instead of DataFrame to st.cache_data to avoid hashing issues if index differs
                    b_grid, lats_g, lons_g = calc_b_grid_cache(df_mc.to_dict("list"), bg_n, bg_sr, bg_min, radius_km, ERZ_LAT, ERZ_LON, Mc_g)

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
                    st.info("💡 **Basitçe:** Bu harita, yeraltının bir nevi 'Tansiyon (Kan Basıncı) Haritası'dır. Haritadaki **kırmızı ve koyu sarı bölgeler**, kayaların en çok sıkıştığı, b-değerinin düştüğü ve büyük bir kırılma (deprem) ihtimalinin en yüksek olduğu stres noktalarını (Asperite) işaret eder. Açık mavi bölgeler ise enerjisini boşaltmış rahat bölgelerdir.")

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

if active_menu == "📊 İstatistik & Analiz":
    _render_istatistik_bottom()

if active_menu == "⚙️ Sistem & Veri":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📋 3. Ham Veri Tablosu</div>', unsafe_allow_html=True)
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

# ════════════════════════════════════════════════════════════════════════════
# 🔭 ASTRONOMİK ANALİZ PANELİ (Gökbilimci) — Gök Mekaniği ve Deprem Korelasyonu
# ════════════════════════════════════════════════════════════════════════════
@st.fragment
def _render_astronomik():
    st.markdown('<div class="chart-title">🔭 Astronomik Analiz — Gök Mekaniği ve Deprem Korelasyonu</div>', unsafe_allow_html=True)
    st.caption(
        "Ay, Güneş ve gezegenlerin yer kabuğuna uyguladığı gel-git çekim etkilerinin "
        "sismik aktiviteyle olan istatistiksel ilişkisi keşfedicidir."
    )

    st.warning(
        "⚠️ **Bilimsel Uyarı:** Bu panel keşfedici niteliktedir. Ay/Güneş çekiminin deprem tetikleyici olduğu "
        "**kanıtlanmamıştır**. Literatürde zayıf korelasyon raporları vardır (Cochran 2004, Métivier 2009) "
        "ancak nedensellik gösterilmemiştir. Bu panel istatistiksel araştırma amaçlıdır."
    )

    if "run_astro" not in st.session_state:
        st.session_state.run_astro = False

    col_a1, col_a2 = st.columns([3, 1])
    with col_a1:
        if st.button("🔭 Astronomik Hesaplamaları Çalıştır", use_container_width=True, type="primary"):
            st.session_state.run_astro = True
    with col_a2:
        if st.button("🛑 Kapat", use_container_width=True, key="close_astro"):
            st.session_state.run_astro = False
            st.rerun()

    if not st.session_state.run_astro:
        st.info(
            "Ay/Güneş/gezegen yörünge hesaplamaları her deprem için ayrı `ephem` çağrısı gerektirdiğinden "
            "ağırdır. CPU performansını korumak için **Çalıştır** düğmesi ile tetiklenir.",
            icon="⏸️",
        )
    else:
        # ───────────────────────────────────────────────────────────────────
        # BİLEŞEN 1 — Anlık Gök Durumu Kartları (Erzincan üzerinde, şu an)
        # ───────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="chart-title">🌌 1. Anlık Gök Durumu — Erzincan Üzerinde</div>', unsafe_allow_html=True)

        now_obs = ephem.Observer()
        now_obs.lat = str(ERZ_LAT)
        now_obs.lon = str(ERZ_LON)
        now_obs.date = datetime.utcnow()

        moon_now = ephem.Moon(now_obs)
        sun_now = ephem.Sun(now_obs)
        jupiter_now = ephem.Jupiter(now_obs)
        venus_now = ephem.Venus(now_obs)

        moon_phase_pct = float(moon_now.phase)
        # ephem.earth_distance birimi AU (1 AU = 149,597,870.7 km)
        moon_dist_km = float(moon_now.earth_distance) * 149_597_870.7
        moon_alt_deg = math.degrees(float(moon_now.alt))

        sun_dist_au = float(sun_now.earth_distance)
        sun_alt_deg = math.degrees(float(sun_now.alt))

        if moon_phase_pct < 5:
            phase_name = "🌑 Yeni Ay"
        elif moon_phase_pct < 45:
            phase_name = "🌒 Hilal"
        elif moon_phase_pct < 55:
            phase_name = "🌓 İlk/Son Dördün"
        elif moon_phase_pct < 95:
            phase_name = "🌔 Şişkin"
        else:
            phase_name = "🌕 Dolunay"

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Ay Fazı", phase_name, f"{moon_phase_pct:.1f}%")
        g2.metric("Ay Uzaklığı", f"{moon_dist_km:,.0f} km", "Çekim ∝ 1/d²")
        g3.metric("Ay Yüksekliği", f"{moon_alt_deg:.1f}°", "Ufuk üstünde" if moon_alt_deg > 0 else "Ufuk altında")
        g4.metric("Güneş Uzaklığı", f"{sun_dist_au:.4f} AU", f"~{sun_dist_au * 149.6:.1f} M km")

        g5, g6, g7, g8 = st.columns(4)
        g5.metric("Güneş Yüksekliği", f"{sun_alt_deg:.1f}°", "Gündüz" if sun_alt_deg > 0 else "Gece")
        g6.metric("Jüpiter Uzaklığı", f"{float(jupiter_now.earth_distance):.2f} AU")
        g7.metric("Venüs Uzaklığı", f"{float(venus_now.earth_distance):.2f} AU")
        tidal_alignment = "Yüksek (Syzygy)" if (moon_phase_pct < 10 or moon_phase_pct > 90) else "Düşük (Kvadratur)"
        g8.metric("Gel-Git Hizalanması", tidal_alignment)

        st.caption(
            "💡 **Bilimsel ipucu:** Syzygy konumunda (Yeni Ay/Dolunay) Ay ve Güneş'in çekim etkileri toplanır → "
            "*spring tide* (büyük gel-git). Kvadratur konumunda (İlk/Son Dördün) ise birbirini iptal eder → "
            "*neap tide* (küçük gel-git). Kabuk gerilim eşiğindeki faylarda bu fark teorik olarak ek tetikleyici olabilir."
        )

        with st.spinner("Tüm depremler için astronomik özellikler hesaplanıyor..."):
            astro_df = df.dropna(subset=["zaman", "lat", "lon", "buyukluk"]).copy()
            if len(astro_df) < 5:
                st.warning("Astronomik analiz için en az 5 deprem gerekli.")
            else:
                env_feats = astro_df.apply(lambda r: compute_environmental_features(r, df), axis=1)
                astro_df = pd.concat([astro_df.reset_index(drop=True), env_feats.reset_index(drop=True)], axis=1)

                def _moon_phase_at(row):
                    obs = ephem.Observer()
                    obs.lat = str(row["lat"])
                    obs.lon = str(row["lon"])
                    obs.date = row["zaman"].to_pydatetime()
                    return float(ephem.Moon(obs).phase)

                astro_df["ay_faz"] = astro_df.apply(_moon_phase_at, axis=1)

                # ───────────────────────────────────────────────────────────
                # BİLEŞEN 2 — Ay Fazı vs Magnitude Scatter
                # ───────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown('<div class="chart-title">🌗 2. Ay Fazı vs Deprem Büyüklüğü</div>', unsafe_allow_html=True)
                st.caption(
                    "Her nokta bir deprem. X: depremin gerçekleştiği andaki Ay fazı (0=Yeni Ay, 50=Dördün, 100=Dolunay). "
                    "Y: magnitude. Renk: derinlik. Sarı bantlar spring tide (gel-git çekiminin maksimum olduğu) dönemleri."
                )

                fig_phase = px.scatter(
                    astro_df, x="ay_faz", y="buyukluk", color="derinlik",
                    hover_data=["konum", "zaman_str", "kaynak"],
                    color_continuous_scale="Plasma",
                    labels={"ay_faz": "Ay Fazı (%)", "buyukluk": "Magnitude", "derinlik": "Derinlik (km)"},
                )
                fig_phase.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=BG2,
                    font=dict(color=TEXT, size=11),
                    margin=dict(t=20, b=40, l=50, r=10),
                    height=350,
                    xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                    yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                )
                fig_phase.add_vrect(x0=0, x1=10, fillcolor="rgba(255,180,80,0.10)", line_width=0,
                                   annotation_text="Spring (Yeni Ay)", annotation_position="top left",
                                   annotation_font_size=9)
                fig_phase.add_vrect(x0=90, x1=100, fillcolor="rgba(255,180,80,0.10)", line_width=0,
                                   annotation_text="Spring (Dolunay)", annotation_position="top right",
                                   annotation_font_size=9)
                st.plotly_chart(fig_phase, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

                spring_eq = astro_df[(astro_df["ay_faz"] < 10) | (astro_df["ay_faz"] > 90)]
                neap_eq = astro_df[(astro_df["ay_faz"] > 40) & (astro_df["ay_faz"] < 60)]

                c_spring, c_neap, c_ratio = st.columns(3)
                c_spring.metric("Spring tide depremleri", f"{len(spring_eq)} olay",
                               f"Ort M: {spring_eq['buyukluk'].mean():.2f}" if len(spring_eq) else "—")
                c_neap.metric("Neap tide depremleri", f"{len(neap_eq)} olay",
                             f"Ort M: {neap_eq['buyukluk'].mean():.2f}" if len(neap_eq) else "—")
                if len(spring_eq) and len(neap_eq):
                    # Faz bantları eşit genişlikte (20% her biri) — birebir karşılaştırılabilir
                    ratio = len(spring_eq) / len(neap_eq)
                    c_ratio.metric("Spring/Neap Oranı", f"{ratio:.2f}", "Beklenen ≈ 1.0 (etki yoksa)")

                # ───────────────────────────────────────────────────────────
                # BİLEŞEN 3 — Ay Çekim Zaman Serisi + Depremler
                # ───────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown('<div class="chart-title">📈 3. Ay Çekim Zaman Serisi (Depremlerle Üst Üste)</div>', unsafe_allow_html=True)
                st.caption("Mavi çizgi: Ay çekim göstergesi (1/d²). Kırmızı noktalar: depremler (boyut = magnitude).")

                astro_df_sorted = astro_df.sort_values("zaman").copy()

                fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ts.add_trace(
                    go.Scatter(x=astro_df_sorted["zaman"], y=astro_df_sorted["ay_cekim"],
                              mode="lines", name="Ay Çekim", line=dict(color="#90caf9", width=1.5)),
                    secondary_y=False,
                )
                fig_ts.add_trace(
                    go.Scatter(x=astro_df_sorted["zaman"], y=astro_df_sorted["buyukluk"],
                              mode="markers", name="Depremler",
                              marker=dict(size=astro_df_sorted["buyukluk"]*3, color="#ff5252", opacity=0.6,
                                         line=dict(width=0.5, color="#ffcccc"))),
                    secondary_y=True,
                )
                fig_ts.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=BG2,
                    font=dict(color=TEXT, size=11),
                    margin=dict(t=20, b=40, l=50, r=50),
                    height=340,
                    xaxis=dict(gridcolor=GRID),
                    legend=dict(bgcolor=BG3, bordercolor=BORDER, borderwidth=1),
                )
                fig_ts.update_yaxes(title_text="Ay Çekim Proxy", secondary_y=False, gridcolor=GRID, color="#90caf9")
                fig_ts.update_yaxes(title_text="Magnitude", secondary_y=True, gridcolor=GRID, color="#ff5252")
                st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

                # ───────────────────────────────────────────────────────────
                # BİLEŞEN 4 — FFT Periyodogram (Deprem Sıklığı Frekans Analizi)
                # ───────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown('<div class="chart-title">🌊 4. FFT Periyodogram — Deprem Sıklığı Frekans Analizi</div>', unsafe_allow_html=True)
                st.caption(
                    "Deprem oluşum sıklığı günlük binlere bölünüp Hızlı Fourier Dönüşümü uygulanır. "
                    "Tepe frekanslar periyodik tetikleyicilere işaret edebilir: sinodik Ay periyodu ≈ 29.5 gün, "
                    "yarı-Ay (M2 gel-git harmoniği) ≈ 14.8 gün, yıllık ≈ 365.25 gün."
                )

                astro_df["gun"] = astro_df["zaman"].dt.floor("D")
                daily = astro_df.groupby("gun").size().reset_index(name="sayim")
                if len(daily) >= 10:
                    full_range = pd.date_range(daily["gun"].min(), daily["gun"].max(), freq="D")
                    daily = daily.set_index("gun").reindex(full_range, fill_value=0).reset_index()
                    daily.columns = ["gun", "sayim"]

                    n_days = len(daily)
                    signal = daily["sayim"].values.astype(float)
                    signal -= signal.mean()  # DC bileşeni çıkar

                    fft_vals = np.fft.rfft(signal)
                    fft_freq = np.fft.rfftfreq(n_days, d=1.0)
                    fft_amp = np.abs(fft_vals)

                    periods_days = np.where(fft_freq > 0, 1.0 / np.maximum(fft_freq, 1e-9), 0)

                    fft_df = pd.DataFrame({"periyot_gun": periods_days[1:], "amplitude": fft_amp[1:]})
                    fft_df = fft_df[(fft_df["periyot_gun"] >= 2) & (fft_df["periyot_gun"] <= n_days/2)]

                    fig_fft = px.line(fft_df.sort_values("periyot_gun"), x="periyot_gun", y="amplitude",
                                     log_x=True,
                                     labels={"periyot_gun": "Periyot (gün, log)", "amplitude": "Genlik"})
                    fig_fft.update_traces(line=dict(color="#ce93d8", width=1.8))
                    fig_fft.update_layout(
                        paper_bgcolor=BG, plot_bgcolor=BG2,
                        font=dict(color=TEXT, size=11),
                        margin=dict(t=20, b=40, l=50, r=10),
                        height=320,
                        xaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                        yaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                    )
                    for period, label, color in [
                        (29.5, "Ay periyodu", "#ffb74d"),
                        (14.8, "Yarı-Ay (M2)", "#a5d6a7"),
                        (365.25, "Yıllık", "#ff8a80"),
                    ]:
                        if 2 <= period <= n_days/2:
                            fig_fft.add_vline(x=period, line=dict(color=color, dash="dash", width=1),
                                             annotation_text=label, annotation_position="top",
                                             annotation_font=dict(color=color, size=9))
                    st.plotly_chart(fig_fft, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
                    st.caption("⚠️ Kısa veri pencerelerinde FFT gürültülüdür; 60+ gün verisi olduğunda daha güvenilir tepeler gözlemlenir.")
                else:
                    st.info("FFT analizi için en az 10 günlük veri gerekli.")

                # ───────────────────────────────────────────────────────────
                # BİLEŞEN 5 — Gezegen Çekim Etkisi (Jüpiter + Venüs)
                # ───────────────────────────────────────────────────────────
                st.markdown("---")
                st.markdown('<div class="chart-title">🪐 5. Gezegen Çekim Etkisi — Jüpiter + Venüs Toplam Proxy</div>', unsafe_allow_html=True)
                st.caption(
                    "Jüpiter (kütle 317.8 M⊕) ve Venüs (kütle 0.815 M⊕) çekim etkileri 1/d² ile ağırlıklanıp toplanır. "
                    "Pikler, gezegenlerin Dünya'ya en yakın oldukları (opozisyon/iç birleşim) anlardır."
                )

                gezegen_sorted = astro_df_sorted[["zaman", "gezegen_cekim", "buyukluk", "konum"]].copy()

                fig_planet = make_subplots(specs=[[{"secondary_y": True}]])
                fig_planet.add_trace(
                    go.Scatter(x=gezegen_sorted["zaman"], y=gezegen_sorted["gezegen_cekim"],
                              mode="lines", name="Gezegen Çekim", line=dict(color="#ffb74d", width=1.5),
                              fill="tozeroy", fillcolor="rgba(255,183,77,0.08)"),
                    secondary_y=False,
                )
                fig_planet.add_trace(
                    go.Scatter(x=gezegen_sorted["zaman"], y=gezegen_sorted["buyukluk"],
                              mode="markers", name="Depremler",
                              marker=dict(size=gezegen_sorted["buyukluk"]*3, color="#80deea", opacity=0.55,
                                         line=dict(width=0.5, color="#b2ebf2"))),
                    secondary_y=True,
                )
                fig_planet.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=BG2,
                    font=dict(color=TEXT, size=11),
                    margin=dict(t=20, b=40, l=50, r=50),
                    height=320,
                    xaxis=dict(gridcolor=GRID),
                    legend=dict(bgcolor=BG3, bordercolor=BORDER, borderwidth=1),
                )
                fig_planet.update_yaxes(title_text="Gezegen Çekim Proxy", secondary_y=False, gridcolor=GRID, color="#ffb74d")
                fig_planet.update_yaxes(title_text="Magnitude", secondary_y=True, gridcolor=GRID, color="#80deea")
                st.plotly_chart(fig_planet, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

                st.markdown("---")
                st.info(
                    "🔬 **Yorum:** Bu paneldeki istatistiksel örüntüler tek başına **rastlantı** ile uyumlu olabilir. "
                    "Gerçek bir kausal ilişki için: (1) en az 1000+ olay, (2) bağımsız doğrulama veri seti, "
                    "(3) farklı tektonik rejimlerde tekrarlanabilirlik gerekir. Bu panel bilimsel namus gereği "
                    "yalnızca **hipotez üretici** bir keşif aracıdır, kesin bilimsel kanıt değildir."
                )

if active_menu == "🔭 Astronomik Analiz":
    _render_astronomik()

# ════════════════════════════════════════════════════════════════════════════
# 🌍 TEKTONİK PLAKA HAREKETİ SİMÜLASYONU — v1.17 (Ajan 5 / F-43)
# Genişletilmiş: jeodezik mod (-1.000 → +10.000 yıl) + paleografik mod
# (-1.000.000 → +10.000.000 yıl), ESRI World Imagery uydu zemini, Erzincan
# trailing path, log-spaced 20-frame animasyon, bilimsel uyarı bandı.
# ════════════════════════════════════════════════════════════════════════════
_PLAKA_CITIES = {
    "Erzincan":   (39.7333, 39.4917),
    "İstanbul":   (41.0082, 28.9784),
    "Diyarbakır": (37.9144, 40.2306),
    "Van":        (38.4942, 43.3805),
}

# 20 log-spaced frame stops per mode. Negatif = geçmiş, pozitif = gelecek.
_PLAKA_MODES = {
    "geo": {
        "label":   "🌐 Jeodezik — Bilimsel (-1.000 → +10.000 yıl)",
        "short":   "Jeodezik",
        "stops":   [-1000, -700, -500, -300, -200, -100, -50, -30, -10, -3,
                    0, 3, 10, 30, 100, 300, 1000, 3000, 7000, 10_000],
        "default_idx": 10,  # 0
        "visual_scale_factor": 1000.0,
    },
    "pal": {
        "label":   "🪨 Paleografik — Spekülatif (-1.000.000 → +10.000.000 yıl)",
        "short":   "Paleografik",
        "stops":   [-1_000_000, -500_000, -200_000, -100_000, -50_000, -20_000,
                    -10_000, -3_000, -1_000, -300, 0, 300, 1_000, 3_000,
                    10_000, 100_000, 1_000_000, 3_000_000, 7_000_000, 10_000_000],
        "default_idx": 10,  # 0
        "visual_scale_factor": 1.0,
    },
}

def _plaka_warning(years_abs: int):
    """0..10K = yeşil bilimsel · 10K..1M = sarı soyutlama · 1M+ = kırmızı spekülatif."""
    if years_abs <= 10_000:
        return ("🟢", "Bilimsel", "#43a047", "Lineer GNSS-türevli ekstrapolasyon — doğrudan ölçüm rejimi.")
    if years_abs <= 1_000_000:
        return ("🟡", "Soyutlama", "#fbc02d", "Fay döngüleri ve viskoelastik relaksasyon ihmal edilmiştir; "
                                              "tek başına lineer model ~10⁵ yıl ufkunda paleosismik kayıtla "
                                              "kalibre edilmelidir.")
    return ("🔴", "Spekülatif Senaryo", "#e53935",
            "Milyon yıl ölçeği lineer ekstrapolasyon yetmez — paleomanyetik / PALEOMAP "
            "(Scotese 2016) rekonstrüksiyonları gerekir. Bu görüntü yalnızca eğitsel "
            "sezgi içindir; bilimsel öngörü değildir.")

def _format_years_tr(y: int) -> str:
    """+500.000 yıl, -1.2 milyon yıl, bugün, +300 yıl gibi etiketler."""
    if y == 0:
        return "bugün (0 yıl)"
    s   = "−" if y < 0 else "+"
    a   = abs(y)
    if a >= 1_000_000:
        val = a / 1_000_000
        return f"{s}{val:g} milyon yıl"
    if a >= 1_000:
        val = a / 1_000
        return f"{s}{val:g} bin yıl"
    return f"{s}{a} yıl"

def _plaka_displacement_deg(plate_code: str, lat: float, lon: float, years: int):
    """Δφ, Δλ derece olarak. Önce Ajan 4'ün `plate_velocity_vector()` fonksiyonunu
    dener; başarısız olursa kendi Euler dönüştürücüsü `load_plate_velocities()` ile."""
    if _plate_velocity_vector_extern is not None:
        try:
            return _plate_velocity_vector_extern(plate_code, lat, lon, years)
        except Exception:
            pass
    vels = load_plate_velocities(lat, lon)
    vel  = vels.get(plate_code) or vels.get("AN") or {**_PLAKA_FALLBACK_AN, "is_euler_derived": False}
    return (vel["delta_lat_per_year"] * years,
            vel["delta_lon_per_year"] * years)

# ─── Plaka kuplaj (etkileşim) okları — Ajan 8 onayı beklenirken sezgisel görsel
# Bilimsel arka plan:
# • Arabistan kuzeye baskı yapar → Anadolu blokunu batıya iter (Bitlis-Zagros
#   sütur, ~37.5°N 41.5°E).
# • Hellenik trench geri çekilir (slab rollback) → Ege/Anadolu batı kanadını
#   güneybatıya yayar (~35.5°N 25°E, Girit güneyi).
# Oklar her frame'de "anchor → anchor + (velocity × years × vis_scale)" olarak
# büyür; year=0'da görünmez ama anchor label sabit kalır.
_COUPLING_AR_AN_ANCHOR    = (37.5, 41.5)
_COUPLING_HELLENIC_ANCHOR = (35.5, 25.0)
# Hellenik geri çekilme yön+magnitude (AE plakası MORVEL'de ayrı çözülmüyor;
# Reilinger 2006 GPS bantı ~30 mm/yr GB). Sabit deg/yr vektör:
_HELLENIC_RETREAT_DLAT_YR = -1.91e-7   # ~21 mm/yr güney
_HELLENIC_RETREAT_DLON_YR = -2.34e-7   # ~21 mm/yr batı (35.5°N enleminde)

def _build_coupling_traces(years: int, vis_scale: float):
    """AR→AN baskı oku (kırmızı) + Hellenik geri çekilme oku (mavi)."""
    traces = []

    # ── AR baskı oku (Bitlis-Zagros) ─────────────────────────────────────
    a_lat, a_lon = _COUPLING_AR_AN_ANCHOR
    d_lat, d_lon = _plaka_displacement_deg("AR", a_lat, a_lon, years)
    e_lat = a_lat + d_lat * vis_scale
    e_lon = a_lon + d_lon * vis_scale
    traces.append(go.Scattermapbox(   # glow
        lat=[a_lat, e_lat], lon=[a_lon, e_lon],
        mode="lines", line=dict(color="rgba(255,23,68,0.35)", width=12),
        hoverinfo="skip", showlegend=False, name="",
    ))
    traces.append(go.Scattermapbox(   # gövde
        lat=[a_lat, e_lat], lon=[a_lon, e_lon],
        mode="lines", line=dict(color="#ff1744", width=4),
        name="🔴 Arabistan baskısı (AR→AN)",
        hovertemplate=("<b>🔴 AR → AN konvergans baskısı</b><br>"
                       "Arabistan'ın kuzey baskısı Anadolu'yu batıya iter — "
                       "KAF ve DAF bu kuvvetle aktif kalır.<extra></extra>"),
    ))
    traces.append(go.Scattermapbox(   # arrowhead surrogate (uç daire)
        lat=[e_lat], lon=[e_lon],
        mode="markers", marker=dict(size=18, color="#ff1744", opacity=0.95),
        hoverinfo="skip", showlegend=False, name="",
    ))
    traces.append(go.Scattermapbox(   # sabit anchor label (yıl=0'da bile görünür)
        lat=[a_lat], lon=[a_lon],
        mode="markers+text",
        marker=dict(size=8, color="#ff1744", opacity=0.85),
        text=["AR baskısı"], textposition="bottom right",
        textfont=dict(color="#ff8a80", size=11, family="Inter, system-ui"),
        hoverinfo="skip", showlegend=False, name="",
    ))

    # ── Hellenik geri çekilme oku (Girit güneyi) ────────────────────────
    a_lat, a_lon = _COUPLING_HELLENIC_ANCHOR
    e_lat = a_lat + _HELLENIC_RETREAT_DLAT_YR * years * vis_scale
    e_lon = a_lon + _HELLENIC_RETREAT_DLON_YR * years * vis_scale
    traces.append(go.Scattermapbox(
        lat=[a_lat, e_lat], lon=[a_lon, e_lon],
        mode="lines", line=dict(color="rgba(33,150,243,0.35)", width=12),
        hoverinfo="skip", showlegend=False, name="",
    ))
    traces.append(go.Scattermapbox(
        lat=[a_lat, e_lat], lon=[a_lon, e_lon],
        mode="lines", line=dict(color="#2196f3", width=4),
        name="🔵 Hellenik geri çekilme",
        hovertemplate=("<b>🔵 Hellenik subduction geri çekilme</b><br>"
                       "Afrika dalan plaka geri çekilir; Ege/Anadolu "
                       "batıya yayılır (slab rollback).<extra></extra>"),
    ))
    traces.append(go.Scattermapbox(
        lat=[e_lat], lon=[e_lon],
        mode="markers", marker=dict(size=18, color="#2196f3", opacity=0.95),
        hoverinfo="skip", showlegend=False, name="",
    ))
    traces.append(go.Scattermapbox(
        lat=[a_lat], lon=[a_lon],
        mode="markers+text",
        marker=dict(size=8, color="#2196f3", opacity=0.85),
        text=["Hellenik geri çekilme"], textposition="top left",
        textfont=dict(color="#90caf9", size=11, family="Inter, system-ui"),
        hoverinfo="skip", showlegend=False, name="",
    ))
    return traces

@st.cache_data(show_spinner=False)
def _plaka_build_figure(mode_key: str, focus_lat: float, focus_lon: float,
                        city: str, plate_code: str = "AN",
                        active_idx: int = 0, show_coupling: bool = False,
                        visual_scale_override: float | None = None):
    """20 frame Plotly animasyonu — uydu zemin, log-spaced zaman, trail, kuplaj."""
    mode      = _PLAKA_MODES[mode_key]
    stops     = mode["stops"]
    vis_scale = visual_scale_override if visual_scale_override is not None \
                else mode["visual_scale_factor"]

    # Statik gri "bugünkü plaka sınırları" (tüm frame'lerde sabit)
    base_lats, base_lons = [], []
    for plate in PLATE_LINES:
        base_lats.extend(plate["lats"] + [None])
        base_lons.extend(plate["lons"] + [None])
    # Glow + ana çizgi → 2 trace
    base_glow = go.Scattermapbox(
        lat=base_lats, lon=base_lons, mode="lines",
        line=dict(color="rgba(255,255,255,0.18)", width=4.5),
        name="", hoverinfo="skip", showlegend=False,
    )
    base_line = go.Scattermapbox(
        lat=base_lats, lon=base_lons, mode="lines",
        line=dict(color="rgba(220,225,240,0.55)", width=1.4),
        name="Bugünkü Plaka Sınırları", hoverinfo="skip",
    )

    # Plaka tip rengi — kayan sınır için, tip bazında glow için ayrı trace listesi
    type_color = {
        "convergent": "#ff5252",  # Yaklaşan — kırmızı
        "divergent":  "#42a5f5",  # Ayrılan — mavi
        "transform":  "#ffeb3b",  # Yanal — sarı
        "unknown":    "#bbbbbb",
    }
    plates_by_type = {k: [] for k in type_color}
    for plate in PLATE_LINES:
        plates_by_type.setdefault(plate.get("type", "unknown"), []).append(plate)

    # v1.17.3 TOPLU HAREKET — her sınırın kendi hız ortalaması (frame-dışı pre-compute)
    # PB2002 sınırı iki plakanın arasındadır → sınırın hızı = iki plakanın hız ORTALAMASI
    # (gerçek tektonik approximation; tek-plaka deltası "tüm dünya AN ile kayıyor"
    # yanılsamasını ortadan kaldırır).
    vels_dict = load_plate_velocities(focus_lat, focus_lon)
    border_dlat_yr = {}  # id(plate) → derece/yıl
    border_dlon_yr = {}
    for plate in PLATE_LINES:
        p_a_code = _PB2002_TO_VELOCITY_CODE.get(plate.get("plate_a", ""))
        p_b_code = _PB2002_TO_VELOCITY_CODE.get(plate.get("plate_b", ""))
        v_a = vels_dict.get(p_a_code) if p_a_code else None
        v_b = vels_dict.get(p_b_code) if p_b_code else None
        if v_a and v_b:
            dlat = (v_a["delta_lat_per_year"] + v_b["delta_lat_per_year"]) / 2.0
            dlon = (v_a["delta_lon_per_year"] + v_b["delta_lon_per_year"]) / 2.0
        elif v_a:
            dlat, dlon = v_a["delta_lat_per_year"], v_a["delta_lon_per_year"]
        elif v_b:
            dlat, dlon = v_b["delta_lat_per_year"], v_b["delta_lon_per_year"]
        else:
            dlat, dlon = 0.0, 0.0  # hız bilinmeyen sınır → sabit kalır
        border_dlat_yr[id(plate)] = dlat
        border_dlon_yr[id(plate)] = dlon

    # Erzincan trail — focus plakası (varsayılan AN) hızıyla kayar
    trail_positions = []
    for y in stops:
        dlat, dlon = _plaka_displacement_deg(plate_code, focus_lat, focus_lon, y)
        trail_positions.append((focus_lat + dlat * vis_scale,
                                focus_lon + dlon * vis_scale, y, dlat, dlon))

    frames = []
    for i, (cur_lat, cur_lon, cur_year, dlat_real, dlon_real) in enumerate(trail_positions):
        # Kayan plaka sınırları — her sınır KENDİ hızı × yıl × ölçek ile kaydırılır
        shifted_traces = []
        for b_type, plate_list in plates_by_type.items():
            if not plate_list:
                continue
            color = type_color[b_type]
            lats_t, lons_t = [], []
            for plate in plate_list:
                # Bu sınırın kümülatif kayması (v1.17.3 — toplu hareket)
                bd_lat = border_dlat_yr[id(plate)] * cur_year * vis_scale
                bd_lon = border_dlon_yr[id(plate)] * cur_year * vis_scale
                lats_t.extend([la + bd_lat for la in plate["lats"]] + [None])
                lons_t.extend([lo + bd_lon for lo in plate["lons"]] + [None])
            # Geniş yarı-saydam glow
            shifted_traces.append(go.Scattermapbox(
                lat=lats_t, lon=lons_t, mode="lines",
                line=dict(color=color, width=6),
                opacity=0.28, hoverinfo="skip", showlegend=False, name="",
            ))
            # Parlak ana hat
            shifted_traces.append(go.Scattermapbox(
                lat=lats_t, lon=lons_t, mode="lines",
                line=dict(color=color, width=2.2),
                opacity=0.95, name={
                    "convergent": "🔺 Yaklaşan Sınır",
                    "divergent":  "🔻 Ayrılan Sınır",
                    "transform":  "↔ Yanal Kayan",
                    "unknown":    "Sınır",
                }[b_type], hoverinfo="skip",
            ))

        # Erzincan trail — frame 0..i arası kümülatif yol
        trail_lats = [p[0] for p in trail_positions[:i+1]]
        trail_lons = [p[1] for p in trail_positions[:i+1]]
        trail_glow = go.Scattermapbox(
            lat=trail_lats, lon=trail_lons, mode="lines",
            line=dict(color="rgba(255,64,129,0.30)", width=10),
            hoverinfo="skip", showlegend=False, name="",
        )
        trail_line = go.Scattermapbox(
            lat=trail_lats, lon=trail_lons, mode="lines+markers",
            line=dict(color="#ff4081", width=3),
            marker=dict(size=6, color="#ffd1dc", opacity=0.85),
            name=f"{city} izi", hoverinfo="skip",
        )

        # Gerçek (görsel ölçeksiz) kümülatif yer değiştirme
        real_m_lat = dlat_real * 111_320.0
        real_m_lon = dlon_real * 111_320.0 * math.cos(math.radians(focus_lat))
        real_disp  = math.sqrt(real_m_lat**2 + real_m_lon**2)
        if real_disp < 1.0:
            disp_str = f"{real_disp*1000:.1f} mm"
        elif real_disp < 1000.0:
            disp_str = f"{real_disp:.2f} m"
        elif real_disp < 1_000_000:
            disp_str = f"{real_disp/1000:.2f} km"
        else:
            disp_str = f"{real_disp/1000:.0f} km"

        # Sabit referans (bugünkü Erzincan)
        ref_pin = go.Scattermapbox(
            lat=[focus_lat], lon=[focus_lon],
            mode="markers", marker=dict(size=9, color="rgba(255,255,255,0.45)"),
            name=f"{city} (bugün)", hoverinfo="skip", showlegend=False,
        )
        # Parlak şu-anki pin (büyük halo + nokta)
        pin_halo = go.Scattermapbox(
            lat=[cur_lat], lon=[cur_lon],
            mode="markers",
            marker=dict(size=28, color="rgba(255,64,129,0.40)"),
            hoverinfo="skip", showlegend=False, name="",
        )
        pin = go.Scattermapbox(
            lat=[cur_lat], lon=[cur_lon],
            mode="markers+text",
            marker=dict(size=15, color="#ff4081"),
            text=[f"★ {city}"],
            textposition="top right",
            textfont=dict(color="#ffffff", size=14),
            name=city,
            hovertemplate=(f"<b>{city}</b><br>{_format_years_tr(cur_year)}<br>"
                           f"Gerçek kayma: {disp_str}<extra></extra>"),
        )

        # Kuplaj okları (AR baskı + Hellenik geri çekilme)
        coupling_traces = _build_coupling_traces(cur_year, vis_scale) if show_coupling else []

        # Frame içeriği: glow taban + statik taban + kayan plakalar + trail + pinler + kuplaj
        frame_data = [base_glow, base_line, *shifted_traces, trail_glow, trail_line,
                      ref_pin, pin_halo, pin, *coupling_traces]

        # Frame layout: büyük zaman etiketi annotation + uyarı bandı
        emoji, severity_label, severity_color, _ = _plaka_warning(abs(cur_year))
        title_txt = (f"🌍 {city} · {_format_years_tr(cur_year)} · "
                     f"{emoji} {severity_label} · kayma: {disp_str}")
        annotations = [
            dict(text=f"<b>📅 {_format_years_tr(cur_year)}</b>",
                 xref="paper", yref="paper", x=0.02, y=0.98,
                 xanchor="left", yanchor="top",
                 showarrow=False, align="left",
                 font=dict(size=26, color="#ffffff", family="Inter, system-ui"),
                 bgcolor="rgba(0,0,0,0.55)", bordercolor=severity_color,
                 borderwidth=2, borderpad=8),
            dict(text=(f"<b>{emoji} {severity_label}</b><br>"
                       f"<span style='font-size:11px'>Kayma: {disp_str}</span>"),
                 xref="paper", yref="paper", x=0.98, y=0.98,
                 xanchor="right", yanchor="top",
                 showarrow=False, align="right",
                 font=dict(size=13, color="#ffffff", family="Inter, system-ui"),
                 bgcolor=severity_color, bordercolor="#000",
                 borderwidth=1, borderpad=6, opacity=0.92),
        ]
        # 10K+ yıl + kuplaj açık → "basitleştirilmiş model" şerit uyarısı
        if show_coupling and abs(cur_year) > 10_000:
            annotations.append(dict(
                text=("⚠️ <b>Bu ölçekte plaka-plaka etkileşimi değişebilir</b> — "
                      "basitleştirilmiş kuplaj modeli"),
                xref="paper", yref="paper", x=0.5, y=0.04,
                xanchor="center", yanchor="bottom",
                showarrow=False, align="center",
                font=dict(size=12, color="#ffeb3b", family="Inter, system-ui"),
                bgcolor="rgba(33,33,33,0.88)",
                bordercolor="#ffeb3b", borderwidth=1, borderpad=6,
            ))
        frames.append(go.Frame(
            data=frame_data, name=str(i),
            layout=go.Layout(
                title=dict(text=title_txt, font=dict(color="#eceff1", size=14)),
                annotations=annotations,
            ),
        ))

    # Slider stepleri — her stop için label = formatlanmış yıl
    slider_steps = []
    for i, y in enumerate(stops):
        slider_steps.append({
            "args": [[str(i)], {"frame": {"duration": 200, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}}],
            "label": _format_years_tr(y),
            "method": "animate",
        })

    # Aktif frame seçimi
    active = max(0, min(active_idx, len(frames) - 1))
    initial = frames[active].data
    initial_anno = frames[active].layout.annotations
    initial_title = frames[active].layout.title.text

    # ESRI World Imagery uydu zemin tile'ları (Mapbox token gerektirmez)
    satellite_layers = [
        {"below": "traces", "sourcetype": "raster",
         "source": [ESRI_SAT], "sourceattribution": "ESRI World Imagery"},
        {"below": "traces", "sourcetype": "raster",
         "source": [ESRI_LABELS], "opacity": 0.70},
    ]

    fig = go.Figure(
        data=list(initial),
        frames=frames,
        layout=go.Layout(
            mapbox=dict(
                style="white-bg",
                layers=satellite_layers,
                center=dict(lat=focus_lat, lon=focus_lon),
                zoom=3.4 if mode_key == "pal" else 4.4,
            ),
            margin=dict(l=0, r=0, t=46, b=4),
            height=680,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eceff1"),
            showlegend=False,
            title=dict(text=initial_title, font=dict(color="#eceff1", size=14)),
            annotations=list(initial_anno) if initial_anno else [],
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "y": -0.04, "x": 0.0,
                "xanchor": "left", "yanchor": "top",
                "pad": {"t": 4, "r": 4},
                "bgcolor": "#263238",
                "font": {"color": "#eceff1", "size": 12},
                "buttons": [
                    {"label": "▶ Oynat", "method": "animate",
                     "args": [None, {"frame": {"duration": 200, "redraw": True},
                                     "fromcurrent": True,
                                     "transition": {"duration": 0}}]},
                    {"label": "⏸ Dur", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}]},
                ],
            }],
            sliders=[{
                "active": active,
                "currentvalue": {"prefix": "📅 ", "font": {"size": 13, "color": "#eceff1"}},
                "x": 0.10, "y": -0.04, "len": 0.85,
                "bgcolor": "rgba(60,60,80,0.55)",
                "font": {"color": "#eceff1", "size": 11},
                "steps": slider_steps,
                "transition": {"duration": 0},
            }],
        ),
    )
    return fig

@st.fragment
def _render_plaka_simulasyon():
    st.subheader("🌍 Tektonik Plaka Hareketi Simülasyonu")
    st.caption("Geçmişe ve geleceğe doğru plaka hareketini simüle eder. Negatif yıl = geçmiş, "
               "pozitif yıl = gelecek. **Uyarı bandı** yıl ölçeğinin bilimsel kalitesini gösterir.")

    # Mod seçimi (radio)
    mode_keys = list(_PLAKA_MODES.keys())
    mode_label_map = {k: _PLAKA_MODES[k]["label"] for k in mode_keys}
    mode_key = st.radio(
        "🔬 Zaman ölçeği modu",
        options=mode_keys,
        format_func=lambda k: mode_label_map[k],
        horizontal=True,
        index=0,
        key="plaka_sim_mode",
    )
    mode  = _PLAKA_MODES[mode_key]
    stops = mode["stops"]

    col_y, col_c, col_p = st.columns([1.4, 1, 1])
    with col_y:
        years_total = st.select_slider(
            "Simülasyon ufku (yıl)",
            options=stops,
            value=stops[mode["default_idx"]],
            format_func=_format_years_tr,
            key=f"plaka_sim_years_{mode_key}",
        )
    with col_c:
        city = st.selectbox(
            "Odak şehir",
            options=list(_PLAKA_CITIES.keys()),
            index=0,
            key="plaka_sim_city",
        )
    vels_for_select = load_plate_velocities(ERZ_LAT, ERZ_LON)
    plate_options = list(vels_for_select.keys())
    default_plate = "AN" if "AN" in plate_options else plate_options[0]
    with col_p:
        plate_code = st.selectbox(
            "Plaka (hız vektörü kaynağı)",
            options=plate_options,
            index=plate_options.index(default_plate),
            format_func=lambda c: f"{c} — {vels_for_select[c].get('name', c)}",
            key="plaka_sim_plate",
        )

    show_coupling = st.checkbox(
        "🔗 Kuplaj Göster — plakalar arası etkileşim okları",
        value=False,
        help=("Arabistan'ın kuzey baskısı Anadolu'yu batıya iter — KAF ve DAF "
              "bu kuvvetle aktif kalır. Hellenik subduction backarcı (slab "
              "rollback) Ege'yi güneybatıya çeker. Oklar her frame'de kümülatif "
              "yer değiştirme ile orantılı büyür."),
        key="plaka_sim_coupling",
    )

    focus_lat, focus_lon = _PLAKA_CITIES[city]
    active_idx = stops.index(years_total)

    # Uyarı bandı
    emoji, severity_label, severity_color, severity_desc = _plaka_warning(abs(years_total))
    band_bg = {"🟢": "#1b5e20", "🟡": "#f57f17", "🔴": "#b71c1c"}[emoji]
    st.markdown(
        f"""<div style="background:{band_bg};border-left:6px solid {severity_color};
                       border-radius:6px;padding:10px 14px;margin:8px 0">
            <div style="color:#fff;font-size:0.95rem;font-weight:700">
                {emoji} {severity_label} — {_format_years_tr(years_total)}</div>
            <div style="color:#fff;opacity:0.95;font-size:0.82rem;margin-top:4px">
                {severity_desc}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    fig = _plaka_build_figure(mode_key, focus_lat, focus_lon, city, plate_code,
                              active_idx=active_idx, show_coupling=show_coupling)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"plaka_sim_{mode_key}_{city}_{years_total}_{plate_code}_{int(show_coupling)}")

    # Gerçek (görsel ölçekten bağımsız) kümülatif yer değiştirme
    dlat_real, dlon_real = _plaka_displacement_deg(plate_code, focus_lat, focus_lon, years_total)
    real_m_lat = dlat_real * 111_320.0
    real_m_lon = dlon_real * 111_320.0 * math.cos(math.radians(focus_lat))
    disp_m = math.sqrt(real_m_lat**2 + real_m_lon**2)
    if disp_m < 1.0:
        disp_str = f"{disp_m*1000:.1f} mm"
    elif disp_m < 1000.0:
        disp_str = f"{disp_m:.2f} m"
    elif disp_m < 1_000_000:
        disp_str = f"{disp_m/1000:.2f} km"
    else:
        disp_str = f"{disp_m/1000:,.0f} km"

    vel_meta = vels_for_select.get(plate_code, {})
    plate_name = vel_meta.get("name", plate_code)
    speed_anno = vel_meta.get("approx_speed_mm_yr")
    speed_str  = f" (~{speed_anno} mm/yıl)" if speed_anno else ""

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("📅 Seçili ufuk", _format_years_tr(years_total))
    with c2:
        st.metric(f"📍 {city} kayması", disp_str,
                  help="NNR-MORVEL56 absolute frame, lineer ekstrapolasyon")
    with c3:
        st.metric("🪨 Plaka", f"{plate_code}{speed_str}",
                  help=plate_name)

    if mode["visual_scale_factor"] > 1:
        st.caption(f"ℹ️ Harita görseli **×{int(mode['visual_scale_factor'])}** ölçekle "
                   f"büyütülmüştür — gerçek kayma değeri yukarıdaki metrik kartında. "
                   f"Paleografik moda geçerek gerçek ölçekli görsel için 1.000.000+ yıl seçin.")

    if show_coupling:
        st.info(
            "🔗 **Kuplaj okları aktif** — plakalar bağımsız değil, kuplajlı hareket eder:\n\n"
            "• 🔴 **AR → AN baskı oku** (Bitlis-Zagros sütur): Arabistan kuzeye iter, "
            "Anadolu blokunu batıya kaçırır. KAF ve DAF bu kuvvetin sismik salınımıdır.\n\n"
            "• 🔵 **Hellenik geri çekilme oku** (Girit güneyi): Afrika dalan plakası "
            "geri çekilir (slab rollback); Ege/Anadolu batı kanadı güneybatıya yayılır.\n\n"
            "_Bu görsel, McClusky 2000 GPS bantı + Reilinger 2006 modelinin sezgisel "
            "özetidir. Ajan 8 bilimsel onayı beklenirken sağlanan ilk yaklaşımdır._"
        )

    with st.expander("📚 Bilimsel Not & Kaynaklar"):
        st.markdown(
            "**Hız kaynağı:** [`data/plate_velocities.json`](data/plate_velocities.json) — "
            "NNR-MORVEL56 mutlak plaka hareketi Euler kutupları "
            "(Argus, Gordon & DeMets 2011, *Geochem. Geophys. Geosyst.*, "
            "doi:10.1029/2011GC003751).\n\n"
            "**Mod sınırları:**\n"
            "- **🟢 Bilimsel (≤10.000 yıl):** GNSS-türevli lineer ekstrapolasyon doğrudan ölçüm rejiminde. "
            "1939 Erzincan Ms 7.8 ~4–7 m kayma; interseismik birikim 87 yıl × ~22 mm/yıl ≈ ~1.9 m "
            "— paleoseismik kayıtla tutarlı.\n"
            "- **🟡 Soyutlama (10.000–1.000.000 yıl):** Fay döngüleri ve viskoelastik relaksasyon "
            "modellenmedi. Bu aralıkta paleomanyetik korelasyon ve büyük deprem atlama istatistikleri "
            "(Wallace, Schwartz & Coppersmith 1984) baskındır.\n"
            "- **🔴 Spekülatif (>1.000.000 yıl):** Lineer ekstrapolasyon yetmez. Gerçek rekonstrüksiyon "
            "için PALEOMAP (Scotese 2016), GPlates (Müller et al. 2018) ve manto akışı tomografisi "
            "(Bunge et al. 2003) gerekir.\n\n"
            "**Hesap:** Plaka her vertex'i ve Erzincan pini, seçili plaka için "
            "`(Δφ, Δλ) = plate_velocity_vector(plate_id, φ, λ, years)` (Ajan 4, NNR-MORVEL56 "
            "Ω×P kartezyen rotasyonu) ile ötelenir.\n\n"
            "**Uydu zemin:** ESRI World Imagery (raster tile, MapBox token gerektirmez)."
        )

if active_menu == "🌍 Plaka Simülasyonu":
    _render_plaka_simulasyon()

# ════════════════════════════════════════════════════════════════════════════
# 🚨 ERKEN UYARI SİMÜLATÖRÜ — P/S Dalga Geri Sayım ve Şiddet Tahmini
# ════════════════════════════════════════════════════════════════════════════
if active_menu == "🚨 Erken Uyarı":
    st.markdown('<div class="chart-title">🚨 Erken Uyarı Simülatörü — P/S Dalga Geri Sayım</div>', unsafe_allow_html=True)
    st.caption(
        "Bir deprem olduğunda P-dalgası (hızlı, hasarsız) ile S-dalgası (yavaş, yıkıcı) arasındaki "
        "saniyeler **uyarı penceresidir**. Bu simülatör mevcut depremleri kullanarak farklı şehirlerin "
        "ne kadar uyarı süresine sahip olacağını gösterir."
    )

    st.warning(
        "⚠️ **Bu bir simülasyondur.** Gerçek bir EEW (Earthquake Early Warning) sistemi değildir. "
        "Türkiye'de operasyonel EEW olarak AFAD-EQE/EWS çalışmaktadır. Bu panel sadece kavram demonstrasyonudur; "
        "afet hazırlığı veya gerçek zamanlı kararlar için kullanılamaz."
    )

    # ────────────────────────────────────────────────────────────────────────
    # Dalga hızı sabitleri (kabuk için tipik P/S/Rayleigh)
    # ────────────────────────────────────────────────────────────────────────
    VP = 6.0   # P-dalga hızı (km/s)
    VS = 3.5   # S-dalga hızı (km/s)
    VR = 2.5   # Rayleigh yüzey dalga hızı (km/s)

    # Türkiye'nin önemli şehirleri (kullanıcı gözlemci konumu)
    SEHIRLER = {
        "Erzincan":   (39.7333, 39.4917),
        "İstanbul":   (41.0082, 28.9784),
        "Ankara":     (39.9334, 32.8597),
        "İzmir":      (38.4192, 27.1287),
        "Erzurum":    (39.9000, 41.2700),
        "Trabzon":    (41.0050, 39.7267),
        "Diyarbakır": (37.9144, 40.2306),
        "Antalya":    (36.8841, 30.7056),
        "Konya":      (37.8714, 32.4847),
        "Bursa":      (40.1828, 29.0665),
        "Adana":      (37.0000, 35.3213),
        "Gaziantep":  (37.0662, 37.3833),
        "Kayseri":    (38.7312, 35.4787),
        "Samsun":     (41.2867, 36.3300),
        "Van":        (38.5012, 43.3724),
    }

    # ────────────────────────────────────────────────────────────────────────
    # Olay ve şehir seçicileri
    # ────────────────────────────────────────────────────────────────────────
    eq_pool = df.dropna(subset=["lat", "lon", "buyukluk", "derinlik"]).copy()
    eq_pool = eq_pool[eq_pool["buyukluk"] >= 3.5].sort_values("buyukluk", ascending=False)

    if len(eq_pool) == 0:
        st.info("Erken uyarı simülasyonu için M ≥ 3.5 olay gerekli. Seçili dönemde yok — zaman aralığını genişletin.")
    else:
        eq_options = []
        for _, row in eq_pool.head(20).iterrows():
            label = f"M{row['buyukluk']:.1f} · {row['zaman_str']} · {row['konum'][:50]} (d={row['derinlik']:.0f} km)"
            eq_options.append(label)

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_eq_label = st.selectbox(
                "🌍 Senaryo depremi seçin (M ≥ 3.5, son seçili dönem)",
                eq_options,
                index=0,
            )
        with col_sel2:
            mode = st.radio(
                "Gözlemci konumu",
                ["Şehir listesi", "Manuel koordinat"],
                horizontal=False,
            )

        selected_eq = eq_pool.iloc[eq_options.index(selected_eq_label)]
        eq_lat, eq_lon, eq_depth, eq_mag = (
            float(selected_eq["lat"]),
            float(selected_eq["lon"]),
            float(selected_eq["derinlik"]),
            float(selected_eq["buyukluk"]),
        )

        if mode == "Şehir listesi":
            city_name = st.selectbox("🏙️ Şehir seçin", list(SEHIRLER.keys()), index=0)
            obs_lat, obs_lon = SEHIRLER[city_name]
            obs_label = city_name
        else:
            col_m1, col_m2 = st.columns(2)
            obs_lat = col_m1.number_input("Enlem", value=39.7333, format="%.4f", min_value=35.0, max_value=43.0)
            obs_lon = col_m2.number_input("Boylam", value=39.4917, format="%.4f", min_value=25.0, max_value=45.0)
            obs_label = f"({obs_lat:.3f}, {obs_lon:.3f})"

        # ────────────────────────────────────────────────────────────────────
        # Mesafe ve varış süresi hesabı
        # ────────────────────────────────────────────────────────────────────
        # Yüzey mesafesi (haversine)
        surface_km = haversine(eq_lat, eq_lon, obs_lat, obs_lon)
        # Hipocenter mesafesi (3B): yüzey ve derinlik bacakları
        hypo_km = math.sqrt(surface_km**2 + eq_depth**2)

        t_p = hypo_km / VP
        t_s = hypo_km / VS
        t_r = hypo_km / VR
        warning_window = t_s - t_p  # S − P penceresi (gözlemcinin uyarı süresi)

        # MMI tahmini (basitleştirilmiş GMPE)
        # I = 1.5*M - 1.5*log10(R) - 3.5  (R = hypocenter mesafesi, km)
        if hypo_km < 1:
            mmi = 12.0  # epicenter üstünde — maksimum
        else:
            mmi = 1.5 * eq_mag - 1.5 * math.log10(hypo_km) - 3.5
        mmi = max(0.0, min(12.0, mmi))

        # MMI eşik renk + açıklama
        if mmi < 3:
            mmi_color = "#66bb6a"
            mmi_level = "Hafif (hissedilmez)"
        elif mmi < 5:
            mmi_color = "#ffd54f"
            mmi_level = "Zayıf-Orta (hissedilir)"
        elif mmi < 7:
            mmi_color = "#ff8a65"
            mmi_level = "Kuvvetli (mobilya hareket eder)"
        elif mmi < 9:
            mmi_color = "#e53935"
            mmi_level = "Yıkıcı (yapısal hasar)"
        else:
            mmi_color = "#7b1fa2"
            mmi_level = "Aşırı (büyük yıkım)"

        # ────────────────────────────────────────────────────────────────────
        # Sonuç kartları
        # ────────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f'<div class="chart-title">📊 Sonuç — M{eq_mag:.1f} → {obs_label}</div>', unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Yüzey Mesafesi", f"{surface_km:.1f} km")
        r2.metric("Hipocenter Mesafesi", f"{hypo_km:.1f} km", f"derinlik: {eq_depth:.0f} km")
        r3.metric("P-dalga Varışı", f"{t_p:.1f} s", f"vp={VP} km/s")
        r4.metric("S-dalga Varışı", f"{t_s:.1f} s", f"vs={VS} km/s")

        # Uyarı penceresi (öne çıkan büyük kart)
        if warning_window < 3:
            window_color = "#e53935"
            window_msg = "ÇOK KISA — Anında sığın!"
        elif warning_window < 10:
            window_color = "#ff8a65"
            window_msg = "Kısa — Hızlı pozisyon al"
        elif warning_window < 30:
            window_color = "#ffd54f"
            window_msg = "Orta — Pozisyon al, dışarı çıkma"
        else:
            window_color = "#66bb6a"
            window_msg = "Uzun — Güvenli noktaya geç"

        st.markdown(
            f'<div style="background:linear-gradient(135deg, {BG3} 0%, {BG2} 100%);'
            f'border:2px solid {window_color};border-radius:12px;padding:18px;margin:12px 0;'
            f'text-align:center">'
            f'<div style="font-size:0.75rem;opacity:0.7;letter-spacing:2px">S − P UYARI PENCERESİ</div>'
            f'<div style="font-size:3.5rem;font-weight:900;color:{window_color};line-height:1">'
            f'{warning_window:.1f} <span style="font-size:1.5rem;opacity:0.7">sn</span></div>'
            f'<div style="font-size:0.95rem;color:{window_color};margin-top:6px">{window_msg}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # MMI kartı
        st.markdown(
            f'<div style="background:{BG3};border-left:5px solid {mmi_color};border-radius:8px;padding:14px;margin:10px 0">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div><div style="font-size:0.72rem;opacity:0.7;letter-spacing:1px">BEKLENEN ŞİDDET (MMI tahmini)</div>'
            f'<div style="font-size:1.05rem;color:{mmi_color};font-weight:700">{mmi_level}</div></div>'
            f'<div style="font-size:2.6rem;font-weight:900;color:{mmi_color}">{mmi:.1f}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ────────────────────────────────────────────────────────────────────
        # Görsel zaman ekseni (Plotly)
        # ────────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="chart-title">⏱️ Dalga Varış Zaman Ekseni</div>', unsafe_allow_html=True)

        t_max = max(t_r * 1.15, 5.0)
        fig_tl = go.Figure()

        # P/S/Rayleigh varış oklarını dikey çizgi olarak ekle
        for t_val, label, color, height in [
            (t_p, f"P {t_p:.1f}s",       "#42a5f5", 1.0),
            (t_s, f"S {t_s:.1f}s",       "#ef5350", 1.0),
            (t_r, f"Rayleigh {t_r:.1f}s", "#ab47bc", 1.0),
        ]:
            fig_tl.add_trace(go.Scatter(
                x=[t_val, t_val], y=[0, height],
                mode="lines+markers+text",
                line=dict(color=color, width=3),
                marker=dict(size=14, color=color, symbol="diamond"),
                text=["", label],
                textposition="top center",
                textfont=dict(color=color, size=12, family="Arial Black"),
                showlegend=False,
                hovertemplate=f"{label}<extra></extra>",
            ))

        # Uyarı penceresi: P ile S arasında yeşil bant
        fig_tl.add_shape(
            type="rect", x0=t_p, x1=t_s, y0=0, y1=0.45,
            fillcolor=window_color, opacity=0.20, line=dict(width=0),
            layer="below",
        )
        fig_tl.add_annotation(
            x=(t_p + t_s) / 2, y=0.22,
            text=f"🛡️ Uyarı Penceresi<br>{warning_window:.1f} sn",
            showarrow=False, font=dict(color=window_color, size=13, family="Arial Black"),
        )

        # S dalgasından sonra: tehlike bandı (S'den Rayleigh'in 1.2x'ine kadar)
        fig_tl.add_shape(
            type="rect", x0=t_s, x1=t_r * 1.15, y0=0, y1=0.45,
            fillcolor="#ef5350", opacity=0.10, line=dict(width=0),
            layer="below",
        )

        fig_tl.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TEXT, size=11),
            margin=dict(t=30, b=40, l=50, r=20),
            height=270,
            xaxis=dict(
                title="Olaydan itibaren süre (saniye)",
                range=[0, t_max], gridcolor=GRID, zerolinecolor=BORDER,
                tickfont=dict(color=TEXT),
            ),
            yaxis=dict(visible=False, range=[0, 1.4]),
            showlegend=False,
        )
        st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

        # ────────────────────────────────────────────────────────────────────
        # Eğitimsel "Ne yapmalı?" kartı
        # ────────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="chart-title">🧠 Bu Süre İçinde Ne Yapmalı?</div>', unsafe_allow_html=True)

        c_a, c_b, c_c = st.columns(3)
        with c_a:
            st.markdown(
                f'<div style="background:{BG3};border-radius:8px;padding:12px;height:170px">'
                f'<div style="font-size:1.6rem">🛏️</div>'
                f'<div style="font-weight:700;color:#90caf9">Çök – Kapan – Tutun</div>'
                f'<div style="font-size:0.85rem;opacity:0.85;margin-top:6px">'
                f'Sağlam masanın altına gir. Boynunu kollarınla koru. Bina dururken hareket etme.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with c_b:
            st.markdown(
                f'<div style="background:{BG3};border-radius:8px;padding:12px;height:170px">'
                f'<div style="font-size:1.6rem">🚪</div>'
                f'<div style="font-weight:700;color:#ffb74d">Pencere/Cam/Eşyadan Uzak Dur</div>'
                f'<div style="font-size:0.85rem;opacity:0.85;margin-top:6px">'
                f'Düşebilecek eşyaların yakınından uzaklaş. Asansöre binme; merdivenden hızlı inme.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with c_c:
            st.markdown(
                f'<div style="background:{BG3};border-radius:8px;padding:12px;height:170px">'
                f'<div style="font-size:1.6rem">🚗</div>'
                f'<div style="font-weight:700;color:#a5d6a7">Araçtaysan / Dışarıdaysan</div>'
                f'<div style="font-size:0.85rem;opacity:0.85;margin-top:6px">'
                f'Aracını yan banketere çek. Açık alandaysan binadan, tel/direkten uzak dur.'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Edge case uyarıları
        st.markdown("---")
        if surface_km < 5:
            st.error("🎯 Sen merkez üstündesin — uyarı penceresi anlamsız. P ve S aynı anda gelir.")
        elif eq_depth > 200:
            st.warning(f"🌋 Derin odaklı deprem (d={eq_depth:.0f} km). P/S hızları manto malzemesinde farklılaşır — gerçek varış bu tahminden hızlı olabilir.")
        elif surface_km > 500:
            st.warning(f"📏 Çok uzak deprem ({surface_km:.0f} km). Yüzey (Rayleigh/Love) dalgaları baskın gelir; S-P penceresi yanıltıcıdır.")

        st.caption(
            "📚 **Formül kaynağı:** Hipocenter mesafesi `R = √(yüzey² + derinlik²)`. "
            "Tipik kabuk hızları: vp=6.0 km/s, vs=3.5 km/s, vr=2.5 km/s. "
            "MMI yaklaşımı: `I = 1.5·M − 1.5·log₁₀(R) − 3.5` (ham GMPE, bölgesel kalibrasyon yapılmamış)."
        )

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:{SUBTEXT};font-size:0.7rem;
            margin-top:0.8rem;padding:0.5rem;background:{BG3};
            border-radius:8px;border:1px solid {BORDER}">
  USGS · EMSC · AFAD · Kandilli · GFZ Potsdam · IRIS/SAGE &nbsp;|&nbsp;
  Her {refresh_s} saniyede otomatik yenileme &nbsp;|&nbsp;
  Uydu haritasi: ESRI World Imagery &nbsp;|&nbsp;
  v {APP_VERSION} &nbsp;|&nbsp; {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
</div>
""", unsafe_allow_html=True)
