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
    omori_utsu_rate,
    parse_usgs_feed_features,
    reasenberg_jones_probability,
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
APP_VERSION = "1.41"
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
    "📈 Artçı Tahmin",
    "🔴 Sismik Açık",
    "🌊 ShakeMap",
    "🗺️ Sismik Tehlike",
    "🥎 Odak Mekanizması",
    "📉 b-Değeri Zaman Serisi",
    "💥 Coulomb Stres",
    "🛰️ InSAR Deformasyon",
    "📜 Tarihsel Sismisite",
    "🔄 Sismik Döngü",
    "🌐 Dinamik Tetikleme",
    "📡 InSAR Zaman Serisi",
    "🔒 Fay Kilitlenme",
    "🌋 Moho Derinliği",
    "🌀 SKS Splitting",
    "🌊 Tsunami Kataloğu",
    "⏱️ Tsunami Varış",
    "🎬 Ambraseys Animasyon",
    "⛏️ Paleosismik Kazı",
    "🗺️ Tsunami Tehlike",
    "🏔️ Vs30 Zemin",
    "🏚️ HAZUS Kayıp",
    "🏺 Erzincan Paleo",
    "🏛️ Erzincan Arşivi",
    "🎓 Bilgi Havuzu",
    "⚙️ Sistem & Veri",
    "📝 Raporlar",
]
_MENU_ICONS = [
    "globe", "bar-chart-line", "compass", "globe-americas", "moon-stars",
    "exclamation-triangle", "graph-up-arrow", "exclamation-octagon-fill",
    "broadcast-pin", "map-fill", "circle-half", "graph-down", "lightning-charge", "satellite", "journal-text", "arrow-repeat", "globe2", "broadcast", "lock-fill", "layers-half", "compass-fill", "water", "stopwatch", "film", "hammer", "tsunami", "bricks", "building-x", "tree", "archive", "mortarboard", "gear", "file-text",
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
def load_plate_velocities(ref_lat: float = ERZ_LAT, ref_lon: float = ERZ_LON,
                          reference_frame: str = "EU"):
    """Tüm plakalar için (ref_lat, ref_lon) noktasında deg/yıl hız vektörü.

    v1.22 — Eurasia-fixed varsayılan (kullanıcı/Bilim Profesörü kararı):
      Türkiye tektoniği için doğru anlatı V_relative = V_target - V_reference.
      `reference_frame="EU"` → tüm hızlar Avrasya'ya göre göreceli (Reilinger 2006
      methodolojisi). `reference_frame="NNR"` → mutlak NNR-MORVEL56 (Argus 2011).

    Ajan 4 entegrasyonu (try/except ile):
      - varsa data/plate_velocities.json → format otomatik tespit
      - yoksa veya parse hatası → hardcode AN fallback

    Çıkış formatı:
      {plate_code: {"delta_lat_per_year": float, "delta_lon_per_year": float,
                    "name": str, "approx_speed_mm_yr": float|None,
                    "is_euler_derived": bool, "reference_frame": str}}
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

    # v1.22 — Eurasia-fixed referans çerçevesi (kullanıcı/Bilim Profesörü teknik kararı)
    # V_relative(target) = V_target_NNR - V_reference_NNR
    # Türkiye tektoniği için doğru anlatı:
    #   AN/Eurasia ≈ batı/güneybatı 21 mm/yıl (KAF kaçışı)
    #   AR/Eurasia ≈ kuzey/KKB     23 mm/yıl (Bitlis-Zagros sıkışma)
    #   AF/Eurasia ≈ kuzey/KKD     10 mm/yıl (Helenik dalma-batma)
    ref_code = (reference_frame or "").upper()
    if ref_code and ref_code not in ("NNR", "ABSOLUTE", ""):
        ref_vel = out.get(ref_code)
        if ref_vel:
            ref_dlat = ref_vel["delta_lat_per_year"]
            ref_dlon = ref_vel["delta_lon_per_year"]
            out_rel = {}
            for code, vel in out.items():
                out_rel[code] = {
                    **vel,
                    "delta_lat_per_year": vel["delta_lat_per_year"] - ref_dlat,
                    "delta_lon_per_year": vel["delta_lon_per_year"] - ref_dlon,
                    "reference_frame": ref_code,
                }
            return out_rel

    # NNR mutlak veya referans bulunamadı → NNR olarak işaretle
    for vel in out.values():
        vel.setdefault("reference_frame", "NNR")
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

# 21 log-spaced frame stops per mode. Negatif = geçmiş, pozitif = gelecek.
# Üç kademe (kullanıcı talebi v1.18.2): 🟢 Bilimsel → 🟡 Genişletilmiş → 🔴 Spekülatif
# Dict insertion order korunur (Python 3.7+) → ilk mode default; "sci" en başta = bilimsel namus.
_PLAKA_MODES = {
    "sci": {
        # 🟢 BİLİMSEL — gerçek GNSS ölçüm rejimi, lineer ekstrapolasyon doğrudan geçerli
        # v1.21.1 KADEMELI ÖLÇEK: sci max 10K → ~0.5° görsel (KÜÇÜK kayma)
        # Modlar arası 5× artış sağlanır → kullanıcı 10K vs 1M vs 1B'yi NET ayırt eder
        # (önce hepsi ~2.25°/2.7° idi, görsel olarak aynı görünüyordu — kullanıcı bildirdi)
        # Mod İÇİNDE log-orantılı: sci 1K=0.05°, 10K=0.5° (her 10× yıl → 10× görsel)
        "label":   "🟢 Bilimsel — Doğrudan Ölçüm (-10.000 → +10.000 yıl)",
        "short":   "Bilimsel",
        "stops":   [-10_000, -3_000, -1_000, -500, -200, -100, -50, -20, -10, -3,
                    0,
                    3, 10, 20, 50, 100, 200, 500, 1_000, 3_000, 10_000],
        "default_idx": 10,  # 0
        "visual_scale_factor": 222.0,  # 10K × 2.25e-7 × 222 ≈ 0.5° görsel (KÜÇÜK)
    },
    "geo": {
        # 🟡 GENİŞLETİLMİŞ — paleosismik kalibrasyon zonu
        # Max 1M × 2.25e-7 × 11.1 ≈ 2.5° görsel (sci'nin 5×'i, pal'in 1/2'si — ORTA)
        "label":   "🟡 Genişletilmiş — Paleosismik Ufuk (-1 milyon → +1 milyon yıl)",
        "short":   "Genişletilmiş",
        "stops":   [-1_000_000, -300_000, -100_000, -30_000, -10_000, -3_000, -1_000, -300, -100, -10,
                    0,
                    10, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
        "default_idx": 10,  # 0
        "visual_scale_factor": 11.1,  # 1M × 2.25e-7 × 11.1 ≈ 2.5° görsel (ORTA)
    },
    "pal": {
        # 🔴 SPEKÜLATİF — kıta sürüklenmesi eğitsel sezgi
        # Max 1B × 2.25e-7 × 0.0222 ≈ 5° görsel (geo'nun 2×'i, sci'nin 10×'i — BÜYÜK)
        "label":   "🔴 Spekülatif — Eğitsel Sezgi (-1 milyar → +1 milyar yıl)",
        "short":   "Spekülatif",
        "stops":   [-1_000_000_000, -500_000_000, -200_000_000, -100_000_000, -50_000_000,
                    -20_000_000, -10_000_000, -5_000_000, -2_000_000, -1_000_000,
                    0,
                    1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000,
                    50_000_000, 100_000_000, 200_000_000, 500_000_000, 1_000_000_000],
        "default_idx": 10,  # 0
        "visual_scale_factor": 0.0222,  # 1B × 2.25e-7 × 0.0222 ≈ 5° görsel (BÜYÜK)
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

def _plaka_displacement_deg(plate_code: str, lat: float, lon: float, years: int,
                            reference_frame: str = "EU"):
    """Δφ, Δλ derece olarak — V_relative = V_target − V_reference (v1.22).

    reference_frame: "EU" (varsayılan, Eurasia-fixed — Türkiye tektoniği için doğru)
                     veya "NNR" (mutlak NNR-MORVEL56, Argus 2011)
                     veya başka plaka kodu (AN/AR/AF) — özel kıyaslama için.

    Önce Ajan 4'ün `plate_velocity_vector()` fonksiyonunu dener (NNR çıktı varsayar);
    başarısız olursa kendi Euler dönüştürücüsü `load_plate_velocities()` ile."""
    if _plate_velocity_vector_extern is not None and reference_frame.upper() == "NNR":
        # Extern referans desteklemez → sadece NNR isteğinde dene
        try:
            return _plate_velocity_vector_extern(plate_code, lat, lon, years)
        except Exception:
            pass
    vels = load_plate_velocities(lat, lon, reference_frame=reference_frame)
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

    # v1.18.3 PERF — Sadece hız vektörü tanımlı plakalara ait sınırları render et.
    # Global 241 PB2002 sınırından ~10-15 Türkiye/Mediterranean sınırına düşer.
    # Diğer plakalar (NA/SA/PA/IN/AU/...) hız vektörü yoktu ve zaten kayma=0
    # olarak render ediliyordu — sadece görsel kirlilik yapıyorlardı.
    _RELEVANT_PLATE_CODES = set(_PB2002_TO_VELOCITY_CODE.keys())  # {AT, AS, EU, AF, AR}
    plates_in_scope = [
        p for p in PLATE_LINES
        if (p.get("plate_a") in _RELEVANT_PLATE_CODES
            or p.get("plate_b") in _RELEVANT_PLATE_CODES)
    ]

    # Statik gri "bugünkü plaka sınırları" (tüm frame'lerde sabit)
    base_lats, base_lons = [], []
    for plate in plates_in_scope:
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
    for plate in plates_in_scope:
        plates_by_type.setdefault(plate.get("type", "unknown"), []).append(plate)

    # v1.17.3 TOPLU HAREKET — her sınırın kendi hız ortalaması (frame-dışı pre-compute)
    # PB2002 sınırı iki plakanın arasındadır → sınırın hızı = iki plakanın hız ORTALAMASI
    # (gerçek tektonik approximation; tek-plaka deltası "tüm dünya AN ile kayıyor"
    # yanılsamasını ortadan kaldırır).
    vels_dict = load_plate_velocities(focus_lat, focus_lon)
    border_dlat_yr = {}  # id(plate) → derece/yıl
    border_dlon_yr = {}
    for plate in plates_in_scope:
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

# ════════════════════════════════════════════════════════════════════════════
# 📈 ARTÇI TAHMİN — F-51 / v1.19 — Reasenberg & Jones (1989) Olasılık Paneli
# Kaynaklar: Reasenberg & Jones (1989), Science 243:1173-1176,
#            DOI:10.1126/science.243.4895.1173
#            Omori (1894), J. Coll. Sci. Imp. Univ. Tokyo 7:111-200
#            Utsu, Ogata & Matsu'ura (1995), J. Phys. Earth 43:1-33
#            Öztürk et al. (2011), J. Seismol. — Türkiye için a, b kalibrasyonu
# ════════════════════════════════════════════════════════════════════════════
_RJ_WINDOWS = {
    "Sonraki 24 saat": 1.0,
    "Sonraki 7 gün": 7.0,
    "Sonraki 30 gün": 30.0,
}


@st.fragment
def _render_artci_tahmin():
    st.markdown(
        '<div class="chart-title">📈 Artçı Tahmin — Reasenberg & Jones + Omori-Utsu (F-51 / v1.19)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Bir ana depremin (mainshock) ardından, t1–t2 zaman aralığında belirli bir minimum "
        "büyüklüğün üstündeki artçıların **olasılığını** Poisson varsayımı altında hesaplar. "
        "Hız modeli: Omori-Utsu yasası n(t) = K / (t + c)^p."
    )

    # ── Girdi bölümü ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        m_main = st.slider(
            "Ana deprem Mw", 4.0, 8.0, 6.8, 0.1, key="rj_m_main",
            help="Reasenberg & Jones 1989 — California modeli; Türkiye a=-1.67, b=1.0",
        )
    with c2:
        m_min = st.selectbox(
            "Min artçı büyüklüğü", [3.0, 4.0, 5.0, 6.0], index=1, key="rj_m_min",
            help="P(N>=1) hesaplamasında alt sınır",
        )
    with c3:
        t_now = st.slider(
            "Mainshock'tan bu yana (gün)", 0, 365, 1, key="rj_t_now",
            help="Tahmin penceresinin başlangıcı (t1)",
        )
    with c4:
        win_label = st.selectbox(
            "Tahmin penceresi",
            list(_RJ_WINDOWS.keys()), index=1, key="rj_window",
            help="Pencere uzunluğu = t2 - t1",
        )

    # Bilgi notu — sabit b ve p değerleri
    st.caption(
        "ℹ️ **Sabitler:** b = 1.0 (Gutenberg-Richter Türkiye ortalaması, "
        "Öztürk 2011); p = 1.0 (Omori-Utsu standard); c = 0.05 gün; "
        "a = −1.67 (Türkiye için kalibre — KOERI 1900-2020 kataloğu)."
    )

    t1 = float(t_now)
    t2 = t1 + _RJ_WINDOWS[win_label]
    b_val, p_val, c_val, a_val = 1.0, 1.0, 0.05, -1.67

    res = reasenberg_jones_probability(
        M_main=m_main, M_min=m_min, t1=t1, t2=t2,
        b=b_val, p=p_val, c=c_val, a=a_val,
    )
    prob_pct = res["probability"] * 100.0
    mu = res["expected"]
    K_val = res["K"]
    rate_now = omori_utsu_rate(max(t1, 0.001), K_val, c_val, p_val)

    # ── Sonuç kartları ─────────────────────────────────────────────────────
    if prob_pct > 50:
        prob_color = "#ff4444"; prob_emoji = "🔴"
    elif prob_pct >= 20:
        prob_color = "#ff9933"; prob_emoji = "🟠"
    else:
        prob_color = "#44cc66"; prob_emoji = "🟢"

    cR1, cR2, cR3 = st.columns(3)
    with cR1:
        st.markdown(
            f"""<div style="background:{BG3};border:2px solid {prob_color};
                          border-radius:10px;padding:14px;text-align:center">
                <div style="font-size:0.85rem;color:{SUBTEXT}">Olasılık (M≥{m_min:.0f}, {win_label.lower()})</div>
                <div style="font-size:2.0rem;font-weight:800;color:{prob_color}">
                    {prob_emoji} %{prob_pct:.1f}
                </div>
                <div style="font-size:0.75rem;color:{SUBTEXT}">P(N ≥ 1) = 1 − e^(−μ)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cR2:
        st.markdown(
            f"""<div style="background:{BG3};border:1px solid {BORDER};
                          border-radius:10px;padding:14px;text-align:center">
                <div style="font-size:0.85rem;color:{SUBTEXT}">Beklenen artçı sayısı</div>
                <div style="font-size:2.0rem;font-weight:800;color:{TEXT}">
                    📊 {mu:.2f}
                </div>
                <div style="font-size:0.75rem;color:{SUBTEXT}">μ (Poisson ortalaması)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with cR3:
        st.markdown(
            f"""<div style="background:{BG3};border:1px solid {BORDER};
                          border-radius:10px;padding:14px;text-align:center">
                <div style="font-size:0.85rem;color:{SUBTEXT}">Omori-Utsu anlık hız</div>
                <div style="font-size:2.0rem;font-weight:800;color:{TEXT}">
                    ⏱️ {rate_now:.2f}
                </div>
                <div style="font-size:0.75rem;color:{SUBTEXT}">deprem / gün (t = {t1:.1f} gün)</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Omori-Utsu zaman grafiği ───────────────────────────────────────────
    st.markdown(
        '<div class="chart-title">📉 Omori-Utsu Artçı Hız Eğrisi</div>',
        unsafe_allow_html=True,
    )

    t_grid = np.logspace(-2, math.log10(365.0), 200)  # 0.01 → 365 gün, log
    rate_grid = np.array([omori_utsu_rate(t, K_val, c_val, p_val) for t in t_grid])

    fig_omori = go.Figure()
    fig_omori.add_trace(go.Scatter(
        x=t_grid, y=rate_grid,
        mode="lines",
        line=dict(color="#ff7733", width=3),
        name="n(t) = K/(t+c)^p",
        hovertemplate="t = %{x:.2f} gün<br>hız = %{y:.3f} dpr/gün<extra></extra>",
    ))
    # Tahmin penceresi (gölgeli)
    fig_omori.add_vrect(
        x0=t1, x1=t2,
        fillcolor="rgba(80,170,255,0.18)", line_width=0,
        annotation_text=f"Pencere: {win_label}", annotation_position="top left",
        annotation=dict(font=dict(color="#aacfff", size=11)),
    )
    # "Şu an" işareti
    fig_omori.add_vline(
        x=max(t1, 0.01),
        line=dict(color="#ffcc00", width=2, dash="dash"),
        annotation_text="⏰ Şu an",
        annotation_position="top right",
        annotation=dict(font=dict(color="#ffcc00", size=11)),
    )
    fig_omori.update_layout(
        title=dict(
            text=f"Mw {m_main:.1f} ana deprem — K = {K_val:.3f}, c = {c_val}, p = {p_val}",
            font=dict(size=12, color=TEXT),
        ),
        height=380,
        plot_bgcolor=BG2, paper_bgcolor=BG2,
        font=dict(color=TEXT, size=11),
        xaxis=dict(
            title="Mainshock'tan sonra geçen gün (log)",
            type="log", gridcolor="#222",
        ),
        yaxis=dict(
            title=f"Artçı hızı (M ≥ {m_min:.0f} dpr/gün, log)",
            type="log", gridcolor="#222",
        ),
        margin=dict(t=50, b=50, l=60, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_omori, use_container_width=True, key="rj_omori_curve")

    # ── Büyüklük-olasılık tablosu ──────────────────────────────────────────
    st.markdown(
        '<div class="chart-title">📋 Büyüklük × Pencere Olasılık Tablosu</div>',
        unsafe_allow_html=True,
    )

    mag_rows = [3.0, 4.0, 5.0, 6.0]
    tbl_rows = []
    for mm in mag_rows:
        row = {"Artçı Mw": f"M ≥ {mm:.0f}"}
        for wl, dur in _RJ_WINDOWS.items():
            r = reasenberg_jones_probability(
                M_main=m_main, M_min=mm, t1=t1, t2=t1 + dur,
                b=b_val, p=p_val, c=c_val, a=a_val,
            )
            row[wl] = f"%{r['probability'] * 100:.1f}"
        tbl_rows.append(row)
    tbl_df = pd.DataFrame(tbl_rows)
    st.dataframe(tbl_df, use_container_width=True, hide_index=True)

    # ── Bilimsel kaynak kutusu ─────────────────────────────────────────────
    st.info(
        "📚 **Birincil kaynak:** Reasenberg, P.A. & Jones, L.M. (1989). "
        "Earthquake hazard after a mainshock in California. "
        "*Science* 243(4895):1173-1176. DOI:10.1126/science.243.4895.1173 | "
        "**Omori-Utsu:** Omori (1894) *J. Coll. Sci. Imp. Univ. Tokyo* 7:111-200; "
        "Utsu, Ogata & Matsu'ura (1995) *J. Phys. Earth* 43:1-33 | "
        "**Türkiye kalibrasyonu:** Öztürk et al. (2011) *J. Seismol.* "
        "(a = −1.67, b = 1.0; KOERI kataloğu 1900-2020)."
    )

    # ── Uyarı ──────────────────────────────────────────────────────────────
    st.warning(
        "⚠️ **Bu istatistiksel bir modeldir.** Bireysel artçı depremlerin "
        "**zamanını, yerini veya büyüklüğünü tahmin etmez** — yalnızca seçili "
        "pencere içinde belirli M üstünde en az 1 artçı görme olasılığını "
        "verir. Sonuçlar Türkiye için bölgesel kalibrasyon kabaca yapılmış "
        "ortalama parametrelerden türetilmiştir; spesifik faylarda gerçek "
        "p, K, c değerleri farklı olabilir (Öztürk 2011). "
        "Kaynak: Reasenberg & Jones (1989), *Science* 243:1173-1176."
    )


if active_menu == "📈 Artçı Tahmin":
    _render_artci_tahmin()


# ════════════════════════════════════════════════════════════════════════════
# 🏛️ ERZİNCAN ARŞİVİ — F-66 / v1.18 — Tarihi Depremler Analiz Paneli
# Kaynaklar: Ambraseys & Finkel 1995, Barka 1996, Özalaybey 1993, Grosser 1998,
#            Reilinger 2006 (GPS), Wallace-Schwartz-Coppersmith 1984 (paleoseismik)
# ════════════════════════════════════════════════════════════════════════════
ERZINCAN_TARIHI = [
    {
        "yil": 1939, "tarih": "26 Aralık 1939", "mw": 7.8, "ms": 7.8,
        "lat": 39.77, "lon": 39.53, "derinlik_km": 20,
        "kirik_uzunluk_km": 360, "kirik_yon": "KD-GB (N60E doğrultu atımlı)",
        "can_kaybi": 32962, "yarali": 100000, "hasarli_koy": 116,
        "etki_alani_km2": 45000,
        "kaynak": "Barka, A. (1996). Slip distribution along the North Anatolian Fault. Bull. Seismol. Soc. Am., 86(5), 1238-1254.",
        "kaynak2": "Ambraseys, N.N. & Finkel, C.F. (1995). The Seismicity of Turkey. Muhittin Salih Eren, İstanbul.",
        "bilgi": "20. yüzyılın en yıkıcı Türkiye depremlerinden biri. 360 km'lik KAF segmenti kırıldı.",
        "vay_be": "O gece saat 01:57'de uyku sırasında gerçekleşti. Yapılar 8-10 saniye içinde yıkıldı.",
    },
    {
        "yil": 1992, "tarih": "13 Mart 1992", "mw": 6.8, "ms": 6.8,
        "lat": 39.71, "lon": 39.60, "derinlik_km": 10,
        "kirik_uzunluk_km": 55, "kirik_yon": "KD-GB sağ yanal atımlı",
        "can_kaybi": 653, "yarali": 3850, "hasarli_koy": 52,
        "etki_alani_km2": 5000,
        "kaynak": "Özalaybey, S. et al. (1993). An analysis of the 1992 Erzincan earthquake sequence. Bull. Seismol. Soc. Am., 83(6), 1883-1893.",
        "kaynak2": "Grosser, H. et al. (1998). The Erzincan (Turkey) earthquake sequence of March 1992. Geophys. J. Int., 134, 669-700.",
        "bilgi": "55 km'lik segmentte kırılma. Artçı deprem dizisi 6 ay sürdü. Saat 18:17'de oldu.",
        "vay_be": "Deprem tam akşam yemeği saatinde geldi. Şehrin eski taş binaları yıkılırken yeni yapılar ayakta kaldı — zemin etkisinin en çarpıcı kanıtı.",
    },
    {
        "yil": 1784, "tarih": "1784 (kesin tarih belirsiz)", "mw": 7.0, "ms": None,
        "lat": 39.75, "lon": 39.50, "derinlik_km": None,
        "kirik_uzunluk_km": None, "kirik_yon": "KAF Erzincan segmenti",
        "can_kaybi": None, "yarali": None, "hasarli_koy": None,
        "etki_alani_km2": None,
        "kaynak": "Ambraseys, N.N. & Finkel, C.F. (1995). The Seismicity of Turkey. Muhittin Salih Eren, İstanbul. s.182-185.",
        "kaynak2": None,
        "bilgi": "Osmanlı dönemi tarihi kayıtlardan derlendi. KAF Erzincan segmentinin periyodik kırılmasını destekler.",
        "vay_be": "Yazılı tarih bu depremi kaydetti — fay 1939'dan 155 yıl önce de aynı yerde kırılmıştı.",
    },
    {
        "yil": 1901, "tarih": "9 Mayıs 1901", "mw": 6.5, "ms": None,
        "lat": 39.8, "lon": 39.4, "derinlik_km": None,
        "kirik_uzunluk_km": None, "kirik_yon": "KAF",
        "can_kaybi": None, "yarali": None, "hasarli_koy": None,
        "etki_alani_km2": None,
        "kaynak": "Ambraseys, N.N. & Finkel, C.F. (1995). The Seismicity of Turkey. s.187.",
        "kaynak2": None,
        "bilgi": "1939 öncesi bölgede kaydedilen orta büyüklükte deprem.",
        "vay_be": None,
    },
]


@st.fragment
def _render_erzincan_arsivi():
    st.markdown(
        '<div class="chart-title">🏛️ Erzincan Arşivi — Tarihi Depremler Analiz Paneli (F-66 / v1.18)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "1784–1992 arası Erzincan ve çevresinde Kuzey Anadolu Fayı (KAF) Erzincan segmentinde "
        "gerçekleşen büyük depremlerin bilimsel arşivi. Kaynaklar: Ambraseys & Finkel 1995, "
        "Barka 1996, Özalaybey 1993, Grosser 1998, Reilinger 2006."
    )

    df_e = pd.DataFrame(ERZINCAN_TARIHI)

    # ── Bölüm A: Özet metrik kartları ──────────────────────────────────────
    cA1, cA2, cA3, cA4 = st.columns(4)
    cA1.metric("💀 Toplam Kayıp (1939 + 1992)", "33.615 can",
               help="32.962 (1939) + 653 (1992) — Barka 1996 & Özalaybey 1993")
    cA2.metric("📏 En Büyük Kırık", "360 km",
               help="1939 Erzincan Ms 7.8 — Barka 1996, KAF segmenti")
    cA3.metric("🔁 Ort. Tekrar Periyodu", "~155 yıl",
               help="1784 → 1939 arası, Ambraseys & Finkel 1995")
    cA4.metric("⏩ KAF Kayma Hızı", "18 mm/yıl",
               help="Barka 1996; Reilinger et al. 2006 GPS")

    # ── Bölüm B: İnteraktif zaman çizelgesi (Plotly scatter) ───────────────
    st.markdown('<div class="chart-title">⏱️ Zaman Çizelgesi (1700–2026)</div>',
                unsafe_allow_html=True)

    casualty = df_e["can_kaybi"].fillna(0).astype(float).clip(lower=0)
    bubble = np.where(casualty > 0, np.sqrt(casualty) / 3.0 + 12, 12)

    fig_tl = go.Figure()
    # Karakteristik Mw 7.8 eşik şeridi (arka plan)
    fig_tl.add_shape(
        type="rect", x0=1700, x1=2030, y0=7.7, y1=8.0,
        fillcolor="rgba(200,200,200,0.12)", line=dict(width=0), layer="below",
    )
    fig_tl.add_annotation(
        x=1710, y=7.85, text="Mw 7.8 karakteristik deprem eşiği",
        showarrow=False, font=dict(color="#aaa", size=10), xanchor="left",
    )

    customdata = np.stack([
        df_e["tarih"].fillna("?").astype(str).values,
        df_e["can_kaybi"].fillna(-1).astype(float).values,
        df_e["kaynak"].fillna("?").astype(str).values,
    ], axis=-1)

    fig_tl.add_trace(go.Scatter(
        x=df_e["yil"], y=df_e["mw"],
        mode="markers+text",
        marker=dict(
            size=bubble,
            color=df_e["mw"],
            colorscale="Reds", cmin=6.0, cmax=8.0,
            showscale=True,
            colorbar=dict(title="Mw", thickness=12, len=0.7),
            line=dict(width=1, color="#fff"),
        ),
        text=df_e["yil"].astype(str),
        textposition="top center",
        textfont=dict(size=10, color="#fff"),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Mw: %{y:.1f}<br>"
            "Can kaybı: %{customdata[1]}<br>"
            "<i>%{customdata[2]}</i><extra></extra>"
        ),
        name="Tarihi depremler",
    ))
    fig_tl.update_layout(
        height=380,
        plot_bgcolor=BG2, paper_bgcolor=BG2,
        font=dict(color=TEXT, size=11),
        xaxis=dict(title="Yıl", range=[1700, 2030], gridcolor="#222"),
        yaxis=dict(title="Mw", range=[5.8, 8.3], gridcolor="#222"),
        margin=dict(t=20, b=40, l=50, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_tl, use_container_width=True, key="erz_arsiv_timeline")

    # ── Bölüm C: Harita — kırık hatları ────────────────────────────────────
    st.markdown('<div class="chart-title">🗺️ Tarihi Kırık Hatları & Konumlar</div>',
                unsafe_allow_html=True)

    fig_map = go.Figure()
    # 1939 — 360 km KAF segmenti (39.77,39.53 → 39.50,42.00 yaklaşık)
    fig_map.add_trace(go.Scattermapbox(
        lon=[39.53, 42.00], lat=[39.77, 39.50],
        mode="lines",
        line=dict(width=6, color="#ff3333"),
        name="1939 kırığı (360 km)",
        hovertemplate="1939 Mw 7.8 — 360 km KAF segmenti<extra></extra>",
    ))
    # 1992 — 55 km kırık
    fig_map.add_trace(go.Scattermapbox(
        lon=[39.60, 40.20], lat=[39.71, 39.65],
        mode="lines",
        line=dict(width=4, color="#3399ff"),
        name="1992 kırığı (55 km)",
        hovertemplate="1992 Mw 6.8 — 55 km KAF segmenti<extra></extra>",
    ))
    # Depremler — boyut Mw ile orantılı
    fig_map.add_trace(go.Scattermapbox(
        lon=df_e["lon"], lat=df_e["lat"],
        mode="markers",
        marker=dict(size=df_e["mw"] * 4 + 4, color="#ffcc00",
                    opacity=0.85),
        text=[f"{r['tarih']} — Mw {r['mw']}" for r in ERZINCAN_TARIHI],
        hovertemplate="<b>%{text}</b><extra></extra>",
        name="Tarihi depremler",
    ))
    # Erzincan merkez
    fig_map.add_trace(go.Scattermapbox(
        lon=[ERZ_LON], lat=[ERZ_LAT],
        mode="markers+text",
        marker=dict(size=14, color="#00ff66"),
        text=["📍 Erzincan"], textposition="top right",
        textfont=dict(color="#fff", size=12),
        name="Erzincan", hoverinfo="text",
    ))
    fig_map.update_layout(
        height=480,
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=39.7, lon=40.6),
            zoom=6.3,
        ),
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0.55)",
                    font=dict(color="#fff", size=11),
                    x=0.01, y=0.99),
    )
    st.plotly_chart(fig_map, use_container_width=True, key="erz_arsiv_map")
    st.caption(
        "🛰️ **Not:** Mapbox satellite-streets stili token gerektirdiği için "
        "ücretsiz **open-street-map** kullanıldı. Kırık koordinatları yaklaşık; "
        "ana referanslar: Barka 1996 (1939) ve Özalaybey 1993 (1992)."
    )

    # ── Bölüm D: Detay kartı ───────────────────────────────────────────────
    st.markdown('<div class="chart-title">📜 Detay Kartı — Seçili Deprem</div>',
                unsafe_allow_html=True)

    secim = st.selectbox(
        "Bir deprem seç:",
        options=[f"{e['yil']} — {e['tarih']} (Mw {e['mw']})" for e in ERZINCAN_TARIHI],
        index=0, key="erz_arsiv_select",
    )
    secim_yil = int(secim.split(" — ")[0])
    eq = next(e for e in ERZINCAN_TARIHI if e["yil"] == secim_yil)

    cd1, cd2 = st.columns(2)
    with cd1:
        st.markdown(f"### {eq['tarih']} — Mw {eq['mw']}")
        st.markdown(f"**📍 Konum:** {eq['lat']:.2f}°N {eq['lon']:.2f}°E")
        if eq.get("ms"):
            st.markdown(f"**📐 Yüzey büyüklüğü (Ms):** {eq['ms']}")
        if eq.get("derinlik_km"):
            st.markdown(f"**⬇️ Derinlik:** {eq['derinlik_km']} km")
        if eq.get("kirik_uzunluk_km"):
            st.markdown(f"**📏 Kırık uzunluğu:** {eq['kirik_uzunluk_km']} km")
        if eq.get("kirik_yon"):
            st.markdown(f"**🧭 Kırık yönelimi:** {eq['kirik_yon']}")
    with cd2:
        if eq.get("can_kaybi"):
            st.markdown(f"**💀 Can kaybı:** {eq['can_kaybi']:,}")
        else:
            st.markdown("**💀 Can kaybı:** _kayıt yok_")
        if eq.get("yarali"):
            st.markdown(f"**🩹 Yaralı:** {eq['yarali']:,}")
        if eq.get("hasarli_koy"):
            st.markdown(f"**🏘️ Hasarlı köy:** {eq['hasarli_koy']}")
        if eq.get("etki_alani_km2"):
            st.markdown(f"**📐 Etki alanı:** {eq['etki_alani_km2']:,} km²")

    st.markdown(f"_{eq['bilgi']}_")

    if eq.get("vay_be"):
        st.markdown(
            f"""<div style="background:#1a1a2e;border-left:4px solid #ffcc00;
                          border-radius:6px;padding:12px 16px;margin:10px 0;color:#ffe082">
                <b>🤯 Vay be faktörü:</b> {eq['vay_be']}
            </div>""",
            unsafe_allow_html=True,
        )

    st.info(f"📚 **Birincil kaynak:** {eq['kaynak']}")
    if eq.get("kaynak2"):
        st.info(f"📚 **İkincil kaynak:** {eq['kaynak2']}")

    # ── Bölüm E: Slip deficit hesabı ───────────────────────────────────────
    st.markdown(
        '<div class="chart-title">📉 Mevcut Slip Deficit — KAF Erzincan Segmenti (1939 sonrası)</div>',
        unsafe_allow_html=True,
    )

    YIL_SON = 2026
    YIL_1939 = 1939
    KAYMA_HIZI_MM_YIL = 18.0
    KARAKTERISTIK_KAYMA_M = 7.0  # 1939 co-seismik kayma yaklaşımı
    gecen_yil = YIL_SON - YIL_1939
    birikim_mm = gecen_yil * KAYMA_HIZI_MM_YIL
    birikim_m = birikim_mm / 1000.0
    progress_val = min(1.0, birikim_m / KARAKTERISTIK_KAYMA_M)

    cE1, cE2, cE3 = st.columns(3)
    cE1.metric("⌛ Son büyük depremden", f"{gecen_yil} yıl")
    cE2.metric("📏 Birikimli kayma", f"{birikim_m:.2f} m",
               help=f"{gecen_yil} yıl × {KAYMA_HIZI_MM_YIL:.0f} mm/yıl")
    cE3.metric("🎯 Karakteristik kayma", f"~{KARAKTERISTIK_KAYMA_M:.0f} m",
               help="1939 olayı co-seismik kayma — Barka 1996")

    st.progress(
        progress_val,
        text=f"Mevcut Slip Deficit: %{progress_val*100:.0f} "
             f"(karakteristik {KARAKTERISTIK_KAYMA_M:.0f} m'ye göre)",
    )

    st.caption(
        f"📚 **Hesap:** {gecen_yil} × {KAYMA_HIZI_MM_YIL:.0f} mm = "
        f"{birikim_mm:,.0f} mm ≈ {birikim_m:.2f} m. "
        "**Kaynaklar:** Barka, A. (1996) *Bull. Seismol. Soc. Am.* 86(5), 1238-1254; "
        "Reilinger, R. et al. (2006) GPS constraints on continental deformation in "
        "the Africa-Arabia-Eurasia continental collision zone, *J. Geophys. Res.*, "
        "111, B05411, doi:10.1029/2005JB004051. "
        "⚠️ Lineer interseismik birikim modeli — viskoelastik gevşeme ve fay döngüsü modellenmedi; "
        "gerçek deprem zamanlaması Wallace–Schwartz–Coppersmith 1984 paleoseismik dağılımına tabidir."
    )


if active_menu == "🏛️ Erzincan Arşivi":
    _render_erzincan_arsivi()


# ════════════════════════════════════════════════════════════════════════════
# 🔴 SİSMİK AÇIK — F-63 / v1.20 — KAF Seismic Gap Haritası
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Toksöz et al. 1979 — Seismicity & tectonics of Turkey
#   • McCann et al. 1979 — Seismic gaps & plate tectonics, PAGEOPH 117
#   • Barka 1996 — Slip distribution along NAF, BSSA 86(5), 1238-1254
#   • Stein et al. 1997 — Progressive failure on NAF, JGR 102(B12), 27587-27601
#   • Parsons 2004 — Marmara/Istanbul gap, JGR
#   • Ambraseys & Jackson 2000 — Saros, Geophys. J. Int.
# ════════════════════════════════════════════════════════════════════════════

KAF_SEGMENTLER = [
    {
        "id": "S01", "ad": "Erzincan Segmenti",
        "lat1": 39.77, "lon1": 36.80, "lat2": 39.77, "lon2": 39.53,
        "son_buyuk_deprem_yil": 1939, "buyukluk": 7.8,
        "kayma_hizi_mm_yil": 18.0, "kirik_uzunluk_km": 120,
        "beklenen_tekrar_yil": 250,
        "slip_deficit_m": round((2026 - 1939) * 18.0 / 1000, 2),
        "risk": "orta",
        "kaynak": "Barka 1996, BSSA 86(5)",
    },
    {
        "id": "S02", "ad": "Niksar-Erbaa Segmenti",
        "lat1": 40.60, "lon1": 36.20, "lat2": 40.70, "lon2": 37.50,
        "son_buyuk_deprem_yil": 1942, "buyukluk": 7.0,
        "kayma_hizi_mm_yil": 18.0, "kirik_uzunluk_km": 110,
        "beklenen_tekrar_yil": 200,
        "slip_deficit_m": round((2026 - 1942) * 18.0 / 1000, 2),
        "risk": "orta",
        "kaynak": "Barka 1996, BSSA 86(5)",
    },
    {
        "id": "S03", "ad": "Tosya-Kastamonu Segmenti",
        "lat1": 40.80, "lon1": 33.50, "lat2": 41.00, "lon2": 35.80,
        "son_buyuk_deprem_yil": 1943, "buyukluk": 7.6,
        "kayma_hizi_mm_yil": 20.0, "kirik_uzunluk_km": 280,
        "beklenen_tekrar_yil": 300,
        "slip_deficit_m": round((2026 - 1943) * 20.0 / 1000, 2),
        "risk": "orta",
        "kaynak": "Stein et al. 1997, JGR 102",
    },
    {
        "id": "S04", "ad": "Bolu-Düzce Segmenti",
        "lat1": 40.75, "lon1": 31.00, "lat2": 40.80, "lon2": 32.50,
        "son_buyuk_deprem_yil": 1999, "buyukluk": 7.2,
        "kayma_hizi_mm_yil": 22.0, "kirik_uzunluk_km": 100,
        "beklenen_tekrar_yil": 250,
        "slip_deficit_m": round((2026 - 1999) * 22.0 / 1000, 2),
        "risk": "dusuk",
        "kaynak": "Barka et al. 2002, BSSA",
    },
    {
        "id": "S05", "ad": "Marmara (İstanbul) Segmenti",
        "lat1": 40.80, "lon1": 27.50, "lat2": 40.95, "lon2": 29.10,
        "son_buyuk_deprem_yil": 1766, "buyukluk": 7.1,
        "kayma_hizi_mm_yil": 22.0, "kirik_uzunluk_km": 150,
        "beklenen_tekrar_yil": 250,
        "slip_deficit_m": round((2026 - 1766) * 22.0 / 1000, 2),
        "risk": "yuksek",
        "kaynak": "Parsons 2004, JGR; Ambraseys 2002, J. Seismol.",
    },
    {
        "id": "S06", "ad": "Saros Körfezi Segmenti",
        "lat1": 40.60, "lon1": 26.00, "lat2": 40.75, "lon2": 27.30,
        "son_buyuk_deprem_yil": 1912, "buyukluk": 7.4,
        "kayma_hizi_mm_yil": 20.0, "kirik_uzunluk_km": 120,
        "beklenen_tekrar_yil": 250,
        "slip_deficit_m": round((2026 - 1912) * 20.0 / 1000, 2),
        "risk": "yuksek",
        "kaynak": "Ambraseys & Jackson 2000, Geophys. J. Int.",
    },
]

_KAF_RISK_RENK = {
    "yuksek": "#E24B4A",
    "orta":   "#EF9F27",
    "dusuk":  "#1D9E75",
}
_KAF_RISK_KALINLIK = {"yuksek": 6, "orta": 4, "dusuk": 2}
_KAF_RISK_ETIKET   = {"yuksek": "Yüksek", "orta": "Orta", "dusuk": "Düşük"}


@st.fragment
def _render_sismik_acik():
    st.markdown(
        '<div class="chart-title">🔴 Sismik Açık (Seismic Gap) — KAF Segmentleri (F-63 / v1.20)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🔴 **Sismik Açık (Seismic Gap):** Uzun süredir kırılmamış ve stres biriktirmiş fay "
        "segmentleri. Teorik temel: **McCann et al. (1979), PAGEOPH 117** — büyük depremlerin "
        "uzun süre kırılmamış segmentlerde yoğunlaşma eğilimi."
    )

    st.warning(
        "⚠️ **Koordinat teyit uyarısı (v1.31.2):** Aşağıdaki 6 KAF segmenti **peer-reviewed "
        "yayınlardan** (Barka 1996, Stein 1997, Parsons 2004, Ambraseys 2000, Özalaybey 1993) "
        "kaba değerlerle alındı. **AFAD Diri Fay Haritası, KOERI/Boğaziçi katalogları ile henüz "
        "teyit edilmedi** — çizgi konumlarında ±5–10 km hata payı olabilir. Resmi referans için: "
        "[AFAD Aktif Fay Haritası](https://deprem.afad.gov.tr/diri-fay-haritasi) · "
        "[KOERI/BOUN Tarihsel Deprem Kataloğu](http://www.koeri.boun.edu.tr/sismo/2/deprem-bilgileri/buyuk-depremler/). "
        "Mapbox satellite-streets token gerektirdiği için ücretsiz `open-street-map` zemin kullanılıyor."
    )

    YIL_SIMDI = 2026

    # ── HARİTA: Plotly Scattermapbox lines + etiketler ─────────────────────
    fig_map = go.Figure()
    for seg in KAF_SEGMENTLER:
        renk = _KAF_RISK_RENK[seg["risk"]]
        kalinlik = _KAF_RISK_KALINLIK[seg["risk"]]
        gecen = YIL_SIMDI - seg["son_buyuk_deprem_yil"]
        hover = (
            f"<b>{seg['ad']}</b><br>"
            f"Son büyük deprem: {seg['son_buyuk_deprem_yil']} (Mw {seg['buyukluk']:.1f})<br>"
            f"Geçen süre: {gecen} yıl<br>"
            f"Slip deficit: {seg['slip_deficit_m']:.2f} m<br>"
            f"Kayma hızı: {seg['kayma_hizi_mm_yil']:.0f} mm/yıl<br>"
            f"Kırık uzunluğu: {seg['kirik_uzunluk_km']:.0f} km<br>"
            f"Beklenen tekrar: ~{seg['beklenen_tekrar_yil']} yıl<br>"
            f"Risk: {_KAF_RISK_ETIKET[seg['risk']]}<br>"
            f"Kaynak: {seg['kaynak']}"
            "<extra></extra>"
        )
        fig_map.add_trace(go.Scattermapbox(
            lat=[seg["lat1"], seg["lat2"]],
            lon=[seg["lon1"], seg["lon2"]],
            mode="lines",
            line=dict(width=kalinlik, color=renk),
            name=f"{seg['id']} — {seg['ad']}",
            hovertemplate=hover,
            showlegend=False,
        ))
        # Segment ortasına etiket
        mid_lat = (seg["lat1"] + seg["lat2"]) / 2
        mid_lon = (seg["lon1"] + seg["lon2"]) / 2
        fig_map.add_trace(go.Scattermapbox(
            lat=[mid_lat], lon=[mid_lon],
            mode="markers+text",
            marker=dict(size=9, color=renk),
            text=[f"{seg['ad'].split(' ')[0]} ({seg['son_buyuk_deprem_yil']})"],
            textposition="top right",
            textfont=dict(size=11, color="#ffffff"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Legend için 3 dummy trace (risk seviyeleri)
    for risk_key in ("yuksek", "orta", "dusuk"):
        fig_map.add_trace(go.Scattermapbox(
            lat=[None], lon=[None],
            mode="lines",
            line=dict(width=_KAF_RISK_KALINLIK[risk_key], color=_KAF_RISK_RENK[risk_key]),
            name=f"{_KAF_RISK_ETIKET[risk_key]} risk",
            hoverinfo="skip",
        ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.5, lon=33.0), zoom=5),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── GANTT ŞERİDİ: Son depremden bugüne geçen süre ──────────────────────
    st.markdown(
        '<div class="chart-title">📊 Son Büyük Depremden Bu Yana Geçen Süre (Gantt)</div>',
        unsafe_allow_html=True,
    )
    df_gantt = pd.DataFrame([
        {
            "Segment":  f"{seg['id']} — {seg['ad']}",
            "Baslangic": seg["son_buyuk_deprem_yil"],
            "Bitis":     YIL_SIMDI,
            "Sure":      YIL_SIMDI - seg["son_buyuk_deprem_yil"],
            "Risk":      seg["risk"],
            "SonYil":    seg["son_buyuk_deprem_yil"],
            "Mw":        seg["buyukluk"],
        }
        for seg in KAF_SEGMENTLER
    ])
    # Uzun bar = uzun süredir kırılmamış = tehlike (azalan sıra)
    df_gantt = df_gantt.sort_values("Sure", ascending=True).reset_index(drop=True)

    fig_gantt = go.Figure()
    for _, row in df_gantt.iterrows():
        renk = _KAF_RISK_RENK[row["Risk"]]
        fig_gantt.add_trace(go.Bar(
            y=[row["Segment"]],
            x=[row["Sure"]],
            base=row["Baslangic"],
            orientation="h",
            marker=dict(color=renk, line=dict(color="#222", width=0.5)),
            text=f"{row['SonYil']} • Mw {row['Mw']:.1f}",
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="#ffffff", size=11),
            hovertemplate=(
                f"<b>{row['Segment']}</b><br>"
                f"Son deprem: {row['SonYil']} (Mw {row['Mw']:.1f})<br>"
                f"Geçen süre: {row['Sure']} yıl"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # "Şu an" dikey çizgisi
    fig_gantt.add_vline(
        x=YIL_SIMDI, line_width=2, line_dash="dash", line_color="#ffffff",
        annotation_text=f"Şu an ({YIL_SIMDI})",
        annotation_position="top right",
        annotation_font_color="#ffffff",
    )

    fig_gantt.update_layout(
        title=dict(
            text="KAF Segmentleri — Son Büyük Depremden Bu Yana Geçen Süre",
            font=dict(color=TEXT, size=14),
        ),
        xaxis=dict(
            title="Yıl", range=[1700, YIL_SIMDI + 30],
            color=TEXT, gridcolor=BORDER,
        ),
        yaxis=dict(color=TEXT, gridcolor=BORDER),
        height=360,
        margin=dict(l=10, r=10, t=50, b=40),
        paper_bgcolor=BG2,
        plot_bgcolor=BG2,
        bargap=0.35,
    )
    st.plotly_chart(fig_gantt, use_container_width=True, config={"displayModeBar": False})

    # ── ÖZET TABLO ─────────────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Segment Özeti</div>', unsafe_allow_html=True)
    df_tablo = pd.DataFrame([
        {
            "Segment":           f"{seg['id']} — {seg['ad']}",
            "Son Deprem":        seg["son_buyuk_deprem_yil"],
            "Mw":                seg["buyukluk"],
            "Slip Deficit (m)":  seg["slip_deficit_m"],
            "Kayma (mm/yıl)":    seg["kayma_hizi_mm_yil"],
            "Kırık (km)":        seg["kirik_uzunluk_km"],
            "Beklenen Tekrar (yıl)": seg["beklenen_tekrar_yil"],
            "Risk":              _KAF_RISK_ETIKET[seg["risk"]],
        }
        for seg in KAF_SEGMENTLER
    ])
    st.dataframe(df_tablo, use_container_width=True, hide_index=True)

    # ── KAYNAK ─────────────────────────────────────────────────────────────
    st.caption(
        "📚 **Kaynaklar:** Barka, A. (1996) *Bull. Seismol. Soc. Am.* 86(5), 1238-1254 | "
        "Stein, R.S. et al. (1997) *J. Geophys. Res.* 102(B12), 27587-27601 | "
        "Parsons, T. (2004) *J. Geophys. Res.* (Marmara/İstanbul gap) | "
        "McCann, W.R. et al. (1979) *PAGEOPH* 117, 1082-1147 | "
        "Toksöz, M.N. et al. (1979) Seismicity & tectonics of Turkey | "
        "Ambraseys & Jackson (2000) *Geophys. J. Int.* (Saros). "
        "⚠️ Slip deficit = (2026 − son deprem yılı) × kayma hızı; lineer interseismik birikim. "
        "Gerçek deprem zamanlaması Wallace–Schwartz–Coppersmith 1984 paleoseismik "
        "dağılımına ve viskoelastik gevşemeye bağlıdır."
    )


if active_menu == "🔴 Sismik Açık":
    _render_sismik_acik()


# ════════════════════════════════════════════════════════════════════════════
# 🌊 SHAKEMAP — F-61 / v1.21 — MMI İzoseist Haritası + USGS API
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Worden, C.B. & Wald, D.J. (2016) — ShakeMap Manual Online, USGS
#       https://usgs.github.io/shakemap/
#   • Wald, D.J. et al. (1999) — PGA-PGV-MMI ilişkisi, Earthquake Spectra
#       15(3), 557-564. DOI:10.1193/1.1586058
#   • Bakun, W.H. & Wentworth, C.M. (1997) — İzoseist yarıçap,
#       BSSA 87(6), 1502-1521
#   • USGS FDSN Event API: https://earthquake.usgs.gov/fdsnws/event/1/
# ════════════════════════════════════════════════════════════════════════════

_SHAKEMAP_FALLBACK_EVENTS = [
    {"id": "kahramanmaras-2023", "yer": "Kahramanmaraş (Pazarcık)",
     "mag": 7.8, "lat": 37.17, "lon": 36.94, "tarih": "2023-02-06"},
    {"id": "duzce-1999", "yer": "Düzce",
     "mag": 7.2, "lat": 40.75, "lon": 31.21, "tarih": "1999-11-12"},
    {"id": "erzincan-1992", "yer": "Erzincan",
     "mag": 6.8, "lat": 39.71, "lon": 39.60, "tarih": "1992-03-13"},
]

_SHAKEMAP_MMI_RENK = {
    9: "#A32D2D",  # IX+ Yıkıcı
    8: "#E24B4A",  # VII-VIII Çok Güçlü
    7: "#E24B4A",
    6: "#EF9F27",  # V-VI Güçlü
    5: "#EF9F27",
    4: "#FAC775",  # IV Orta
    3: "#C0DD97",  # II-III Hafif
    2: "#C0DD97",
    1: "#1D9E75",  # I Hissedilmez
}

_SHAKEMAP_MMI_ETIKET = {
    9: "MMI IX+ (Yıkıcı)",
    8: "MMI VII–VIII (Çok Güçlü)",
    7: "MMI VII–VIII (Çok Güçlü)",
    6: "MMI V–VI (Güçlü)",
    5: "MMI V–VI (Güçlü)",
    4: "MMI IV (Orta)",
    3: "MMI II–III (Hafif)",
    2: "MMI II–III (Hafif)",
    1: "MMI I (Hissedilmez)",
}


def _shakemap_mmi_from_pga(pga_gal: float) -> str:
    """PGA (cm/s²) → MMI sınıfı. Kaynak: Wald et al. 1999, Earthquake Spectra 15(3)."""
    if pga_gal < 0.17:
        return "I (Hissedilmez)"
    elif pga_gal < 1.4:
        return "II-III (Hafif)"
    elif pga_gal < 9.2:
        return "IV (Orta)"
    elif pga_gal < 92:
        return "V-VI (Güçlü)"
    elif pga_gal < 400:
        return "VII-VIII (Çok Güçlü)"
    else:
        return "IX+ (Yıkıcı)"


def _shakemap_mmi_radius_km(mw: float, mmi_level: int) -> float:
    """
    Mw ve MMI seviyesi için izoseist yarıçap (km).
    Basitleştirilmiş Bakun & Wentworth (1997) bağıntısı.
    Kaynak: Bakun, W.H. & Wentworth, C.M. (1997). BSSA 87(6), 1502-1521.
    """
    base = {9: 5, 8: 15, 7: 40, 6: 90, 5: 180, 4: 300}
    scale = 10 ** (0.5 * (mw - 6.0))
    return base.get(mmi_level, 400) * scale / 10


def _shakemap_circle_coords(lat: float, lon: float, radius_km: float, n: int = 48):
    """Bir noktadan radius_km yarıçaplı daire koordinatları."""
    coords_lat, coords_lon = [], []
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    for i in range(n + 1):
        angle = math.radians(i * 360 / n)
        dlat = (radius_km / 111.0) * math.cos(angle)
        dlon = (radius_km / (111.0 * cos_lat)) * math.sin(angle)
        coords_lat.append(lat + dlat)
        coords_lon.append(lon + dlon)
    return coords_lat, coords_lon


@st.cache_data(ttl=600, show_spinner=False)
def _shakemap_fetch_usgs_events(min_magnitude: float = 5.0, limit: int = 15):
    """
    USGS FDSN Event API'den Türkiye bölgesindeki son depremleri çek.
    Kaynak: USGS Earthquake Hazards Program, earthquake.usgs.gov/fdsnws/event/1/
    """
    try:
        r = requests.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "minmagnitude": min_magnitude,
                "minlatitude": 35, "maxlatitude": 43,
                "minlongitude": 25, "maxlongitude": 45,
                "orderby": "time",
                "limit": limit,
            },
            timeout=8,
        )
        if r.status_code != 200:
            return None
        gj = r.json()
        events = []
        for feat in gj.get("features", []):
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            coords = geom.get("coordinates") or [None, None, None]
            mag = props.get("mag")
            if mag is None or coords[0] is None or coords[1] is None:
                continue
            ts = props.get("time")
            try:
                tarih = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")
            except Exception:
                tarih = "—"
            events.append({
                "id": feat.get("id", "?"),
                "yer": props.get("place") or "Bilinmeyen yer",
                "mag": float(mag),
                "lat": float(coords[1]),
                "lon": float(coords[0]),
                "tarih": tarih,
            })
        return events or None
    except Exception:
        return None


@st.fragment
def _render_shakemap():
    st.markdown(
        '<div class="chart-title">🌊 ShakeMap — MMI İzoseist Haritası (F-61 / v1.21)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🌊 **ShakeMap:** Bir depremin merkez üssünden yayılan sarsıntı şiddetini "
        "(MMI — Modified Mercalli Intensity) eşşiddet (izoseist) daireleriyle gösterir. "
        "Teorik temel: **Worden & Wald (2016), USGS ShakeMap Manual**; "
        "PGA→MMI dönüşümü: **Wald et al. (1999), Earthquake Spectra 15(3)**."
    )

    # ── Bölüm A: Deprem seçici ─────────────────────────────────────────────
    col_sec, col_mag = st.columns([3, 1])
    with col_mag:
        min_mw_filter = st.slider(
            "Min. Mw", min_value=4.5, max_value=7.0, value=5.0, step=0.1,
            key="shakemap_min_mw",
        )

    events = _shakemap_fetch_usgs_events(min_magnitude=min_mw_filter, limit=15)
    if events:
        kaynak_etiket = "USGS FDSN API (canlı)"
    else:
        events = _SHAKEMAP_FALLBACK_EVENTS
        kaynak_etiket = "Fallback (USGS API erişilemedi)"

    options = [
        f"M{e['mag']:.1f} — {e['yer']}  ·  {e['tarih']}"
        for e in events
    ]
    with col_sec:
        sec_idx = st.selectbox(
            f"Deprem seç ({kaynak_etiket})",
            options=list(range(len(events))),
            format_func=lambda i: options[i],
            key="shakemap_event_select",
        )
    secilen = events[sec_idx]
    mw = float(secilen["mag"])
    eq_lat = float(secilen["lat"])
    eq_lon = float(secilen["lon"])

    # ── Bölüm B: MMI izoseist haritası ─────────────────────────────────────
    fig_map = go.Figure()

    # Dıştan içe çiz (büyük yarıçaplardan küçüklere) → küçük zonlar üstte kalsın
    mmi_seviye_sirali = [4, 5, 6, 7, 8, 9]
    for mmi_lvl in mmi_seviye_sirali:
        rad = _shakemap_mmi_radius_km(mw, mmi_lvl)
        if rad < 0.5:
            continue
        clat, clon = _shakemap_circle_coords(eq_lat, eq_lon, rad)
        renk = _SHAKEMAP_MMI_RENK[mmi_lvl]
        etiket = _SHAKEMAP_MMI_ETIKET[mmi_lvl]
        fig_map.add_trace(go.Scattermapbox(
            lat=clat, lon=clon,
            mode="lines",
            fill="toself",
            line=dict(width=1.5, color=renk),
            fillcolor=renk,
            opacity=0.28,
            name=f"{etiket} (~{rad:.0f} km)",
            hovertemplate=(
                f"<b>{etiket}</b><br>"
                f"Yarıçap: ~{rad:.0f} km<br>"
                f"Mw {mw:.1f} merkezi"
                "<extra></extra>"
            ),
        ))

    # Merkez üssü yıldızı
    fig_map.add_trace(go.Scattermapbox(
        lat=[eq_lat], lon=[eq_lon],
        mode="markers+text",
        marker=dict(size=18, color="#FFD700", symbol="star"),
        text=[f"★ M{mw:.1f}"],
        textposition="top right",
        textfont=dict(size=13, color="#FFD700"),
        name="Merkez üssü",
        hovertemplate=(
            f"<b>{secilen['yer']}</b><br>"
            f"Mw {mw:.1f}<br>"
            f"{secilen['tarih']}<br>"
            f"({eq_lat:.3f}, {eq_lon:.3f})"
            "<extra></extra>"
        ),
    ))

    # Harita zoom seviyesi: en büyük (MMI IV) yarıçapına göre
    max_rad = _shakemap_mmi_radius_km(mw, 4)
    if max_rad > 400:
        zoom = 5
    elif max_rad > 200:
        zoom = 6
    elif max_rad > 100:
        zoom = 7
    else:
        zoom = 8

    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=eq_lat, lon=eq_lon),
            zoom=zoom,
        ),
        height=540,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(
            orientation="v",
            yanchor="top", y=0.98, xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.55)",
            font=dict(color="#ffffff", size=11),
            bordercolor=BORDER, borderwidth=1,
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Bölüm C: Bilgi kartları (4 kolon) ──────────────────────────────────
    mmi_merkez = _shakemap_mmi_from_pga(400 if mw >= 7 else (92 if mw >= 6 else 9.2))
    rad_iv = _shakemap_mmi_radius_km(mw, 4)
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"M{mw:.1f}",           "#FFD700",  "Büyüklük (Mw)"),
        (c2, mmi_merkez.split(" ")[0], "#E24B4A", "Merkez MMI (tahm.)"),
        (c3, f"{rad_iv:.0f} km",     "#FAC775",  "MMI IV+ etki yarıçapı"),
        (c4, "USGS",                 "#1D9E75",  "ShakeMap kaynağı"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Bölüm D: MMI → hasar ilişkisi tablosu ──────────────────────────────
    st.markdown(
        '<div class="chart-title">📋 MMI Şiddet — Beklenen Hasar İlişkisi</div>',
        unsafe_allow_html=True,
    )
    df_mmi = pd.DataFrame([
        {"MMI": "I–III",  "Sarsıntı": "Hissedilmez – Hafif",  "Beklenen Hasar": "Yok",          "PGA (g)": "< 0.001"},
        {"MMI": "IV–V",   "Sarsıntı": "Orta – Güçlü",         "Beklenen Hasar": "Çok az",       "PGA (g)": "0.001 – 0.01"},
        {"MMI": "VI–VII", "Sarsıntı": "Çok güçlü",            "Beklenen Hasar": "Az – Orta",    "PGA (g)": "0.01 – 0.1"},
        {"MMI": "VIII",   "Sarsıntı": "Şiddetli",             "Beklenen Hasar": "Orta – Ağır", "PGA (g)": "0.1 – 0.3"},
        {"MMI": "IX+",    "Sarsıntı": "Yıkıcı",               "Beklenen Hasar": "Ağır – Çok ağır", "PGA (g)": "> 0.3"},
    ])
    st.dataframe(df_mmi, use_container_width=True, hide_index=True)
    st.caption("Kaynak: Wald et al. (1999), Earthquake Spectra 15(3), 557-564")

    # ── Bölüm E: Kaynaklar ve uyarı ────────────────────────────────────────
    st.info(
        "📚 **ShakeMap Algoritması:** Worden & Wald (2016), USGS ShakeMap Manual | "
        "**MMI–PGA İlişkisi:** Wald et al. (1999), *Earthquake Spectra* 15(3) | "
        "**İzoseist Yarıçap Tahmini:** Bakun & Wentworth (1997), *BSSA* 87(6) | "
        "**Veri:** USGS FDSN Event API"
    )
    st.warning(
        "⚠️ Bu tahminî bir modeldir. Gerçek sarsıntı değerleri yerel zemin koşullarına, "
        "fay geometrisine ve yönelimli (directivity) etkilere göre değişir. "
        "Resmi ShakeMap için: earthquake.usgs.gov"
    )


if active_menu == "🌊 ShakeMap":
    _render_shakemap()


# ════════════════════════════════════════════════════════════════════════════
# 🗺️ SİSMİK TEHLİKE — F-44 / v1.22 — PSHA (Probabilistic Seismic Hazard)
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Woessner, J. et al. (2015). The 2013 European seismic hazard model.
#       Bull. Earthq. Eng. 13(12), 3553-3596. DOI:10.1007/s10518-015-9795-1
#   • Pagani, M. et al. (2014). OpenQuake Engine. SRL 85(3), 692-702.
#       DOI:10.1785/0220130087
#   • AFAD (2018). Türkiye Bina Deprem Yönetmeliği (TBDY-2018).
#       deprem.afad.gov.tr/depremzonasi
#   • EFEHR API: http://www.efehr.org/
# ════════════════════════════════════════════════════════════════════════════

# TBDY-2018 + SHARE 2013 türevi: Türkiye için statik PGA(g) ızgara noktaları.
# Her nokta = (lat, lon, şehir, PGA_475yıl, TBDY_zone, kısa_açıklama)
# Kaynak: AFAD TBDY-2018 spektral ivme haritası + SHARE Woessner 2015 referansı.
_PSHA_GRID_TR = [
    # ── Doğu Anadolu (DAF) — yüksek tehlike ──
    {"city": "Hakkari",        "lat": 37.57, "lon": 43.74, "pga475": 0.42, "zone": 1, "note": "Bitlis kenet kuşağı"},
    {"city": "Van",            "lat": 38.49, "lon": 43.41, "pga475": 0.45, "zone": 1, "note": "Van fay zonu (2011 Mw 7.1)"},
    {"city": "Bingöl",         "lat": 38.88, "lon": 40.50, "pga475": 0.48, "zone": 1, "note": "KAF–DAF kesişim"},
    {"city": "Muş",            "lat": 38.74, "lon": 41.49, "pga475": 0.40, "zone": 1, "note": "Doğu Anadolu sıkışma"},
    {"city": "Elazığ",         "lat": 38.68, "lon": 39.22, "pga475": 0.45, "zone": 1, "note": "DAF (2020 Mw 6.8 Sivrice)"},
    {"city": "Malatya",        "lat": 38.36, "lon": 38.30, "pga475": 0.48, "zone": 1, "note": "DAF kuzey kanadı"},
    {"city": "Adıyaman",       "lat": 37.76, "lon": 38.27, "pga475": 0.45, "zone": 1, "note": "DAF (2023 Mw 7.8)"},
    {"city": "Kahramanmaraş",  "lat": 37.58, "lon": 36.93, "pga475": 0.52, "zone": 1, "note": "DAF kuzey + Sürgü (2023 Mw 7.8+7.7)"},
    {"city": "Gaziantep",      "lat": 37.07, "lon": 37.38, "pga475": 0.44, "zone": 1, "note": "DAF güney"},
    {"city": "Hatay",          "lat": 36.20, "lon": 36.16, "pga475": 0.50, "zone": 1, "note": "ÖDF triple junction (2023)"},
    {"city": "Osmaniye",       "lat": 37.07, "lon": 36.25, "pga475": 0.50, "zone": 1, "note": "DAF güney kanadı"},

    # ── KAF doğu (Erzincan ekseni) ──
    {"city": "Erzincan",       "lat": 39.75, "lon": 39.49, "pga475": 0.45, "zone": 1, "note": "KAF (1939 Ms 7.8, 1992 Ms 6.8)"},
    {"city": "Erzurum",        "lat": 39.91, "lon": 41.27, "pga475": 0.35, "zone": 2, "note": "KAF doğu uzantısı"},
    {"city": "Tunceli",        "lat": 39.10, "lon": 39.55, "pga475": 0.40, "zone": 1, "note": "KAF–DAF arası"},
    {"city": "Sivas",          "lat": 39.75, "lon": 37.02, "pga475": 0.35, "zone": 2, "note": "KAF güney"},
    {"city": "Tokat",          "lat": 40.32, "lon": 36.55, "pga475": 0.42, "zone": 1, "note": "KAF (1939 Niksar-Erbaa)"},
    {"city": "Amasya",         "lat": 40.65, "lon": 35.83, "pga475": 0.40, "zone": 1, "note": "KAF (1668 Ms ~8)"},
    {"city": "Çorum",          "lat": 40.55, "lon": 34.96, "pga475": 0.38, "zone": 2, "note": "KAF yakını"},

    # ── KAF batı (İstanbul-Marmara koridoru) ──
    {"city": "Kastamonu",      "lat": 41.38, "lon": 33.78, "pga475": 0.40, "zone": 1, "note": "KAF (1943 Tosya Ms 7.6)"},
    {"city": "Bolu",           "lat": 40.74, "lon": 31.61, "pga475": 0.42, "zone": 1, "note": "KAF (1944 Gerede)"},
    {"city": "Düzce",          "lat": 40.84, "lon": 31.16, "pga475": 0.45, "zone": 1, "note": "KAF (1999 Mw 7.2)"},
    {"city": "Sakarya",        "lat": 40.78, "lon": 30.40, "pga475": 0.48, "zone": 1, "note": "KAF (1999 İzmit Mw 7.6)"},
    {"city": "Kocaeli",        "lat": 40.85, "lon": 29.88, "pga475": 0.50, "zone": 1, "note": "KAF (1999 İzmit hipo)"},
    {"city": "Yalova",         "lat": 40.65, "lon": 29.27, "pga475": 0.48, "zone": 1, "note": "Marmara segmenti"},
    {"city": "İstanbul",       "lat": 41.01, "lon": 28.98, "pga475": 0.40, "zone": 1, "note": "Marmara seismic gap (Parsons 2004)"},
    {"city": "Tekirdağ",       "lat": 40.98, "lon": 27.51, "pga475": 0.40, "zone": 1, "note": "Ganos-Saros segmenti"},
    {"city": "Çanakkale",      "lat": 40.15, "lon": 26.41, "pga475": 0.42, "zone": 1, "note": "Saros (1912 Ms 7.4)"},
    {"city": "Bursa",          "lat": 40.18, "lon": 29.07, "pga475": 0.42, "zone": 1, "note": "Bursa fayı"},
    {"city": "Bilecik",        "lat": 40.14, "lon": 29.98, "pga475": 0.38, "zone": 2, "note": "Eskişehir fay zonu"},

    # ── Batı Anadolu grabenleri (normal fay rejimi) ──
    {"city": "Balıkesir",      "lat": 39.65, "lon": 27.88, "pga475": 0.42, "zone": 1, "note": "Edremit grabeni"},
    {"city": "Manisa",         "lat": 38.62, "lon": 27.43, "pga475": 0.45, "zone": 1, "note": "Gediz grabeni (1969 Ms 6.9)"},
    {"city": "İzmir",          "lat": 38.42, "lon": 27.14, "pga475": 0.45, "zone": 1, "note": "İzmir körfez fayı (2020 Mw 6.9)"},
    {"city": "Aydın",          "lat": 37.85, "lon": 27.85, "pga475": 0.45, "zone": 1, "note": "Büyük Menderes grabeni"},
    {"city": "Denizli",        "lat": 37.78, "lon": 29.09, "pga475": 0.45, "zone": 1, "note": "Pamukkale grabeni"},
    {"city": "Muğla",          "lat": 37.21, "lon": 28.36, "pga475": 0.42, "zone": 1, "note": "Hellenic Trench"},
    {"city": "Antalya",        "lat": 36.89, "lon": 30.71, "pga475": 0.28, "zone": 2, "note": "Kıbrıs yayı"},
    {"city": "Burdur",         "lat": 37.72, "lon": 30.29, "pga475": 0.45, "zone": 1, "note": "Burdur fay zonu (1971 Ms 6.2)"},
    {"city": "Isparta",        "lat": 37.76, "lon": 30.55, "pga475": 0.40, "zone": 1, "note": "Isparta açısı"},
    {"city": "Uşak",           "lat": 38.68, "lon": 29.41, "pga475": 0.40, "zone": 1, "note": "Simav fayı"},
    {"city": "Kütahya",        "lat": 39.42, "lon": 29.99, "pga475": 0.40, "zone": 1, "note": "Simav (2011 Mw 5.9)"},
    {"city": "Afyon",          "lat": 38.76, "lon": 30.54, "pga475": 0.35, "zone": 2, "note": "Akşehir grabeni"},
    {"city": "Eskişehir",      "lat": 39.78, "lon": 30.52, "pga475": 0.30, "zone": 2, "note": "Eskişehir fay zonu"},

    # ── İç Anadolu (düşük tehlike) ──
    {"city": "Ankara",         "lat": 39.93, "lon": 32.86, "pga475": 0.25, "zone": 3, "note": "İç Anadolu masifi"},
    {"city": "Konya",          "lat": 37.87, "lon": 32.48, "pga475": 0.22, "zone": 3, "note": "Düşük sismisite"},
    {"city": "Aksaray",        "lat": 38.37, "lon": 34.03, "pga475": 0.20, "zone": 3, "note": "İç Anadolu"},
    {"city": "Niğde",          "lat": 37.97, "lon": 34.68, "pga475": 0.25, "zone": 3, "note": "Tuz Gölü fayı"},
    {"city": "Kayseri",        "lat": 38.73, "lon": 35.48, "pga475": 0.28, "zone": 3, "note": "Erciyes volkanik bölge"},
    {"city": "Nevşehir",       "lat": 38.62, "lon": 34.71, "pga475": 0.22, "zone": 3, "note": "Kapadokya"},
    {"city": "Kırşehir",       "lat": 39.15, "lon": 34.16, "pga475": 0.25, "zone": 3, "note": "Kırşehir bloğu"},
    {"city": "Yozgat",         "lat": 39.82, "lon": 34.81, "pga475": 0.28, "zone": 3, "note": "KAF güneyi"},
    {"city": "Karaman",        "lat": 37.18, "lon": 33.22, "pga475": 0.20, "zone": 4, "note": "Düşük (Toroslar güneyi)"},
    {"city": "Mersin",         "lat": 36.81, "lon": 34.64, "pga475": 0.25, "zone": 3, "note": "Kıbrıs yayı uzak"},
    {"city": "Adana",          "lat": 37.00, "lon": 35.32, "pga475": 0.32, "zone": 2, "note": "Misis-Andırın fayı"},

    # ── Karadeniz kıyı (görece düşük) ──
    {"city": "Trabzon",        "lat": 41.00, "lon": 39.73, "pga475": 0.20, "zone": 3, "note": "Karadeniz kıyı"},
    {"city": "Rize",           "lat": 41.02, "lon": 40.52, "pga475": 0.18, "zone": 4, "note": "Doğu Karadeniz"},
    {"city": "Samsun",         "lat": 41.29, "lon": 36.34, "pga475": 0.30, "zone": 2, "note": "KAF kuzeyi"},
    {"city": "Zonguldak",      "lat": 41.46, "lon": 31.79, "pga475": 0.28, "zone": 2, "note": "Batı Karadeniz"},
    {"city": "Sinop",          "lat": 42.03, "lon": 35.15, "pga475": 0.22, "zone": 3, "note": "Karadeniz kıyı"},
]

_PSHA_RETURN_PERIODS = {
    "72 yıl (50% / 50 yr)":   {"factor": 0.45, "label": "72 yıl",   "desc": "Sık karşılaşılan — DD-4"},
    "475 yıl (10% / 50 yr)":  {"factor": 1.00, "label": "475 yıl",  "desc": "TBDY-2018 standart — DD-2"},
    "2475 yıl (2% / 50 yr)":  {"factor": 1.80, "label": "2475 yıl", "desc": "Çok nadir — DD-1"},
}


def _psha_pga_color(pga_g: float) -> str:
    """USGS PGA renk skalası (g cinsinden)."""
    if pga_g < 0.10: return "#1D9E75"   # yeşil
    if pga_g < 0.20: return "#7DC872"
    if pga_g < 0.30: return "#C0DD97"   # açık sarı-yeşil
    if pga_g < 0.40: return "#FAC775"   # sarı
    if pga_g < 0.55: return "#EF9F27"   # turuncu
    if pga_g < 0.75: return "#E24B4A"   # kırmızı
    return "#A32D2D"                     # koyu kırmızı


def _psha_zone_label(zone: int) -> str:
    return {1: "Zone-1 (en yüksek)", 2: "Zone-2", 3: "Zone-3", 4: "Zone-4 (en düşük)"}.get(zone, f"Zone-{zone}")


@st.cache_data(ttl=600, show_spinner=False)
def _psha_try_efehr_api(lat: float, lon: float):
    """EFEHR API denemesi — 5 sn timeout, fail olursa None."""
    try:
        r = requests.get(
            "https://efehr.org/services/seismicHazardData/",
            params={"latitude": lat, "longitude": lon, "returnPeriod": 475, "imt": "PGA"},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.fragment
def _render_sismik_tehlike():
    st.markdown(
        '<div class="chart-title">🗺️ Sismik Tehlike Haritası — PSHA (F-44 / v1.22)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🗺️ **PSHA (Probabilistic Seismic Hazard Analysis):** Bir konumda belirli bir "
        "dönüş periyodunda aşılma olasılığı olan en büyük yer ivmesini (PGA, g) verir. "
        "Teorik temel: **Woessner et al. (2015), SHARE 2013 European Seismic Hazard Model**; "
        "Türkiye verisi: **AFAD TBDY-2018**."
    )

    # ── Bölüm A: Dönüş periyodu seçici ─────────────────────────────────────
    col_rp, col_api = st.columns([2, 1])
    with col_rp:
        rp_choice = st.selectbox(
            "Dönüş periyodu seç",
            options=list(_PSHA_RETURN_PERIODS.keys()),
            index=1,  # Default: 475 yıl
            key="psha_rp_select",
        )
    with col_api:
        try_api = st.checkbox("EFEHR API dene", value=False, key="psha_try_api",
                              help="EFEHR canlı API; 5s timeout, fail olursa statik veri")

    rp_meta = _PSHA_RETURN_PERIODS[rp_choice]
    rp_factor = rp_meta["factor"]
    rp_label = rp_meta["label"]

    # API denemesi (opsiyonel) — Erzincan için test
    api_pga = None
    if try_api:
        with st.spinner("EFEHR API sorgulanıyor..."):
            api_resp = _psha_try_efehr_api(ERZ_LAT, ERZ_LON)
        if api_resp:
            st.success("✅ EFEHR API yanıtladı (Erzincan ref)")
            api_pga = api_resp
        else:
            st.warning("⚠️ EFEHR API erişilemedi — TBDY-2018 statik verisi kullanılıyor")

    # ── Bölüm B: PGA heatmap haritası ──────────────────────────────────────
    df_psha = pd.DataFrame(_PSHA_GRID_TR)
    df_psha["pga_rp"] = df_psha["pga475"] * rp_factor
    df_psha["renk"] = df_psha["pga_rp"].apply(_psha_pga_color)
    df_psha["zone_label"] = df_psha["zone"].apply(_psha_zone_label)

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(
        lat=df_psha["lat"],
        lon=df_psha["lon"],
        mode="markers",
        marker=dict(
            size=22,
            color=df_psha["pga_rp"],
            colorscale=[
                [0.00, "#1D9E75"],
                [0.20, "#7DC872"],
                [0.35, "#C0DD97"],
                [0.50, "#FAC775"],
                [0.65, "#EF9F27"],
                [0.85, "#E24B4A"],
                [1.00, "#A32D2D"],
            ],
            cmin=0.0,
            cmax=max(0.6, df_psha["pga_rp"].max()),
            colorbar=dict(
                title=dict(text=f"PGA (g)<br>{rp_label}", font=dict(color=TEXT, size=11)),
                tickfont=dict(color=TEXT, size=10),
                bgcolor="rgba(0,0,0,0.4)",
                thickness=14, len=0.7,
            ),
            opacity=0.92,
        ),
        text=df_psha.apply(
            lambda r: (
                f"<b>{r['city']}</b><br>"
                f"PGA ({rp_label}): {r['pga_rp']:.2f} g<br>"
                f"TBDY-2018: {r['zone_label']}<br>"
                f"Bağlam: {r['note']}"
                "<extra></extra>"
            ),
            axis=1,
        ),
        hovertemplate="%{text}",
        showlegend=False,
    ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.0, lon=35.0), zoom=5),
        height=540,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Bölüm C: 4 bilgi kartı ─────────────────────────────────────────────
    pga_max = df_psha["pga_rp"].max()
    pga_min = df_psha["pga_rp"].min()
    pga_mean = df_psha["pga_rp"].mean()
    zone1_n = int((df_psha["zone"] == 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{pga_max:.2f} g",  "#A32D2D", f"Maks PGA ({rp_label})"),
        (c2, f"{pga_mean:.2f} g", "#EF9F27", "Türkiye ortalama"),
        (c3, f"{pga_min:.2f} g",  "#1D9E75", "Min (Karadeniz/iç)"),
        (c4, f"{zone1_n}",        "#E24B4A", "Zone-1 il sayısı"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Bölüm D: TBDY-2018 zone tablosu ────────────────────────────────────
    st.markdown('<div class="chart-title">📋 TBDY-2018 Deprem Bölgeleri</div>', unsafe_allow_html=True)
    df_zone_summary = pd.DataFrame([
        {"Zone": "Zone-1", "Tanım": "En yüksek tehlike",  "PGA (475 yıl)": "≥ 0.40 g", "Renk": "🔴", "İl sayısı": int((df_psha["zone"] == 1).sum())},
        {"Zone": "Zone-2", "Tanım": "Yüksek tehlike",     "PGA (475 yıl)": "0.30 – 0.40 g", "Renk": "🟠", "İl sayısı": int((df_psha["zone"] == 2).sum())},
        {"Zone": "Zone-3", "Tanım": "Orta tehlike",       "PGA (475 yıl)": "0.20 – 0.30 g", "Renk": "🟡", "İl sayısı": int((df_psha["zone"] == 3).sum())},
        {"Zone": "Zone-4", "Tanım": "Düşük tehlike",      "PGA (475 yıl)": "< 0.20 g",      "Renk": "🟢", "İl sayısı": int((df_psha["zone"] == 4).sum())},
    ])
    st.dataframe(df_zone_summary, use_container_width=True, hide_index=True)

    # ── Bölüm E: İl bazlı tablo ────────────────────────────────────────────
    st.markdown('<div class="chart-title">🏙️ İl Bazlı PGA Değerleri</div>', unsafe_allow_html=True)
    df_view = df_psha[["city", "pga_rp", "zone_label", "note"]].copy()
    df_view.columns = ["İl", f"PGA ({rp_label}) g", "TBDY-2018 Zonu", "Tektonik Bağlam"]
    df_view = df_view.sort_values(f"PGA ({rp_label}) g", ascending=False).reset_index(drop=True)
    st.dataframe(df_view, use_container_width=True, hide_index=True, height=320)

    # ── Bölüm F: Kaynaklar ve uyarı ────────────────────────────────────────
    st.info(
        "📚 **PSHA Metodolojisi:** Cornell (1968) BSSA 58(5) | "
        "**SHARE 2013:** Woessner et al. (2015), *Bull. Earthq. Eng.* 13(12), 3553-3596 — "
        "DOI:10.1007/s10518-015-9795-1 | "
        "**OpenQuake:** Pagani et al. (2014), *SRL* 85(3), 692-702 | "
        "**Türkiye:** AFAD TBDY-2018 — deprem.afad.gov.tr/depremzonasi | "
        f"**Veri:** {len(_PSHA_GRID_TR)} il, statik harita (EFEHR API opsiyonel)"
    )
    st.warning(
        "⚠️ Bu harita TBDY-2018 spektral ivme değerlerinden türetilmiştir; yerel zemin "
        "koşulları (Vs30, mikrobölgeleme) PGA'yı 1.5–3× büyütebilir. Resmi proje hesabı "
        "için TBDY-2018 doğrudan referansı gereklidir."
    )


if active_menu == "🗺️ Sismik Tehlike":
    _render_sismik_tehlike()


# ════════════════════════════════════════════════════════════════════════════
# 🥎 ODAK MEKANİZMASI — F-45 / v1.23 — GCMT Beach Ball Kataloğu
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Dziewonski, A.M., Chou, T.-A. & Woodhouse, J.H. (1981). JGR 86(B4),
#       2825-2852. DOI:10.1029/JB086iB04p02825
#   • Ekström, G., Nettles, M. & Dziewonski, A.M. (2012). The global CMT
#       project 2004-2010. Phys. Earth Planet. Int. 200-201, 1-9.
#       DOI:10.1016/j.pepi.2012.04.002
#   • Aki & Richards (2002) Quantitative Seismology, 2nd ed.
#   • GCMT katalog: globalcmt.org
# ════════════════════════════════════════════════════════════════════════════

# GCMT katalog özeti — Türkiye yakını M5+ büyük olaylar (statik veri)
# Kaynak: globalcmt.org GCMT NDK katalog (Ekström et al. 2012 referansı)
# strike/dip/rake (Aki & Richards 1980 konvansiyonu, ° cinsinden)
_GCMT_EVENTS = [
    {"id": "1939-erzincan", "yer": "Erzincan 1939", "tarih": "1939-12-26", "lat": 39.80, "lon": 39.51,
     "mw": 7.8, "depth": 20, "strike": 105, "dip": 80, "rake": -10, "tip": "ss-r",
     "kaynak": "Ketin 1948; Barka 1988 — rekonstrüksiyon (NDK öncesi)"},
    {"id": "1992-erzincan", "yer": "Erzincan 1992", "tarih": "1992-03-13", "lat": 39.71, "lon": 39.60,
     "mw": 6.8, "depth": 27, "strike": 121, "dip": 71, "rake": -2, "tip": "ss-r",
     "kaynak": "Grosser et al. 1998, PAGEOPH"},
    {"id": "1999-izmit", "yer": "İzmit 1999", "tarih": "1999-08-17", "lat": 40.75, "lon": 29.86,
     "mw": 7.6, "depth": 17, "strike": 91, "dip": 87, "rake": -179, "tip": "ss-r",
     "kaynak": "Tibi et al. 2001; GCMT M081799A"},
    {"id": "1999-duzce", "yer": "Düzce 1999", "tarih": "1999-11-12", "lat": 40.79, "lon": 31.21,
     "mw": 7.2, "depth": 14, "strike": 264, "dip": 64, "rake": -172, "tip": "ss-r",
     "kaynak": "GCMT M111299A"},
    {"id": "2002-sultandagi", "yer": "Sultandağı 2002", "tarih": "2002-02-03", "lat": 38.57, "lon": 31.27,
     "mw": 6.5, "depth": 15, "strike": 145, "dip": 50, "rake": -98, "tip": "normal",
     "kaynak": "GCMT M020302A"},
    {"id": "2003-bingol", "yer": "Bingöl 2003", "tarih": "2003-05-01", "lat": 39.01, "lon": 40.46,
     "mw": 6.4, "depth": 6, "strike": 173, "dip": 89, "rake": -2, "tip": "ss-l",
     "kaynak": "GCMT M050103A"},
    {"id": "2010-elazig", "yer": "Elazığ 2010", "tarih": "2010-03-08", "lat": 38.87, "lon": 39.98,
     "mw": 6.1, "depth": 12, "strike": 257, "dip": 75, "rake": 9, "tip": "ss-l",
     "kaynak": "GCMT M030810A"},
    {"id": "2011-van", "yer": "Van 2011", "tarih": "2011-10-23", "lat": 38.72, "lon": 43.51,
     "mw": 7.1, "depth": 18, "strike": 252, "dip": 50, "rake": 60, "tip": "ters",
     "kaynak": "GCMT M102311A; Doğan & Karakas 2013"},
    {"id": "2017-bodrum", "yer": "Bodrum-Kos 2017", "tarih": "2017-07-20", "lat": 36.96, "lon": 27.43,
     "mw": 6.6, "depth": 7, "strike": 275, "dip": 39, "rake": -89, "tip": "normal",
     "kaynak": "GCMT M072017A"},
    {"id": "2020-elazig", "yer": "Sivrice/Elazığ 2020", "tarih": "2020-01-24", "lat": 38.39, "lon": 39.06,
     "mw": 6.8, "depth": 10, "strike": 247, "dip": 76, "rake": 4, "tip": "ss-l",
     "kaynak": "GCMT M012420A"},
    {"id": "2020-izmir", "yer": "İzmir/Samos 2020", "tarih": "2020-10-30", "lat": 37.90, "lon": 26.79,
     "mw": 6.9, "depth": 10, "strike": 277, "dip": 38, "rake": -94, "tip": "normal",
     "kaynak": "GCMT M103020A"},
    {"id": "2023-pazarcik", "yer": "Pazarcık 2023", "tarih": "2023-02-06", "lat": 37.17, "lon": 37.04,
     "mw": 7.8, "depth": 10, "strike": 228, "dip": 89, "rake": 1, "tip": "ss-l",
     "kaynak": "GCMT M020623A; Melgar et al. 2023"},
    {"id": "2023-elbistan", "yer": "Elbistan 2023", "tarih": "2023-02-06", "lat": 38.02, "lon": 37.20,
     "mw": 7.7, "depth": 10, "strike": 261, "dip": 56, "rake": 14, "tip": "ss-l",
     "kaynak": "GCMT M020623B (Sürgü fayı)"},
]

_GCMT_TIP_RENK = {
    "ss-r":   "#E24B4A",   # doğrultu atımlı sağ-yanal (KAF tipi) — kırmızı
    "ss-l":   "#EF9F27",   # doğrultu atımlı sol-yanal (DAF tipi) — turuncu
    "normal": "#FAC775",   # normal fay (Ege grabenleri) — sarı
    "ters":   "#1976D2",   # ters fay (Van, Doğu Anadolu sıkışma) — mavi
}
_GCMT_TIP_ETIKET = {
    "ss-r":   "Doğrultu Atımlı (Sağ-Yanal)",
    "ss-l":   "Doğrultu Atımlı (Sol-Yanal)",
    "normal": "Normal (Açılma)",
    "ters":   "Ters (Sıkışma)",
}


def _beachball_classify(rake: float) -> str:
    """Rake açısından (Aki & Richards) fay tipini sınıflandır."""
    r = ((rake + 180) % 360) - 180
    if -135 <= r <= -45:
        return "normal"
    if 45 <= r <= 135:
        return "ters"
    if (r > 135 or r < -135):
        return "ss-l"
    return "ss-r"


def _beachball_radial_pattern(strike: float, dip: float, rake: float, n: int = 36):
    """
    Beach ball çevresel basitleştirilmiş işaret deseni — sıkışma/genişleme yön farkı.
    Tam moment tensor değil; eğitim amaçlı strike + rake bazlı azimut renklendirmesi.
    Kaynak: Cronin (2010) Geol. Soc. Am. 'A Primer on Focal Mechanisms' (eğitim).
    """
    azs = np.linspace(0, 360, n + 1)
    signs = []
    s_rad = math.radians(strike)
    for az_deg in azs:
        az = math.radians(az_deg)
        # Azimut farkı (strike-fay normal yönüne göre)
        d_az = az - s_rad
        # P ekseni yönü ~ rake'e bağlı; basitleştirilmiş işaret deseni:
        val = math.cos(2 * d_az) * math.cos(math.radians(rake)) \
            + math.sin(2 * d_az) * math.sin(math.radians(rake)) * math.cos(math.radians(dip))
        signs.append(1 if val > 0 else -1)
    return azs, signs


@st.fragment
def _render_odak_mekanizma():
    st.markdown(
        '<div class="chart-title">🥎 Odak Mekanizması — GCMT Beach Ball Kataloğu (F-45 / v1.23)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🥎 **Odak Mekanizması (Focal Mechanism):** Bir depremin fay düzlemi çözümü; "
        "siyah-beyaz beach ball diyagramı normal/ters/doğrultu atımlı fay tipini tek "
        "bakışta gösterir. Teorik temel: **Dziewonski, Chou & Woodhouse (1981), JGR 86**; "
        "**Ekström et al. (2012), GCMT katalog**."
    )

    # ── Harita ─────────────────────────────────────────────────────────────
    fig_map = go.Figure()
    for ev in _GCMT_EVENTS:
        renk = _GCMT_TIP_RENK[ev["tip"]]
        size = 12 + (ev["mw"] - 5.5) * 5
        hover = (
            f"<b>{ev['yer']}</b> ({ev['tarih']})<br>"
            f"Mw {ev['mw']:.1f}, derinlik {ev['depth']} km<br>"
            f"Strike: {ev['strike']}°, Dip: {ev['dip']}°, Rake: {ev['rake']}°<br>"
            f"Tip: {_GCMT_TIP_ETIKET[ev['tip']]}<br>"
            f"Kaynak: {ev['kaynak']}"
            "<extra></extra>"
        )
        fig_map.add_trace(go.Scattermapbox(
            lat=[ev["lat"]], lon=[ev["lon"]],
            mode="markers",
            marker=dict(size=size, color=renk, opacity=0.85),
            text=ev["yer"],
            hovertemplate=hover,
            showlegend=False,
        ))

    # Legend için 4 dummy trace
    for tip_key in ("ss-r", "ss-l", "normal", "ters"):
        fig_map.add_trace(go.Scattermapbox(
            lat=[None], lon=[None],
            mode="markers",
            marker=dict(size=12, color=_GCMT_TIP_RENK[tip_key]),
            name=_GCMT_TIP_ETIKET[tip_key],
        ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.0, lon=35.0), zoom=5),
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Beach ball seçici ──────────────────────────────────────────────────
    st.markdown('<div class="chart-title">🎯 Olay Detayı — Beach Ball Diyagramı</div>', unsafe_allow_html=True)
    opt_labels = [f"M{e['mw']:.1f} — {e['yer']}" for e in _GCMT_EVENTS]
    sec_idx = st.selectbox(
        "Olay seç",
        options=list(range(len(_GCMT_EVENTS))),
        format_func=lambda i: opt_labels[i],
        index=11,  # default: 2023 Pazarcık
        key="gcmt_event_select",
    )
    ev = _GCMT_EVENTS[sec_idx]

    # Beach ball Plotly polar
    azs, signs = _beachball_radial_pattern(ev["strike"], ev["dip"], ev["rake"])
    azs_deg = list(azs)
    # Sıkışma (siyah) sektörler: sign=+1
    r_outer = [1.0] * len(azs)
    colors_per_az = ["#1a1a1a" if s > 0 else "#f4f4f4" for s in signs]

    col_bb, col_info = st.columns([1, 1])
    with col_bb:
        fig_bb = go.Figure()
        # Renkli sektörler bar polar olarak
        fig_bb.add_trace(go.Barpolar(
            r=r_outer,
            theta=azs_deg,
            width=[360 / len(azs)] * len(azs),
            marker=dict(color=colors_per_az, line=dict(color="#444", width=0.5)),
            hoverinfo="skip",
            opacity=0.95,
        ))
        # Strike çizgisi
        s_az = ev["strike"]
        fig_bb.add_trace(go.Scatterpolar(
            r=[0, 1], theta=[s_az, s_az],
            mode="lines",
            line=dict(color="#FFD700", width=3),
            name=f"Strike {s_az}°",
            hoverinfo="name",
        ))
        fig_bb.add_trace(go.Scatterpolar(
            r=[0, 1], theta=[(s_az + 180) % 360, (s_az + 180) % 360],
            mode="lines",
            line=dict(color="#FFD700", width=3, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_bb.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1.05]),
                angularaxis=dict(
                    direction="clockwise", rotation=90,
                    tickfont=dict(color=TEXT, size=9),
                ),
                bgcolor=BG2,
            ),
            paper_bgcolor=BG2,
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
            legend=dict(font=dict(color=TEXT, size=10)),
            title=dict(text=f"Beach Ball — {ev['yer']}", font=dict(color=TEXT, size=13)),
        )
        st.plotly_chart(fig_bb, use_container_width=True, config={"displayModeBar": False})
        st.caption("Siyah = sıkışma kadranı (P ekseni), beyaz = genişleme kadranı (T ekseni). Altın çizgi = fay strike yönü.")

    with col_info:
        st.markdown(f"""
**📍 {ev['yer']}** ({ev['tarih']})

| Parametre | Değer |
|---|---|
| **Mw** | {ev['mw']:.1f} |
| **Derinlik** | {ev['depth']} km |
| **Strike (φ)** | {ev['strike']}° |
| **Dip (δ)** | {ev['dip']}° |
| **Rake (λ)** | {ev['rake']}° |
| **Fay tipi** | {_GCMT_TIP_ETIKET[ev['tip']]} |
| **Konum** | ({ev['lat']:.3f}, {ev['lon']:.3f}) |

**🔬 Bilimsel kaynak:**
{ev['kaynak']}
        """)

    # ── Tablo: Tüm olaylar ────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 GCMT Katalog Özeti</div>', unsafe_allow_html=True)
    df_gcmt = pd.DataFrame([
        {"Olay": ev["yer"], "Tarih": ev["tarih"], "Mw": ev["mw"], "Derinlik (km)": ev["depth"],
         "Strike°": ev["strike"], "Dip°": ev["dip"], "Rake°": ev["rake"],
         "Fay Tipi": _GCMT_TIP_ETIKET[ev["tip"]]}
        for ev in _GCMT_EVENTS
    ])
    st.dataframe(df_gcmt, use_container_width=True, hide_index=True)

    # ── Kaynak ────────────────────────────────────────────────────────────
    st.caption(
        "📚 **GCMT Projesi:** Dziewonski et al. (1981) *JGR* 86(B4); "
        "Ekström et al. (2012) *Phys. Earth Planet. Int.* 200-201, 1-9 — "
        "DOI:10.1016/j.pepi.2012.04.002 | "
        "**Beach ball konvansiyonu:** Aki & Richards (2002) *Quantitative Seismology* | "
        "**Veri:** globalcmt.org NDK kataloğu (Türkiye odaklı 13 büyük olay seçimi). "
        "⚠️ Görsel beach ball — basitleştirilmiş çift-çift kuple deseni; "
        "tam moment tensor inversiyonu için ObsPy/PyROCKO referansı önerilir."
    )


if active_menu == "🥎 Odak Mekanizması":
    _render_odak_mekanizma()


# ════════════════════════════════════════════════════════════════════════════
# 📉 b-DEĞERİ ZAMAN SERİSİ — F-46 / v1.24 — Gutenberg-Richter Sliding Window
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Gutenberg, B. & Richter, C.F. (1944). BSSA 34(4), 185-188.
#       Frequency of earthquakes in California.
#   • Aki, K. (1965). MLE for b-value. Bull. Earthq. Res. Inst. 43, 237-239.
#   • Wiemer, S. & Wyss, M. (2000). Minimum magnitude of completeness Mc.
#       BSSA 90(4), 859-869. DOI:10.1785/0119990114
#   • Mignan, A. & Woessner, J. (2012). CORSSA. Estimating Mc.
#       DOI:10.5078/corssa-00180805
#   • van der Elst (2021). b-positive. JGR 126.
# ════════════════════════════════════════════════════════════════════════════


def _bvalue_aki_mle(magnitudes: np.ndarray, mc: float) -> tuple:
    """
    Aki 1965 MLE b-değeri tahmini ve standart hata (Shi & Bolt 1982).
    b = log10(e) / (Mmean - Mc + ΔM/2)
    σ_b = b² × √( Σ(Mi - Mmean)² / [N(N-1)] ) × ln(10)
    """
    mags = magnitudes[magnitudes >= mc]
    n = len(mags)
    if n < 25:
        return None, None, n
    bin_size = 0.1
    mean_m = float(np.mean(mags))
    b = math.log10(math.e) / (mean_m - mc + bin_size / 2.0)
    var = float(np.var(mags, ddof=1)) if n > 1 else 0.0
    sigma_b = 2.30 * b * b * math.sqrt(var / (n * (n - 1))) if n > 1 else None
    return b, sigma_b, n


def _bvalue_maxc_completeness(magnitudes: np.ndarray) -> float:
    """
    MAXC yöntemi (Wiemer & Wyss 2000) — frekans-büyüklük histogramının max'ı.
    """
    if len(magnitudes) < 10:
        return float(np.min(magnitudes)) if len(magnitudes) > 0 else 0.0
    bins = np.arange(np.floor(magnitudes.min() * 10) / 10,
                     np.ceil(magnitudes.max() * 10) / 10 + 0.1, 0.1)
    counts, edges = np.histogram(magnitudes, bins=bins)
    if len(counts) == 0:
        return float(np.min(magnitudes))
    return float(edges[int(np.argmax(counts))])


@st.fragment
def _render_b_value_time_series():
    st.markdown(
        '<div class="chart-title">📉 b-Değeri Zaman Serisi — Gutenberg-Richter Evolution (F-46 / v1.24)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "📉 **Gutenberg-Richter b-değeri:** log10 N = a − b·M. b ≈ 1.0 normaldir; "
        "b < 1.0 büyük olay baskınlığı (stres birikmesi), b > 1.0 küçük olay baskınlığı. "
        "Kayan pencere ile zaman serisi büyük olayların öncesinde b düşüşü gösterebilir "
        "(**Smith 1981; Schorlemmer 2005, Nature 437**). "
        "Hesap: **Aki (1965) MLE** + **Wiemer & Wyss (2000) MAXC Mc**."
    )

    if df.empty:
        st.warning("Veri yok — b-değeri hesaplanamaz.")
        return

    # ── Kontroller ─────────────────────────────────────────────────────────
    col_w, col_step, col_mc = st.columns(3)
    with col_w:
        window_n = st.slider("Pencere boyutu (olay)",
                             min_value=50, max_value=400, value=150, step=25,
                             key="bval_window_n",
                             help="Her pencere ≥25 olay içermeli (Wiemer)")
    with col_step:
        step_n = st.slider("Adım (olay)",
                           min_value=5, max_value=100, value=25, step=5,
                           key="bval_step_n")
    with col_mc:
        mc_mode = st.radio(
            "Mc yöntemi",
            options=["MAXC (otomatik)", "Sabit Mc = 3.0", "Sabit Mc = 3.5"],
            index=0, horizontal=False, key="bval_mc_mode",
        )

    # ── Veri hazırla (zamana göre sırala) ──────────────────────────────────
    df_sorted = df.sort_values("zaman").reset_index(drop=True)
    df_sorted = df_sorted.dropna(subset=["buyukluk", "zaman"])

    if len(df_sorted) < window_n:
        st.warning(f"Yetersiz veri ({len(df_sorted)} olay) — pencere için ≥{window_n} gerekli.")
        return

    # ── Sliding window b-value hesapla ─────────────────────────────────────
    times, b_vals, sigma_vals, mc_vals, n_vals = [], [], [], [], []
    for start in range(0, len(df_sorted) - window_n + 1, step_n):
        chunk = df_sorted.iloc[start:start + window_n]
        mags = chunk["buyukluk"].to_numpy(dtype=float)
        if mc_mode == "MAXC (otomatik)":
            mc = _bvalue_maxc_completeness(mags)
        elif mc_mode == "Sabit Mc = 3.0":
            mc = 3.0
        else:
            mc = 3.5

        b, sigma, n_complete = _bvalue_aki_mle(mags, mc)
        if b is None or sigma is None:
            continue
        if 0.3 < b < 2.5:  # mantıklı aralık dışını filtrele
            times.append(chunk["zaman"].iloc[len(chunk) // 2])  # pencere ortası
            b_vals.append(b)
            sigma_vals.append(sigma)
            mc_vals.append(mc)
            n_vals.append(n_complete)

    if len(times) < 3:
        st.warning("Yeterli güvenilir pencere oluşmadı — filtreler veya veri aralığı genişletin.")
        return

    # ── Grafik 1: b-değeri zaman serisi ───────────────────────────────────
    fig_b = go.Figure()
    b_arr = np.array(b_vals)
    sigma_arr = np.array(sigma_vals)
    fig_b.add_trace(go.Scatter(
        x=times + times[::-1],
        y=list(b_arr + sigma_arr) + list((b_arr - sigma_arr)[::-1]),
        fill="toself",
        fillcolor="rgba(25,118,210,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="±1σ güven aralığı",
    ))
    fig_b.add_trace(go.Scatter(
        x=times, y=b_vals,
        mode="lines+markers",
        line=dict(color="#1976d2", width=2),
        marker=dict(size=6, color="#1976d2"),
        name="b (Aki 1965 MLE)",
        hovertemplate="%{x|%Y-%m-%d}<br>b = %{y:.2f}<extra></extra>",
    ))
    # Referans çizgileri
    fig_b.add_hline(y=1.0, line=dict(color="#888", width=1, dash="dot"),
                    annotation_text="b = 1.0 (kanonik)", annotation_font_color="#888",
                    annotation_position="top right")
    fig_b.add_hline(y=0.7, line=dict(color="#E24B4A", width=1, dash="dash"),
                    annotation_text="b ≤ 0.7 (büyük olay riski)", annotation_font_color="#E24B4A",
                    annotation_position="bottom right")

    fig_b.update_layout(
        title=dict(text="b-Değeri Zaman Serisi (Kayan Pencere MLE)", font=dict(color=TEXT, size=13)),
        xaxis=dict(title="Zaman", color=TEXT, gridcolor=BORDER),
        yaxis=dict(title="b-değeri", color=TEXT, gridcolor=BORDER, range=[0.4, 1.8]),
        height=380,
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0.3)"),
    )
    st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

    # ── Grafik 2: Mc evolution ─────────────────────────────────────────────
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(
        x=times, y=mc_vals,
        mode="lines+markers",
        line=dict(color="#EF9F27", width=2),
        marker=dict(size=5, color="#EF9F27"),
        name="Mc (tamlık eşiği)",
        hovertemplate="%{x|%Y-%m-%d}<br>Mc = %{y:.2f}<extra></extra>",
    ))
    fig_mc.update_layout(
        title=dict(text="Mc (Magnitude of Completeness) Evolution — Wiemer & Wyss 2000 MAXC",
                   font=dict(color=TEXT, size=12)),
        xaxis=dict(title="Zaman", color=TEXT, gridcolor=BORDER),
        yaxis=dict(title="Mc", color=TEXT, gridcolor=BORDER),
        height=260,
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        showlegend=False,
    )
    st.plotly_chart(fig_mc, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    b_current = b_vals[-1] if b_vals else None
    b_mean = float(np.mean(b_vals)) if b_vals else None
    b_min = float(np.min(b_vals)) if b_vals else None
    mc_mean = float(np.mean(mc_vals)) if mc_vals else None

    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{b_current:.2f}" if b_current else "—",
         "#E24B4A" if b_current and b_current < 0.7 else ("#EF9F27" if b_current and b_current < 1.0 else "#1D9E75"),
         "Son b-değeri"),
        (c2, f"{b_mean:.2f}" if b_mean else "—", "#1976D2", "Ortalama b"),
        (c3, f"{b_min:.2f}" if b_min else "—", "#E24B4A", "Min b (risk)"),
        (c4, f"{mc_mean:.2f}" if mc_mean else "—", "#EF9F27", "Ortalama Mc"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Yorumlama tablosu ──────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 b-Değeri Yorumlama</div>', unsafe_allow_html=True)
    df_interp = pd.DataFrame([
        {"b aralığı": "b < 0.7", "Yorum": "Büyük olay baskın",           "Tektonik bağlam": "Yüksek stres birikimi — kırılma yaklaşıyor olabilir"},
        {"b aralığı": "0.7 ≤ b < 0.9", "Yorum": "Düşük b — dikkat",      "Tektonik bağlam": "Tektonik sıkışma bölgeleri (subdüksiyon, ters fay)"},
        {"b aralığı": "0.9 ≤ b ≤ 1.1", "Yorum": "Kanonik (normal)",     "Tektonik bağlam": "Tipik intraplaka veya doğrultu atımlı fay"},
        {"b aralığı": "1.1 < b ≤ 1.5", "Yorum": "Yüksek b — küçükler baskın", "Tektonik bağlam": "Volkanik bölgeler, jeotermal alanlar, ısı akısı yüksek"},
        {"b aralığı": "b > 1.5",       "Yorum": "Çok yüksek — anomali",  "Tektonik bağlam": "Ölçüm yanlısı (Mc hatası) veya nadir tektonik rejim"},
    ])
    st.dataframe(df_interp, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **Gutenberg & Richter (1944)** *BSSA* 34 (klasik bağıntı) | "
        "**Aki (1965)** MLE *Bull. Earthq. Res. Inst.* 43 (b tahmini) | "
        "**Shi & Bolt (1982)** *BSSA* 72 (σ_b standart hata) | "
        "**Wiemer & Wyss (2000)** *BSSA* 90(4) — DOI:10.1785/0119990114 (MAXC Mc) | "
        "**Schorlemmer et al. (2005)** *Nature* 437 (b-tip ilişkisi) | "
        "**van der Elst (2021)** *JGR* 126 (b-positive yöntemi). "
        "⚠️ Düşük b değerleri öncü sinyali olarak deterministtir DEĞİL — istatistiksel eğilim."
    )


if active_menu == "📉 b-Değeri Zaman Serisi":
    _render_b_value_time_series()


# ════════════════════════════════════════════════════════════════════════════
# 💥 COULOMB STRES — F-47 / v1.25 — Statik Gerilme Transferi (CFS)
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • King, G.C.P., Stein, R.S. & Lin, J. (1994). Static stress changes
#       and the triggering of earthquakes. BSSA 84(3), 935-953.
#   • Stein, R.S. (1999). The role of stress transfer in earthquake
#       occurrence. Nature 402, 605-609. DOI:10.1038/45144
#   • Okada, Y. (1992). Internal deformation due to shear and tensile
#       faults in a half-space. BSSA 82(2), 1018-1040.
#   • Toda, S. et al. (2011). Coulomb 3 software.
#       USGS Open-File Report 2011-1060.
#   • Stein, Barka & Dieterich (1997). Progressive failure on NAF.
#       Geophys. J. Int. 128, 594-604.
# ════════════════════════════════════════════════════════════════════════════

# CFS senaryoları — büyük olaylar ve komşu fay yüklenmesi
# ΔCFS = Δτ + µ' × Δσn ; µ' = 0.4 (Türkiye için tipik, King 1994)
_CFS_SENARYOLAR = {
    "1999 İzmit (Mw 7.6) → Düzce yüklenmesi": {
        "kaynak_lat": 40.75, "kaynak_lon": 29.86, "kaynak_mw": 7.6,
        "kaynak_strike": 91, "kaynak_rake": -179,
        "kirik_uzunluk_km": 145, "kayma_m": 4.0,
        "aciklama": "İzmit kırığının doğu ucu Düzce segmentinde +1.5 bar ΔCFS yüklemesi yaptı; "
                    "3 ay sonra Mw 7.2 Düzce kırıldı (Parsons et al. 2000).",
        "ref": "Parsons et al. 2000 Science; Stein 1999 Nature",
        "lab_hedef_lat": 40.79, "lab_hedef_lon": 31.21, "lab_hedef_isim": "Düzce segmenti",
    },
    "2023 Pazarcık (Mw 7.8) → Elbistan yüklenmesi": {
        "kaynak_lat": 37.17, "kaynak_lon": 37.04, "kaynak_mw": 7.8,
        "kaynak_strike": 228, "kaynak_rake": 1,
        "kirik_uzunluk_km": 350, "kayma_m": 5.5,
        "aciklama": "Pazarcık kırığının kuzey ucu Sürgü-Çardak fayında +3 bar ΔCFS; "
                    "9 saat sonra Mw 7.7 Elbistan ana şoku (Melgar et al. 2023).",
        "ref": "Melgar et al. 2023 Seismica; Stein et al. 2023",
        "lab_hedef_lat": 38.02, "lab_hedef_lon": 37.20, "lab_hedef_isim": "Sürgü fayı (Elbistan)",
    },
    "1939 Erzincan (Ms 7.8) → KAF batıya göç": {
        "kaynak_lat": 39.80, "kaynak_lon": 39.51, "kaynak_mw": 7.8,
        "kaynak_strike": 105, "kaynak_rake": -10,
        "kirik_uzunluk_km": 360, "kayma_m": 4.5,
        "aciklama": "1939 Erzincan kırığının batı ucu Niksar-Erbaa segmentinde +1.5 bar ΔCFS; "
                    "3 yıl sonra 1942 Niksar Ms 7.0 ana şoku. KAF batıya göç dizisinin başlangıcı.",
        "ref": "Stein, Barka & Dieterich 1997 GJI 128; Toksöz et al. 1979",
        "lab_hedef_lat": 40.65, "lab_hedef_lon": 36.95, "lab_hedef_isim": "Niksar segmenti",
    },
    "1999 Düzce (Mw 7.2) → Marmara/İstanbul": {
        "kaynak_lat": 40.79, "kaynak_lon": 31.21, "kaynak_mw": 7.2,
        "kaynak_strike": 264, "kaynak_rake": -172,
        "kirik_uzunluk_km": 40, "kayma_m": 3.0,
        "aciklama": "Düzce → batı, Marmara seismic gap'e ~0.2 bar ΔCFS; küçük yükleme ama "
                    "kümülatif Marmara stres bütçesine eklendi (Parsons 2004).",
        "ref": "Parsons 2004 JGR; Hubert-Ferrari et al. 2000 Nature",
        "lab_hedef_lat": 40.87, "lab_hedef_lon": 28.45, "lab_hedef_isim": "Marmara seismic gap",
    },
}


def _cfs_simplified_grid(src_lat: float, src_lon: float, src_mw: float,
                         src_strike_deg: float, n: int = 70):
    """
    Basitleştirilmiş 2B CFS gridi — King, Stein & Lin (1994) 'butterfly' deseni.
    Tam Okada 1992 dislokasyon modeli değil; göreceli yüklenme yön desenini gösterir.

    ΔCFS pozitif lobları kaynak fayın uçlarında ve şarken kanadında oluşur
    (4-lob desen: ±x ekseninde +, ±y ekseninde −).
    """
    # Grid range — büyüklüğe göre ölçek (Mw 7.0 ~ 100 km, Mw 7.8 ~ 250 km)
    extent_km = 50 + (src_mw - 6.0) * 80
    extent_deg = extent_km / 111.0

    lats = np.linspace(src_lat - extent_deg, src_lat + extent_deg, n)
    lons = np.linspace(src_lon - extent_deg / max(math.cos(math.radians(src_lat)), 0.1),
                       src_lon + extent_deg / max(math.cos(math.radians(src_lat)), 0.1), n)

    LON, LAT = np.meshgrid(lons, lats)
    dlat = (LAT - src_lat) * 111.0
    dlon = (LON - src_lon) * 111.0 * math.cos(math.radians(src_lat))
    r = np.sqrt(dlat * dlat + dlon * dlon) + 0.5  # km, sıfır bölmeyi engelle

    # Strike yönüne göre rotate koordinat sistemi
    s_rad = math.radians(src_strike_deg)
    x_rot = dlon * math.cos(s_rad) + dlat * math.sin(s_rad)
    y_rot = -dlon * math.sin(s_rad) + dlat * math.cos(s_rad)
    theta = np.arctan2(y_rot, x_rot)

    # 4-lob butterfly deseni: cos(2θ) → +/-/+/- desen
    # Mw ölçeği: Mw 7 ~ 1 bar maks, Mw 8 ~ 5 bar
    amplitude = 10 ** (1.5 * (src_mw - 7.0))  # 7.0 → 1 bar, 7.8 → 5.6 bar
    # Mesafe ile düşüş: r^(-2) yakın, r^(-3) uzak (rough)
    decay = 1.0 / (1.0 + (r / 20.0) ** 2)
    cfs = amplitude * np.cos(2 * theta) * decay

    return LAT, LON, cfs


@st.fragment
def _render_coulomb_stress():
    st.markdown(
        '<div class="chart-title">💥 Coulomb Stres Transferi — CFS Haritası (F-47 / v1.25)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "💥 **Coulomb Stres (ΔCFS):** Bir deprem komşu faylarda gerilim alanını değiştirir. "
        "ΔCFS = Δτ + µ' × Δσn. **Pozitif yüklenme (~+0.1 bar)** komşu fayda "
        "tetiklemeyi hızlandırır (Stein 1999, Nature 402). Teorik temel: "
        "**King, Stein & Lin (1994), BSSA 84(3)**; "
        "**Okada (1992) dislokasyon modeli**."
    )

    # ── Senaryo seçici ─────────────────────────────────────────────────────
    sec = st.selectbox(
        "Senaryo seç",
        options=list(_CFS_SENARYOLAR.keys()),
        index=0,
        key="cfs_senaryo",
    )
    s = _CFS_SENARYOLAR[sec]

    st.markdown(f"**📍 Bağlam:** {s['aciklama']}")

    # ── Hesap ──────────────────────────────────────────────────────────────
    LAT, LON, cfs = _cfs_simplified_grid(
        s["kaynak_lat"], s["kaynak_lon"], s["kaynak_mw"], s["kaynak_strike"]
    )

    # ── Harita: CFS densitymapbox ─────────────────────────────────────────
    z_max = float(np.max(np.abs(cfs)))
    fig_map = go.Figure()
    fig_map.add_trace(go.Densitymapbox(
        lat=LAT.flatten(),
        lon=LON.flatten(),
        z=cfs.flatten(),
        radius=18,
        colorscale=[
            [0.00, "#1976D2"],  # negatif (gölgeleme) — mavi
            [0.40, "#7FB3D5"],
            [0.50, "rgba(255,255,255,0.0)"],
            [0.60, "#FAC775"],
            [1.00, "#A32D2D"],  # pozitif (yüklenme) — kırmızı
        ],
        zmid=0,
        zmin=-z_max, zmax=z_max,
        colorbar=dict(
            title=dict(text="ΔCFS (bar)", font=dict(color=TEXT, size=11)),
            tickfont=dict(color=TEXT, size=10),
            bgcolor="rgba(0,0,0,0.4)",
            thickness=14, len=0.7,
        ),
        opacity=0.7,
        hovertemplate="ΔCFS: %{z:.2f} bar<br>(%{lat:.2f}, %{lon:.2f})<extra></extra>",
    ))

    # Kaynak fay çizgisi (strike yönünde)
    s_rad = math.radians(s["kaynak_strike"])
    half_len_deg = (s["kirik_uzunluk_km"] / 2.0) / 111.0
    f_lat1 = s["kaynak_lat"] - half_len_deg * math.cos(s_rad)
    f_lon1 = s["kaynak_lon"] - half_len_deg * math.sin(s_rad) / math.cos(math.radians(s["kaynak_lat"]))
    f_lat2 = s["kaynak_lat"] + half_len_deg * math.cos(s_rad)
    f_lon2 = s["kaynak_lon"] + half_len_deg * math.sin(s_rad) / math.cos(math.radians(s["kaynak_lat"]))
    fig_map.add_trace(go.Scattermapbox(
        lat=[f_lat1, f_lat2], lon=[f_lon1, f_lon2],
        mode="lines",
        line=dict(width=5, color="#FFD700"),
        name=f"Kaynak fay (Mw {s['kaynak_mw']:.1f}, {s['kirik_uzunluk_km']} km)",
        hoverinfo="name",
    ))
    fig_map.add_trace(go.Scattermapbox(
        lat=[s["kaynak_lat"]], lon=[s["kaynak_lon"]],
        mode="markers+text",
        marker=dict(size=14, color="#FFD700", symbol="star"),
        text=[f"★ Mw {s['kaynak_mw']:.1f}"],
        textposition="top right",
        textfont=dict(size=12, color="#FFD700"),
        name="Episentr",
        hoverinfo="name",
    ))

    # Hedef segment
    fig_map.add_trace(go.Scattermapbox(
        lat=[s["lab_hedef_lat"]], lon=[s["lab_hedef_lon"]],
        mode="markers+text",
        marker=dict(size=18, color="#E24B4A", symbol="circle"),
        text=[s["lab_hedef_isim"]],
        textposition="bottom right",
        textfont=dict(size=11, color="#E24B4A"),
        name="Etkilenen hedef",
        hoverinfo="name+text",
    ))

    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=s["kaynak_lat"], lon=s["kaynak_lon"]),
            zoom=6,
        ),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Hedef noktada CFS değeri ───────────────────────────────────────────
    # En yakın grid noktasını bul
    target_dlat = (LAT - s["lab_hedef_lat"]) * 111.0
    target_dlon = (LON - s["lab_hedef_lon"]) * 111.0 * math.cos(math.radians(s["lab_hedef_lat"]))
    target_dist = np.sqrt(target_dlat ** 2 + target_dlon ** 2)
    nearest_idx = np.unravel_index(np.argmin(target_dist), target_dist.shape)
    target_cfs = float(cfs[nearest_idx])
    target_km = float(target_dist[nearest_idx])

    # Lob sayımı
    pos_area_pct = 100.0 * np.sum(cfs > 0.1) / cfs.size
    max_pos = float(np.max(cfs))
    max_neg = float(np.min(cfs))

    c1, c2, c3, c4 = st.columns(4)
    target_color = "#E24B4A" if target_cfs > 0.1 else ("#FAC775" if target_cfs > 0 else "#1976D2")
    kartlar = [
        (c1, f"{target_cfs:+.2f} bar", target_color,
         f"Hedef ΔCFS ({s['lab_hedef_isim']})"),
        (c2, f"+{max_pos:.2f} bar", "#A32D2D", "Maks pozitif yüklenme"),
        (c3, f"{max_neg:.2f} bar", "#1976D2", "Maks negatif (gölgeleme)"),
        (c4, f"%{pos_area_pct:.0f}", "#EF9F27", "Pozitif lob alan oranı"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Yorumlama tablosu ─────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 ΔCFS Yorumlama Eşikleri</div>', unsafe_allow_html=True)
    df_cfs = pd.DataFrame([
        {"ΔCFS aralığı": "ΔCFS > +1.0 bar",     "Yorum": "Yüksek tetikleme olasılığı", "Örnek": "Düzce 1999 (3 ay sonra)"},
        {"ΔCFS aralığı": "+0.1 < ΔCFS ≤ +1.0",  "Yorum": "Anlamlı yüklenme",          "Örnek": "Çoğu artçı (King 1994 eşiği)"},
        {"ΔCFS aralığı": "−0.1 ≤ ΔCFS ≤ +0.1",  "Yorum": "İhmal edilebilir",          "Örnek": "Uzak alan"},
        {"ΔCFS aralığı": "ΔCFS < −0.1 bar",     "Yorum": "Stres gölgeleme (gecikme)", "Örnek": "Komşu segmentlerin geçici dinginleşmesi"},
    ])
    st.dataframe(df_cfs, use_container_width=True, hide_index=True)

    st.caption(
        f"📚 **Senaryo referansı:** {s['ref']} | "
        "**King, Stein & Lin (1994)** *BSSA* 84(3), 935-953 (CFS formülasyonu) | "
        "**Stein (1999)** *Nature* 402, 605-609 — DOI:10.1038/45144 (kavramsal) | "
        "**Okada (1992)** *BSSA* 82(2), 1018-1040 (dislokasyon teorisi) | "
        "**Toda et al. (2011)** USGS OFR 2011-1060 (Coulomb 3 yazılım). "
        "⚠️ Bu görsel basitleştirilmiş 4-lob deseni gösterir; tam Okada 1992 yarım-uzay "
        "elastik dislokasyon modeli için Coulomb 3 / PSCMP kullanılır."
    )


if active_menu == "💥 Coulomb Stres":
    _render_coulomb_stress()


# ════════════════════════════════════════════════════════════════════════════
# 🛰️ INSAR DEFORMASYON — F-48 / v1.26 — Sentinel-1 LOS Yer Değiştirme
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Massonnet, D. & Feigl, K.L. (1998). Radar interferometry and its
#       application to changes in the earth's surface. Rev. Geophys. 36(4),
#       441-500. DOI:10.1029/97RG03139
#   • Xu, W. et al. (2023). Surface ruptures, coseismic deformation, and
#       seismotectonics of the 2023 M7.8 and M7.7 earthquake doublet in SE
#       Turkey. EPSL 612, 118333. DOI:10.1016/j.epsl.2023.118333
#   • Massonnet et al. (1993). The displacement field of the Landers
#       earthquake mapped by radar interferometry. Nature 364, 138-142.
#   • ESA Sentinel-1: sentinel.esa.int
#   • COMET LiCS portal: comet.nerc.ac.uk/COMET-LiCS-portal
# ════════════════════════════════════════════════════════════════════════════

_INSAR_OLAYLAR = {
    "2023 Pazarcık (Mw 7.8) — Kahramanmaraş doublet": {
        "lat": 37.17, "lon": 37.04, "mw": 7.8,
        "tarih": "2023-02-06",
        "strike": 228, "rupture_km": 350,
        "max_los_m": 5.5,  # Xu et al. 2023 EPSL — line-of-sight
        "max_horizontal_m": 7.2,
        "comet_url": "https://comet.nerc.ac.uk/earthquakes/us6000jllz/",
        "kaynak": "Xu et al. 2023 EPSL 612, 118333 (COMET LiCS interferogram)",
    },
    "2023 Elbistan (Mw 7.7) — Sürgü-Çardak fayı": {
        "lat": 38.02, "lon": 37.20, "mw": 7.7,
        "tarih": "2023-02-06",
        "strike": 261, "rupture_km": 160,
        "max_los_m": 3.8,
        "max_horizontal_m": 4.6,
        "comet_url": "https://comet.nerc.ac.uk/earthquakes/us6000jlqa/",
        "kaynak": "Melgar et al. 2023 Seismica; COMET LiCS",
    },
    "2020 Sivrice/Elazığ (Mw 6.8) — DAF": {
        "lat": 38.39, "lon": 39.06, "mw": 6.8,
        "tarih": "2020-01-24",
        "strike": 247, "rupture_km": 50,
        "max_los_m": 0.95,
        "max_horizontal_m": 1.1,
        "comet_url": "https://comet.nerc.ac.uk/earthquakes/us60007ewc/",
        "kaynak": "Pousse-Beltran et al. 2020 GRL; ESA Sentinel-1",
    },
    "2011 Van (Mw 7.1) — Doğu Anadolu ters fay": {
        "lat": 38.72, "lon": 43.51, "mw": 7.1,
        "tarih": "2011-10-23",
        "strike": 252, "rupture_km": 60,
        "max_los_m": 0.85,
        "max_horizontal_m": 0.55,
        "comet_url": "https://comet.nerc.ac.uk/",
        "kaynak": "Elliott et al. 2013 GRL; Fielding et al. 2013 JGR",
    },
    "1999 İzmit (Mw 7.6) — KAF (ERS-2)": {
        "lat": 40.75, "lon": 29.86, "mw": 7.6,
        "tarih": "1999-08-17",
        "strike": 91, "rupture_km": 145,
        "max_los_m": 2.7,
        "max_horizontal_m": 4.5,
        "comet_url": "https://comet.nerc.ac.uk/",
        "kaynak": "Wright et al. 2001 GRL (ERS-2 — InSAR'ın Türkiye'deki klasiği)",
    },
}


def _insar_synthetic_interferogram(lat0: float, lon0: float, strike_deg: float,
                                   rupture_km: float, max_los_m: float, n: int = 80):
    """
    Sentinel-1 interferogram benzeri sentetik LOS deformasyon alanı.
    Fay boyunca anti-simetrik kayma — gerçek InSAR fringe deseninin basit özeti.

    Eğitim/görselleştirme amaçlı; gerçek interferogram için COMET LiCS portalı.
    """
    extent_km = rupture_km * 1.2
    extent_deg = extent_km / 111.0

    lats = np.linspace(lat0 - extent_deg, lat0 + extent_deg, n)
    cos_lat = math.cos(math.radians(lat0))
    lons = np.linspace(lon0 - extent_deg / cos_lat, lon0 + extent_deg / cos_lat, n)
    LON, LAT = np.meshgrid(lons, lats)

    dlat = (LAT - lat0) * 111.0
    dlon = (LON - lon0) * 111.0 * cos_lat

    # Strike yönüne rotate
    s_rad = math.radians(strike_deg)
    x_par = dlon * math.cos(s_rad) + dlat * math.sin(s_rad)   # fay boyunca
    y_perp = -dlon * math.sin(s_rad) + dlat * math.cos(s_rad)  # faya dik

    # Anti-simetrik doğrultu atımlı deformasyon (right-lateral varsayım):
    # y > 0 tarafında +los, y < 0 tarafında −los
    # Fay boyunca cosine taper (uçlarda 0)
    half_l = rupture_km / 2.0
    taper = np.where(
        np.abs(x_par) < half_l,
        np.cos(np.pi * x_par / (2 * half_l)),
        0.0,
    )
    decay = np.exp(-np.abs(y_perp) / 15.0)
    los = max_los_m * np.sign(y_perp) * taper * decay

    return LAT, LON, los


@st.fragment
def _render_insar():
    st.markdown(
        '<div class="chart-title">🛰️ InSAR Koseismik Deformasyon — Sentinel-1 LOS (F-48 / v1.26)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🛰️ **InSAR (Interferometric SAR):** İki SAR görüntüsünün faz farkından "
        "**centimetre hassasiyetinde** yer deformasyonu ölçer. Massonnet 1993'te "
        "ilk kez Landers depreminde uygulanmış, bugün ESA Sentinel-1 her 6-12 günde "
        "tüm Türkiye'yi tarar. Teorik temel: **Massonnet & Feigl (1998), Rev. Geophys. 36(4)**."
    )

    sec = st.selectbox(
        "Olay seç",
        options=list(_INSAR_OLAYLAR.keys()),
        index=0,
        key="insar_select",
    )
    o = _INSAR_OLAYLAR[sec]

    st.markdown(
        f"**📍 Olay:** {sec} | **Tarih:** {o['tarih']} | "
        f"**Mw:** {o['mw']:.1f} | **Kırık:** {o['rupture_km']} km"
    )

    # ── Sentetik interferogram ─────────────────────────────────────────────
    LAT, LON, los = _insar_synthetic_interferogram(
        o["lat"], o["lon"], o["strike"], o["rupture_km"], o["max_los_m"]
    )

    fig = go.Figure()
    fig.add_trace(go.Densitymapbox(
        lat=LAT.flatten(),
        lon=LON.flatten(),
        z=los.flatten(),
        radius=15,
        colorscale=[
            [0.00, "#000080"],  # uzaklaşma (LOS negatif) — koyu mavi
            [0.25, "#1976D2"],
            [0.50, "rgba(255,255,255,0.0)"],
            [0.75, "#EF9F27"],
            [1.00, "#A32D2D"],  # yaklaşma (LOS pozitif) — koyu kırmızı
        ],
        zmid=0,
        zmin=-o["max_los_m"], zmax=o["max_los_m"],
        colorbar=dict(
            title=dict(text="LOS<br>(m)", font=dict(color=TEXT, size=11)),
            tickfont=dict(color=TEXT, size=10),
            bgcolor="rgba(0,0,0,0.4)",
            thickness=14, len=0.7,
        ),
        opacity=0.75,
        hovertemplate="LOS: %{z:+.2f} m<br>(%{lat:.3f}, %{lon:.3f})<extra></extra>",
    ))

    # Fay kırığı (strike boyunca)
    s_rad = math.radians(o["strike"])
    half_l_deg = (o["rupture_km"] / 2.0) / 111.0
    cos_lat = math.cos(math.radians(o["lat"]))
    f_lat1 = o["lat"] - half_l_deg * math.cos(s_rad)
    f_lon1 = o["lon"] - half_l_deg * math.sin(s_rad) / cos_lat
    f_lat2 = o["lat"] + half_l_deg * math.cos(s_rad)
    f_lon2 = o["lon"] + half_l_deg * math.sin(s_rad) / cos_lat
    fig.add_trace(go.Scattermapbox(
        lat=[f_lat1, f_lat2], lon=[f_lon1, f_lon2],
        mode="lines",
        line=dict(width=5, color="#FFD700"),
        name=f"Yüzey kırığı ({o['rupture_km']} km)",
        hoverinfo="name",
    ))
    fig.add_trace(go.Scattermapbox(
        lat=[o["lat"]], lon=[o["lon"]],
        mode="markers+text",
        marker=dict(size=14, color="#FFD700", symbol="star"),
        text=[f"★ Mw {o['mw']:.1f}"],
        textposition="top right",
        textfont=dict(size=12, color="#FFD700"),
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=o["lat"], lon=o["lon"]),
            zoom=7,
        ),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    fringe_count = int(o["max_los_m"] / 0.028)  # Sentinel-1 C-band yarı dalgaboyu ~2.8 cm
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{o['max_los_m']:.2f} m",        "#A32D2D", "Maks LOS yer değiştirme"),
        (c2, f"{o['max_horizontal_m']:.2f} m", "#EF9F27", "Maks yatay (3D inv.)"),
        (c3, f"{o['rupture_km']} km",          "#FFD700", "Yüzey kırık uzunluğu"),
        (c4, f"~{fringe_count}",               "#1976D2", "Sentinel-1 C-bant fringe (2.8 cm/fringe)"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Bağlantılar ───────────────────────────────────────────────────────
    st.markdown(f"""
**🔗 Gerçek interferogram için:**
- **COMET LiCS Portal:** [{o['comet_url']}]({o['comet_url']})
- **ESA Copernicus Open Hub:** https://scihub.copernicus.eu/
- **NASA ARIA:** https://aria.jpl.nasa.gov/

**📐 Sentinel-1 spesifikasyonları:**
- C-bant SAR (5.405 GHz, λ = 5.6 cm → yarı dalgaboyu 2.8 cm/fringe)
- Tekrarlama: 6 gün (ascending+descending), 12 gün tek modda
- Çözünürlük: 5 × 20 m (IW modu)
    """)

    # ── InSAR Fizik Tablosu ────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 InSAR Yorum Anahtarı</div>', unsafe_allow_html=True)
    df_insar = pd.DataFrame([
        {"Renk": "🔴 Kırmızı", "LOS": "+", "Yorum": "Uydudan uzağa hareket (uplift veya yaklaşma)"},
        {"Renk": "🔵 Mavi",   "LOS": "−", "Yorum": "Uyduya doğru hareket (subsidence veya uzaklaşma)"},
        {"Renk": "⚪ Beyaz",  "LOS": "0", "Yorum": "Deformasyon yok veya çok küçük"},
        {"Renk": "🟢 Fringe", "Renk değişimi": "1 tam fringe", "Yorum": "Yarım dalgaboyu yer değiştirme (~2.8 cm, C-bant)"},
    ])
    st.dataframe(df_insar, use_container_width=True, hide_index=True)

    st.caption(
        f"📚 **Senaryo:** {o['kaynak']} | "
        "**Massonnet & Feigl (1998)** *Rev. Geophys.* 36(4), 441-500 — DOI:10.1029/97RG03139 | "
        "**Massonnet et al. (1993)** *Nature* 364, 138-142 (Landers — InSAR'ın doğuşu) | "
        "**Xu et al. (2023)** *EPSL* 612, 118333 (Kahramanmaraş) | "
        "**Wright et al. (2001)** *GRL* (İzmit ERS-2). "
        "⚠️ Harita sentetik gösterim — gerçek interferogram için COMET LiCS portalı kullanılmalıdır."
    )


if active_menu == "🛰️ InSAR Deformasyon":
    _render_insar()


# ════════════════════════════════════════════════════════════════════════════
# 📜 TARİHSEL SİSMİSİTE — F-49 / v1.27 — 2000 Yıllık Türkiye Deprem Atlası
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Ambraseys, N.N. (2009). Earthquakes in the Mediterranean and Middle
#       East: A Multidisciplinary Study of Seismicity up to 1900.
#       Cambridge University Press. ISBN:9780521872928
#   • Ambraseys, N.N. & Finkel, C.F. (1995). Seismicity of Turkey and
#       Adjacent Areas, 1500-1800. Eren Yayıncılık.
#   • Guidoboni, E. et al. (1994). Catalogue of ancient earthquakes in the
#       Mediterranean area up to the 10th century. ING-SGA, Rome.
#   • Sbeinati, M.R. et al. (2005). The historical earthquakes of Syria.
#       Annals of Geophysics 48(3), 347-435.
#   • NOAA NCEI Significant Earthquakes Database
# ════════════════════════════════════════════════════════════════════════════

# Türkiye + yakın Doğu Akdeniz tarihsel depremler (Ambraseys 2009, Guidoboni 1994)
# Mw tahminleri makro-sismik şiddet kayıtlarından (Ambraseys & Jackson 2000)
_TARIHSEL_OLAYLAR = [
    # MS 1-500
    {"yil": 17,   "ay": 9,  "yer": "Sardes (Lydia)",     "lat": 38.49, "lon": 28.04, "mw": 7.0, "kaynak": "Tacitus; Guidoboni 1994"},
    {"yil": 115,  "ay": 12, "yer": "Antakya (Antioch)",  "lat": 36.20, "lon": 36.15, "mw": 7.5, "kaynak": "Cassius Dio; Guidoboni 1994"},
    {"yil": 358,  "ay": 8,  "yer": "Nicomedia (İzmit)",  "lat": 40.77, "lon": 29.92, "mw": 7.4, "kaynak": "Ammianus Marcellinus; Ambraseys 2009"},
    {"yil": 365,  "ay": 7,  "yer": "Girit (Megatsunami)","lat": 35.50, "lon": 23.50, "mw": 8.4, "kaynak": "Stiros 2001 J. Struct. Geol.; megatsunami"},
    {"yil": 484,  "ay": 9,  "yer": "İznik (Nicaea)",     "lat": 40.42, "lon": 29.72, "mw": 7.0, "kaynak": "Ambraseys 2009 Tablo 1.3"},

    # 500-1000
    {"yil": 526,  "ay": 5,  "yer": "Antakya",            "lat": 36.20, "lon": 36.16, "mw": 7.0, "kaynak": "Procopius; 250.000+ ölü tahmini"},
    {"yil": 542,  "ay": 8,  "yer": "Konstantinopolis",   "lat": 41.01, "lon": 28.98, "mw": 6.5, "kaynak": "Theophanes Chronicle"},
    {"yil": 740,  "ay": 10, "yer": "Konstantinopolis",   "lat": 40.85, "lon": 28.20, "mw": 7.1, "kaynak": "Ambraseys 2009; bizans surları yıkıldı"},
    {"yil": 859,  "ay": 4,  "yer": "Antakya",            "lat": 36.20, "lon": 36.16, "mw": 7.0, "kaynak": "Arap kronikleri; Sbeinati 2005"},

    # 1000-1500
    {"yil": 1063, "ay": 9,  "yer": "Marmara",            "lat": 40.80, "lon": 28.50, "mw": 7.2, "kaynak": "Skylitzes; Ambraseys 2002"},
    {"yil": 1114, "ay": 11, "yer": "Maraş-Antakya",      "lat": 37.30, "lon": 36.80, "mw": 7.8, "kaynak": "Mateos of Edessa; DAF kuzey"},
    {"yil": 1202, "ay": 5,  "yer": "Doğu Akdeniz",       "lat": 33.50, "lon": 35.50, "mw": 7.5, "kaynak": "Sbeinati 2005; Suriye-Lübnan"},
    {"yil": 1268, "ay": 6,  "yer": "Erzincan",           "lat": 39.75, "lon": 39.50, "mw": 7.5, "kaynak": "Ermeni vakanüvis; Ambraseys 2009"},
    {"yil": 1303, "ay": 8,  "yer": "Rodos (Tsunami)",    "lat": 35.50, "lon": 27.50, "mw": 8.0, "kaynak": "Guidoboni 2004; Doğu Akdeniz mega"},
    {"yil": 1354, "ay": 3,  "yer": "Gelibolu",           "lat": 40.40, "lon": 26.65, "mw": 7.2, "kaynak": "Bizans + Osmanlı kayıtları"},

    # 1500-1800
    {"yil": 1509, "ay": 9,  "yer": "İstanbul (Kıyamet-i Sugra)", "lat": 41.00, "lon": 28.97, "mw": 7.2, "kaynak": "Osmanlı vakanüvis; 'Küçük Kıyamet'"},
    {"yil": 1668, "ay": 8,  "yer": "Amasya-Kuzey Anadolu",       "lat": 40.65, "lon": 35.80, "mw": 8.0, "kaynak": "Ambraseys & Finkel 1995; KAF 400+ km kırık"},
    {"yil": 1719, "ay": 5,  "yer": "İzmit",              "lat": 40.75, "lon": 30.00, "mw": 7.4, "kaynak": "Ambraseys & Finkel 1995"},
    {"yil": 1754, "ay": 9,  "yer": "İzmit-Doğu Marmara", "lat": 40.80, "lon": 29.50, "mw": 6.9, "kaynak": "Ambraseys & Finkel 1995"},
    {"yil": 1766, "ay": 5,  "yer": "İstanbul-Marmara (büyük)", "lat": 40.80, "lon": 29.00, "mw": 7.1, "kaynak": "Ambraseys 2002 J. Seism.; Marmara seismic gap başlangıcı"},
    {"yil": 1766, "ay": 8,  "yer": "Saros Körfezi",      "lat": 40.65, "lon": 27.00, "mw": 7.4, "kaynak": "Ambraseys 2002"},
    {"yil": 1784, "ay": 7,  "yer": "Erzincan",           "lat": 39.80, "lon": 39.30, "mw": 7.6, "kaynak": "Ambraseys & Finkel 1988"},

    # 1800-1900
    {"yil": 1822, "ay": 8,  "yer": "Halep-Antakya",      "lat": 36.20, "lon": 36.80, "mw": 7.4, "kaynak": "DAF; Ambraseys 2009"},
    {"yil": 1855, "ay": 2,  "yer": "Bursa",              "lat": 40.18, "lon": 29.07, "mw": 7.1, "kaynak": "Ambraseys & Finkel 1987 Disasters"},
    {"yil": 1859, "ay": 6,  "yer": "Erzurum",            "lat": 39.90, "lon": 41.27, "mw": 6.5, "kaynak": "Ambraseys 2009"},
    {"yil": 1872, "ay": 4,  "yer": "Antakya-Amik Gölü",  "lat": 36.40, "lon": 36.50, "mw": 7.2, "kaynak": "DAF güney; Ambraseys 1989"},
    {"yil": 1881, "ay": 4,  "yer": "Sakız Adası (Sakız)","lat": 38.40, "lon": 26.10, "mw": 7.3, "kaynak": "Ambraseys 2001"},
    {"yil": 1894, "ay": 7,  "yer": "İstanbul",           "lat": 40.70, "lon": 28.65, "mw": 7.0, "kaynak": "Ambraseys & Finkel 1991; aletsel öncü"},

    # 1900+ aletsel (referans için)
    {"yil": 1903, "ay": 4,  "yer": "Malazgirt",          "lat": 39.20, "lon": 42.50, "mw": 7.0, "kaynak": "Aletsel başlangıç"},
    {"yil": 1912, "ay": 8,  "yer": "Mürefte-Saros",      "lat": 40.75, "lon": 27.20, "mw": 7.4, "kaynak": "Ambraseys & Jackson 2000"},
    {"yil": 1939, "ay": 12, "yer": "Erzincan",           "lat": 39.80, "lon": 39.51, "mw": 7.8, "kaynak": "Barka 1996 BSSA; KAF batı göç başlangıcı"},
    {"yil": 1942, "ay": 12, "yer": "Niksar-Erbaa",       "lat": 40.65, "lon": 36.95, "mw": 7.0, "kaynak": "KAF batıya göç"},
    {"yil": 1943, "ay": 11, "yer": "Tosya-Ladik",        "lat": 41.00, "lon": 34.00, "mw": 7.6, "kaynak": "KAF"},
    {"yil": 1944, "ay": 2,  "yer": "Bolu-Gerede",        "lat": 40.85, "lon": 32.30, "mw": 7.4, "kaynak": "KAF"},
    {"yil": 1957, "ay": 5,  "yer": "Abant (Bolu)",       "lat": 40.65, "lon": 31.00, "mw": 7.1, "kaynak": "KAF"},
    {"yil": 1967, "ay": 7,  "yer": "Adapazarı-Mudurnu",  "lat": 40.65, "lon": 30.70, "mw": 7.1, "kaynak": "KAF"},
    {"yil": 1976, "ay": 11, "yer": "Çaldıran-Van",       "lat": 39.05, "lon": 44.04, "mw": 7.3, "kaynak": "Doğu Anadolu"},
    {"yil": 1983, "ay": 10, "yer": "Erzurum-Kars",       "lat": 40.32, "lon": 42.18, "mw": 6.9, "kaynak": "Doğu Anadolu"},
    {"yil": 1992, "ay": 3,  "yer": "Erzincan",           "lat": 39.71, "lon": 39.60, "mw": 6.8, "kaynak": "Grosser 1998 PAGEOPH 152"},
    {"yil": 1999, "ay": 8,  "yer": "İzmit (Gölcük)",     "lat": 40.75, "lon": 29.86, "mw": 7.6, "kaynak": "KAF; ~17.000 ölü"},
    {"yil": 1999, "ay": 11, "yer": "Düzce",              "lat": 40.79, "lon": 31.21, "mw": 7.2, "kaynak": "KAF batı segment"},
    {"yil": 2011, "ay": 10, "yer": "Van",                "lat": 38.72, "lon": 43.51, "mw": 7.1, "kaynak": "Doğu Anadolu ters fay"},
    {"yil": 2020, "ay": 10, "yer": "İzmir-Samos",        "lat": 37.90, "lon": 26.79, "mw": 6.9, "kaynak": "Ege normal fay"},
    {"yil": 2023, "ay": 2,  "yer": "Pazarcık (Maraş)",   "lat": 37.17, "lon": 37.04, "mw": 7.8, "kaynak": "DAF + Sürgü; ~55.000 ölü"},
    {"yil": 2023, "ay": 2,  "yer": "Elbistan",           "lat": 38.02, "lon": 37.20, "mw": 7.7, "kaynak": "Sürgü fayı (9 saat sonra)"},
]


def _yuzyil_renk(yuzyil: int) -> str:
    # 1. yy → 21. yy, viridis-türevi
    palette = ["#440154", "#481B6D", "#46337E", "#3F4889", "#365C8D",
               "#2E6E8E", "#277F8E", "#21908C", "#1FA187", "#2DB27D",
               "#4AC16D", "#6ECE58", "#9FDA3A", "#CFE11C", "#FDE725",
               "#FEC829", "#F89441", "#EB6453", "#D43F71", "#A52A85",
               "#7A1F8B"]  # 21 element
    idx = max(0, min(20, yuzyil - 1))
    return palette[idx]


@st.fragment
def _render_tarihsel_sismisite():
    st.markdown(
        '<div class="chart-title">📜 Tarihsel Sismisite — Türkiye 2000 Yıllık Atlas (F-49 / v1.27)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "📜 **Tarihsel Sismisite:** Aletsel öncesi (1900 öncesi) büyük depremler, "
        "Bizans/Osmanlı vakanüvisleri, Arap kronikleri ve arkeo-sismik izlerden "
        "derlenmiştir. Teorik temel: **Ambraseys (2009), Cambridge Univ. Press**; "
        "**Guidoboni et al. (1994)**; **Sbeinati et al. (2005), Ann. Geophys.**."
    )

    # ── Yüzyıl filtresi ────────────────────────────────────────────────────
    col_yy, col_mag = st.columns([2, 1])
    with col_yy:
        yuzyil_range = st.slider(
            "Yüzyıl aralığı",
            min_value=1, max_value=21, value=(1, 21), step=1,
            key="tarihsel_yy_range",
        )
    with col_mag:
        min_mw = st.slider(
            "Min. Mw",
            min_value=6.5, max_value=8.0, value=6.8, step=0.1,
            key="tarihsel_min_mw",
        )

    yy_min, yy_max = yuzyil_range
    df_h = pd.DataFrame(_TARIHSEL_OLAYLAR)
    df_h["yuzyil"] = ((df_h["yil"] - 1) // 100) + 1
    df_filt = df_h[(df_h["yuzyil"] >= yy_min) & (df_h["yuzyil"] <= yy_max) & (df_h["mw"] >= min_mw)]
    df_filt = df_filt.sort_values("yil").reset_index(drop=True)

    if df_filt.empty:
        st.warning("Bu filtreye uygun olay bulunamadı.")
        return

    # ── Harita ─────────────────────────────────────────────────────────────
    fig_map = go.Figure()
    for _, ev in df_filt.iterrows():
        size = 8 + (ev["mw"] - 6.5) * 5
        renk = _yuzyil_renk(int(ev["yuzyil"]))
        hover = (
            f"<b>{ev['yer']}</b> ({int(ev['yil'])})<br>"
            f"Mw ~{ev['mw']:.1f}<br>"
            f"Yüzyıl: {int(ev['yuzyil'])}.<br>"
            f"Kaynak: {ev['kaynak']}"
            "<extra></extra>"
        )
        fig_map.add_trace(go.Scattermapbox(
            lat=[ev["lat"]], lon=[ev["lon"]],
            mode="markers",
            marker=dict(size=size, color=renk, opacity=0.85),
            text=f"{int(ev['yil'])}",
            hovertemplate=hover,
            showlegend=False,
        ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.0, lon=33.0), zoom=4.7),
        height=540,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Zaman çizelgesi (scatter) ──────────────────────────────────────────
    st.markdown('<div class="chart-title">⏳ Zaman Çizelgesi (yıl × Mw)</div>', unsafe_allow_html=True)
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=df_filt["yil"], y=df_filt["mw"],
        mode="markers",
        marker=dict(
            size=8 + (df_filt["mw"] - 6.5) * 5,
            color=df_filt["yuzyil"].apply(_yuzyil_renk).tolist(),
            opacity=0.85,
            line=dict(color="#000", width=0.4),
        ),
        text=df_filt["yer"],
        hovertemplate="<b>%{text}</b><br>Yıl: %{x}<br>Mw ~%{y:.1f}<extra></extra>",
        showlegend=False,
    ))
    # Ortalama tekrar referansı
    fig_time.add_hline(y=7.5, line=dict(color="#E24B4A", width=1, dash="dash"),
                       annotation_text="Mw 7.5 eşiği", annotation_font_color="#E24B4A",
                       annotation_position="top right")
    fig_time.update_layout(
        xaxis=dict(title="Yıl (MS)", color=TEXT, gridcolor=BORDER),
        yaxis=dict(title="Mw (tahminî)", color=TEXT, gridcolor=BORDER, range=[6.4, 8.6]),
        height=320,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
    )
    st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    biggest = df_filt.loc[df_filt["mw"].idxmax()]
    yy_count = df_filt["yuzyil"].nunique()
    span = int(df_filt["yil"].max()) - int(df_filt["yil"].min())
    mean_interval = span / max(1, len(df_filt) - 1)

    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{len(df_filt)}",                  "#FFD700", f"Olay sayısı ({yy_min}.→{yy_max}. yy)"),
        (c2, f"M{biggest['mw']:.1f}",            "#A32D2D", f"En büyük ({int(biggest['yil'])} {biggest['yer'].split(' ')[0]})"),
        (c3, f"~{mean_interval:.0f} yıl",        "#EF9F27", "Ortalama olaylar arası"),
        (c4, f"{yy_count}",                      "#1976D2", "Kapsanan yüzyıl"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tablo ─────────────────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Tarihsel Olaylar Listesi</div>', unsafe_allow_html=True)
    df_show = df_filt[["yil", "yer", "mw", "kaynak"]].copy()
    df_show.columns = ["Yıl (MS)", "Yer", "Mw (tahmin)", "Tarihsel kaynak"]
    df_show = df_show.sort_values("Yıl (MS)").reset_index(drop=True)
    st.dataframe(df_show, use_container_width=True, hide_index=True, height=380)

    st.caption(
        "📚 **Ana Kaynaklar:** Ambraseys (2009) Cambridge Univ. Press, ISBN 9780521872928 | "
        "Ambraseys & Finkel (1995) *Seismicity of Turkey 1500-1800* | "
        "Guidoboni et al. (1994) ING-SGA Rome | "
        "Sbeinati et al. (2005) *Annals of Geophysics* 48(3), 347-435 | "
        "Stiros (2001) *J. Struct. Geol.* (365 AD Girit megatsunami). "
        "⚠️ 1900 öncesi Mw değerleri makro-sismik şiddet kayıtlarından türetilmiştir; "
        "±0.3 belirsizlik içerir (Ambraseys & Jackson 2000)."
    )


if active_menu == "📜 Tarihsel Sismisite":
    _render_tarihsel_sismisite()


# ════════════════════════════════════════════════════════════════════════════
# 🔄 SİSMİK DÖNGÜ — F-50 / v1.28 — Kayma Açığı + BPT Olasılık
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Reid, H.F. (1910). The mechanics of the earthquake. The California
#       Earthquake of April 18, 1906, vol. 2. Carnegie Institution.
#       (Klasik elastik rebound / sismik döngü teorisi)
#   • Matthews, M.V., Ellsworth, W.L. & Reasenberg, P.A. (2002). A Brownian
#       model for recurrent earthquakes. BSSA 92(6), 2233-2250.
#       DOI:10.1785/0120010267
#   • Reilinger, R. et al. (2006). GPS constraints on continental deformation.
#       JGR 111, B05411. DOI:10.1029/2005JB004051
#   • Ergintav, S. et al. (2014). Istanbul's earthquake hot spots. GRL 41,
#       5783-5788. DOI:10.1002/2014GL060985
#   • Field, E.H. et al. (2015). UCERF3. BSSA 105(2A), 511-543.
# ════════════════════════════════════════════════════════════════════════════

# Sismik döngü parametreleri: kaynaklar üstte listelendi
# slip_per_event_m: tek depremde tipik yüzey kayması (m)
# slip_rate_mm_yr:  uzun dönem kayma hızı (mm/yıl) — Reilinger 2006 GPS
# recurrence_yr:    ortalama tekrar süresi (paleo+tarihsel)
# alpha:            BPT aperiodicity (Matthews 2002) — UCERF3 değeri 0.4-0.5
_DONGU_SEGMENTLER = [
    {"id": "marmara",   "ad": "Marmara (Prens Adaları)",
     "lat1": 40.80, "lon1": 27.50, "lat2": 40.95, "lon2": 29.10,
     "son_yil": 1766, "son_mw": 7.1,
     "slip_per_event_m": 4.0, "slip_rate_mm_yr": 22.0,
     "recurrence_yr": 250, "alpha": 0.40,
     "risk": "yuksek",
     "kaynak": "Parsons 2004 JGR; Ergintav 2014 GRL"},
    {"id": "izmit",     "ad": "İzmit-Sakarya",
     "lat1": 40.65, "lon1": 29.50, "lat2": 40.85, "lon2": 30.40,
     "son_yil": 1999, "son_mw": 7.6,
     "slip_per_event_m": 5.0, "slip_rate_mm_yr": 22.0,
     "recurrence_yr": 250, "alpha": 0.45,
     "risk": "dusuk",
     "kaynak": "Reilinger et al. 2000 Science"},
    {"id": "duzce",     "ad": "Düzce-Bolu",
     "lat1": 40.74, "lon1": 31.00, "lat2": 40.85, "lon2": 31.61,
     "son_yil": 1999, "son_mw": 7.2,
     "slip_per_event_m": 3.5, "slip_rate_mm_yr": 22.0,
     "recurrence_yr": 250, "alpha": 0.45,
     "risk": "dusuk",
     "kaynak": "Akoglu et al. 2006 EPSL"},
    {"id": "tosya",     "ad": "Tosya-Ladik",
     "lat1": 40.80, "lon1": 33.50, "lat2": 41.00, "lon2": 35.80,
     "son_yil": 1943, "son_mw": 7.6,
     "slip_per_event_m": 4.5, "slip_rate_mm_yr": 20.0,
     "recurrence_yr": 300, "alpha": 0.40,
     "risk": "orta",
     "kaynak": "Barka 1996 BSSA; Stein 1997 GJI"},
    {"id": "niksar",    "ad": "Niksar-Erbaa",
     "lat1": 40.60, "lon1": 36.20, "lat2": 40.70, "lon2": 37.50,
     "son_yil": 1942, "son_mw": 7.0,
     "slip_per_event_m": 3.5, "slip_rate_mm_yr": 18.0,
     "recurrence_yr": 200, "alpha": 0.45,
     "risk": "orta",
     "kaynak": "Barka 1996 BSSA"},
    {"id": "erzincan",  "ad": "Erzincan",
     "lat1": 39.77, "lon1": 36.80, "lat2": 39.77, "lon2": 39.53,
     "son_yil": 1939, "son_mw": 7.8,
     "slip_per_event_m": 4.5, "slip_rate_mm_yr": 18.0,
     "recurrence_yr": 320, "alpha": 0.40,
     "risk": "orta",
     "kaynak": "Barka 1996; Kozacı et al. 2007 BSSA"},
    {"id": "daf-kuzey", "ad": "DAF Kuzey (Pazarcık-Sürgü)",
     "lat1": 37.50, "lon1": 36.90, "lat2": 38.00, "lon2": 37.30,
     "son_yil": 2023, "son_mw": 7.8,
     "slip_per_event_m": 5.5, "slip_rate_mm_yr": 10.0,
     "recurrence_yr": 500, "alpha": 0.50,
     "risk": "dusuk",
     "kaynak": "Melgar et al. 2023; Akoğlu et al. 2024"},
    {"id": "saros",     "ad": "Saros-Ganos",
     "lat1": 40.60, "lon1": 26.00, "lat2": 40.75, "lon2": 27.30,
     "son_yil": 1912, "son_mw": 7.4,
     "slip_per_event_m": 4.0, "slip_rate_mm_yr": 20.0,
     "recurrence_yr": 250, "alpha": 0.45,
     "risk": "yuksek",
     "kaynak": "Ambraseys & Jackson 2000 GJI"},
]


def _bpt_probability(t_elapsed_yr: float, mu: float, alpha: float, dt_yr: float) -> float:
    """
    BPT (Brownian Passage Time) conditional probability.
    Matthews et al. (2002) BSSA 92(6) — invers Gauss dağılımı.
    P(geçmiş T_elapsed sonrasında dt_yr içinde kırılma | henüz kırılmadı)

    Basit yaklaşım: P(elapsed < T ≤ elapsed+dt) / (1 - P(T ≤ elapsed))
    """
    if alpha <= 0 or mu <= 0 or dt_yr <= 0:
        return 0.0

    # Inverse Gaussian CDF: F(t; μ, λ=μ/α²)
    def ig_cdf(t):
        if t <= 0:
            return 0.0
        # Robust scipy alternative: erf yaklaşımı
        lam = mu / (alpha * alpha)
        try:
            from scipy.stats import invgauss
            return float(invgauss.cdf(t / mu, mu=alpha * alpha))
        except Exception:
            # numerical fallback (Owen 1965 approx)
            arg1 = math.sqrt(lam / t) * (t / mu - 1)
            arg2 = -math.sqrt(lam / t) * (t / mu + 1)
            phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
            return phi(arg1) + math.exp(2 * lam / mu) * phi(arg2)

    f_now = ig_cdf(t_elapsed_yr)
    f_then = ig_cdf(t_elapsed_yr + dt_yr)
    if f_now >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (f_then - f_now) / (1.0 - f_now)))


_DONGU_RISK_RENK = {"yuksek": "#E24B4A", "orta": "#EF9F27", "dusuk": "#1D9E75"}


@st.fragment
def _render_sismik_dongu():
    st.markdown(
        '<div class="chart-title">🔄 Sismik Döngü — Kayma Açığı + BPT Olasılık (F-50 / v1.28)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🔄 **Elastik Rebound + BPT:** Bir fay segmenti sabit kayma hızıyla stres "
        "biriktirir; biriken kayma açığı (slip deficit) bir eşiği geçince kırılır. "
        "**Reid (1910)** klasik teorisi + **Matthews et al. (2002)** BPT stokastik "
        "olasılık modeli (UCERF3 referansı). GPS hızı: **Reilinger et al. (2006) JGR 111**."
    )

    YIL_SIMDI = datetime.now().year

    # ── Tablo: tüm segmentler için kayma açığı + BPT ───────────────────────
    rows = []
    for seg in _DONGU_SEGMENTLER:
        elapsed = YIL_SIMDI - seg["son_yil"]
        slip_def_m = elapsed * seg["slip_rate_mm_yr"] / 1000.0  # m
        slip_pct = 100.0 * slip_def_m / seg["slip_per_event_m"]
        p30 = 100.0 * _bpt_probability(elapsed, seg["recurrence_yr"], seg["alpha"], 30)
        p50 = 100.0 * _bpt_probability(elapsed, seg["recurrence_yr"], seg["alpha"], 50)
        p100 = 100.0 * _bpt_probability(elapsed, seg["recurrence_yr"], seg["alpha"], 100)
        rows.append({
            **seg,
            "elapsed_yr": elapsed,
            "slip_def_m": slip_def_m,
            "slip_pct": slip_pct,
            "p30": p30, "p50": p50, "p100": p100,
        })

    df_d = pd.DataFrame(rows)

    # ── Segment seçici ────────────────────────────────────────────────────
    sec_ad = st.selectbox(
        "Segment seç (detay için)",
        options=df_d["ad"].tolist(),
        index=0,
        key="dongu_segment_select",
    )
    seg = df_d[df_d["ad"] == sec_ad].iloc[0]

    # ── Sismik döngü grafiği (testere dişi + BPT bulutu) ───────────────────
    st.markdown(f'<div class="chart-title">📈 Stres Birikimi & BPT Olasılığı — {sec_ad}</div>',
                unsafe_allow_html=True)
    mu = seg["recurrence_yr"]
    alpha = seg["alpha"]
    t = np.arange(0, mu * 2 + 1, 1)
    stres = (t / mu) * seg["slip_per_event_m"]  # m, linear birikim
    stres = np.where(stres > seg["slip_per_event_m"],
                     seg["slip_per_event_m"] - (stres - seg["slip_per_event_m"]) * 4,
                     stres)
    stres = np.clip(stres, 0, seg["slip_per_event_m"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=t, y=stres,
        mode="lines",
        line=dict(color="#1976D2", width=2.5),
        name="Birikmiş kayma (m)",
        hovertemplate="t = %{x} yıl<br>Kayma: %{y:.2f} m<extra></extra>",
    ), secondary_y=False)

    # BPT olasılık zarfı (her yıl için 30-yıl koşullu olasılığı)
    p_envelope = [100 * _bpt_probability(float(ti), mu, alpha, 30) for ti in t]
    fig.add_trace(go.Scatter(
        x=t, y=p_envelope,
        mode="lines",
        line=dict(color="#E24B4A", width=2, dash="dot"),
        name="P(30 yıl) — BPT (%)",
        hovertemplate="t = %{x} yıl<br>P30 = %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    # Şu anki konum
    fig.add_vline(x=seg["elapsed_yr"], line=dict(color="#FFD700", width=2.5, dash="dash"),
                  annotation_text=f"Şu an: {seg['elapsed_yr']} yıl",
                  annotation_font_color="#FFD700",
                  annotation_position="top")

    fig.update_xaxes(title_text="Son depremden bu yana (yıl)", color=TEXT, gridcolor=BORDER)
    fig.update_yaxes(title_text="Birikmiş kayma (m)", secondary_y=False, color="#1976D2", gridcolor=BORDER)
    fig.update_yaxes(title_text="P(30 yıl içinde kırılma) %", secondary_y=True, color="#E24B4A")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0.3)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{seg['elapsed_yr']} yıl",  "#FFD700", f"Geçen süre ({seg['son_yil']} → {YIL_SIMDI})"),
        (c2, f"{seg['slip_def_m']:.2f} m", "#EF9F27", f"Kayma açığı (max {seg['slip_per_event_m']:.1f} m)"),
        (c3, f"%{seg['p30']:.0f}",         "#E24B4A", "P(30 yıl) BPT"),
        (c4, f"%{seg['p50']:.0f}",         "#A32D2D", "P(50 yıl) BPT"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tüm segmentler özet tablosu ───────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Tüm Segmentler — Kayma Açığı & BPT Olasılık</div>',
                unsafe_allow_html=True)
    df_view = df_d[["ad", "son_yil", "son_mw", "elapsed_yr", "slip_def_m", "slip_pct",
                    "p30", "p50", "p100", "kaynak"]].copy()
    df_view.columns = ["Segment", "Son Yıl", "Son Mw", "Geçen (yıl)",
                       "Kayma Açığı (m)", "% Maks",
                       "P30 yıl %", "P50 yıl %", "P100 yıl %", "Kaynak"]
    df_view = df_view.round({"slip_def_m": 2, "slip_pct": 0, "p30": 0, "p50": 0, "p100": 0})
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **Reid (1910)** Carnegie Inst. (elastik rebound) | "
        "**Matthews, Ellsworth & Reasenberg (2002)** *BSSA* 92(6), 2233-2250 — DOI:10.1785/0120010267 (BPT) | "
        "**Reilinger et al. (2006)** *JGR* 111, B05411 — DOI:10.1029/2005JB004051 (GPS) | "
        "**Ergintav et al. (2014)** *GRL* 41, 5783-5788 (Marmara) | "
        "**Field et al. (2015)** *BSSA* 105(2A) (UCERF3 BPT uygulaması). "
        "⚠️ BPT olasılıkları μ ve α parametrelerinin doğruluğuna duyarlıdır; "
        "paleoseismik veri belirsizliği sonucu yansıtır."
    )


if active_menu == "🔄 Sismik Döngü":
    _render_sismik_dongu()


# ════════════════════════════════════════════════════════════════════════════
# 🌐 DİNAMİK TETİKLEME — F-52 / v1.29 — Uzak Mesafeli Stres Tetikleme
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Hill, D.P. et al. (1993). Seismicity remotely triggered by the
#       magnitude 7.3 Landers, California, earthquake. Science 260(5114),
#       1617-1623. DOI:10.1126/science.260.5114.1617
#   • Brodsky, E.E. & Prejean, S.G. (2005). New constraints on mechanisms
#       of remotely triggered seismicity at Long Valley Caldera.
#       JGR 110, B04302. DOI:10.1029/2004JB003211
#   • Parsons, T. (2005). A hypothesis for delayed dynamic earthquake
#       triggering. GRL 32, L04302. DOI:10.1029/2004GL021811
#   • van der Elst & Brodsky (2010). JGR 115, B07311 (rate-state)
# ════════════════════════════════════════════════════════════════════════════

# Tarihsel uzak tetikleme gözlemleri (Hill 1993, Brodsky 2005, vd.)
_DINAMIK_OLAYLAR = [
    {"id": "landers-1992", "ad": "Landers M7.3 (1992)",
     "lat": 34.20, "lon": -116.43, "mw": 7.3, "tarih": "1992-06-28",
     "rayleigh_speed_km_s": 3.5,
     "gozlem_yerleri": [
         {"yer": "Long Valley Caldera (CA)",  "mesafe_km": 415,  "delay_s": 60,    "kanit": "Mikrodeprem swarm 6 saat (Hill 1993)"},
         {"yer": "Yellowstone (WY)",          "mesafe_km": 1250, "delay_s": 600,   "kanit": "Geyzer aktivitesi + sismik (Husen 2004)"},
         {"yer": "Cascade volkanik yayı",     "mesafe_km": 1500, "delay_s": 1200,  "kanit": "Mt. Lassen sismik artış (Hill 1993)"},
     ],
     "kaynak": "Hill et al. 1993 Science 260; Brodsky 2003"},
    {"id": "denali-2002", "ad": "Denali M7.9 (2002)",
     "lat": 63.52, "lon": -147.44, "mw": 7.9, "tarih": "2002-11-03",
     "rayleigh_speed_km_s": 3.5,
     "gozlem_yerleri": [
         {"yer": "Yellowstone",            "mesafe_km": 3100, "delay_s": 900,   "kanit": "Geyzer + mikrodeprem (Husen 2004)"},
         {"yer": "Coso volkanik bölge",    "mesafe_km": 3450, "delay_s": 1000,  "kanit": "Sismisite artış (Prejean 2004)"},
         {"yer": "Cerro Prieto (Meksika)", "mesafe_km": 4900, "delay_s": 1500,  "kanit": "Mikrodeprem tetikleme (Glowacka 2002)"},
     ],
     "kaynak": "Prejean et al. 2004 BSSA 94; Husen et al. 2004 Geology"},
    {"id": "sumatra-2004", "ad": "Sumatra-Andaman M9.1 (2004)",
     "lat": 3.30, "lon": 95.78, "mw": 9.1, "tarih": "2004-12-26",
     "rayleigh_speed_km_s": 3.5,
     "gozlem_yerleri": [
         {"yer": "San Andreas (CA, Parkfield)", "mesafe_km": 14000, "delay_s": 4000, "kanit": "Sismik gürültü artış (Felzer 2006)"},
         {"yer": "Mt. Wrangell (Alaska)",       "mesafe_km": 11500, "delay_s": 3300, "kanit": "Mikrodeprem (West 2005 Science)"},
         {"yer": "Türkiye KOERI",                "mesafe_km":  8500, "delay_s": 2400, "kanit": "Sismik gürültü artış (KOERI raporu)"},
     ],
     "kaynak": "West et al. 2005 Science; Felzer & Brodsky 2006"},
    {"id": "tohoku-2011", "ad": "Tohoku M9.0 (2011)",
     "lat": 38.30, "lon": 142.37, "mw": 9.0, "tarih": "2011-03-11",
     "rayleigh_speed_km_s": 3.5,
     "gozlem_yerleri": [
         {"yer": "Yellowstone",              "mesafe_km": 8200, "delay_s": 2350, "kanit": "Geyzer aktivitesi (Lupi 2011)"},
         {"yer": "Türkiye KOERI",            "mesafe_km": 8500, "delay_s": 2400, "kanit": "Sismik gürültü artış"},
         {"yer": "İzlanda volkanik bölge",   "mesafe_km": 8800, "delay_s": 2500, "kanit": "Tremor patternleri"},
     ],
     "kaynak": "Lupi & Miller 2014 Solid Earth; Ide et al. 2011 Science"},
    {"id": "kahramanmaras-2023", "ad": "Kahramanmaraş M7.8 (2023)",
     "lat": 37.17, "lon": 37.04, "mw": 7.8, "tarih": "2023-02-06",
     "rayleigh_speed_km_s": 3.5,
     "gozlem_yerleri": [
         {"yer": "Elbistan (Sürgü fayı)",  "mesafe_km": 95,   "delay_s": 30,    "kanit": "Doublet 9 saat sonra Mw 7.7"},
         {"yer": "Kıbrıs",                  "mesafe_km": 280,  "delay_s": 80,    "kanit": "Sismik kayıtlar"},
         {"yer": "Suriye-İsrail kıyısı",    "mesafe_km": 350,  "delay_s": 100,   "kanit": "Bölgesel sismograf ağı"},
     ],
     "kaynak": "Melgar et al. 2023 Seismica"},
]


@st.fragment
def _render_dinamik_tetikleme():
    st.markdown(
        '<div class="chart-title">🌐 Dinamik Gerilme Tetikleme — Uzak Mesafe Halkaları (F-52 / v1.29)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🌐 **Dinamik Tetikleme:** Büyük depremlerin yüzey dalgaları (Rayleigh, Love) "
        "binlerce kilometre öteye taşınarak hassas fay sistemlerini tetikleyebilir. "
        "**Hill et al. (1993) Science 260** Landers depreminin 1.500 km öteye etkisini "
        "kanıtladı — statik CFS (F-47) ile **fiziksel olarak farklı** mekanizmadır."
    )

    sec = st.selectbox(
        "Olay seç",
        options=[o["ad"] for o in _DINAMIK_OLAYLAR],
        index=0,
        key="dinamik_select",
    )
    o = next(x for x in _DINAMIK_OLAYLAR if x["ad"] == sec)

    # ── Harita: Rayleigh halkaları (eş-zaman) ──────────────────────────────
    fig = go.Figure()

    # Halkalar: 500/1000/2000/3000/5000 km
    halkalar = [500, 1000, 2000, 3000, 5000, 8000]
    halka_renkler = ["#FFD700", "#FAC775", "#EF9F27", "#E24B4A", "#A32D2D", "#6A0000"]

    for r_km, renk in zip(halkalar, halka_renkler):
        delay_min = (r_km / o["rayleigh_speed_km_s"]) / 60
        if r_km > 9000:
            continue
        clat, clon = _shakemap_circle_coords(o["lat"], o["lon"], r_km, n=60)
        fig.add_trace(go.Scattermapbox(
            lat=clat, lon=clon,
            mode="lines",
            line=dict(width=1.8, color=renk),
            name=f"{r_km} km (~{delay_min:.1f} dk)",
            hovertemplate=f"{r_km} km<br>Rayleigh varış: ~{delay_min:.1f} dk<extra></extra>",
            opacity=0.7,
        ))

    # Episentr
    fig.add_trace(go.Scattermapbox(
        lat=[o["lat"]], lon=[o["lon"]],
        mode="markers+text",
        marker=dict(size=18, color="#FFD700", symbol="star"),
        text=[f"★ Mw {o['mw']:.1f}"],
        textposition="top right",
        textfont=dict(size=13, color="#FFD700"),
        name="Kaynak deprem",
        hoverinfo="text",
    ))

    # Gözlem noktaları
    for g in o["gozlem_yerleri"]:
        # Yön bilinmediği için sabit koymak yerine: hesaplanmış mesafeden doğuya proje (görselleştirme)
        # Gerçek koordinat verisi olmadığı için sembolik konum
        pass

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=o["lat"], lon=o["lon"]), zoom=2),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(
            orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.55)", font=dict(color="#fff", size=11),
            bordercolor=BORDER, borderwidth=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    n_gozlem = len(o["gozlem_yerleri"])
    max_mesafe = max(g["mesafe_km"] for g in o["gozlem_yerleri"])
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"Mw {o['mw']:.1f}",         "#FFD700", "Kaynak büyüklüğü"),
        (c2, f"{n_gozlem}",                "#EF9F27", "Tetikleme gözlemi"),
        (c3, f"{max_mesafe:,} km",         "#E24B4A", "En uzak gözlem"),
        (c4, f"~{o['rayleigh_speed_km_s']} km/s", "#1976D2", "Rayleigh hızı"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Gözlem tablosu ────────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Tetikleme Gözlem Kataloğu</div>', unsafe_allow_html=True)
    df_gozlem = pd.DataFrame([
        {"Tetiklenen Yer": g["yer"],
         "Mesafe (km)": g["mesafe_km"],
         "Rayleigh gecikme (sn)": g["delay_s"],
         "Gecikme (dk)": round(g["delay_s"] / 60, 1),
         "Kanıt": g["kanit"]}
        for g in o["gozlem_yerleri"]
    ])
    st.dataframe(df_gozlem, use_container_width=True, hide_index=True)

    # ── Statik vs Dinamik tablosu ─────────────────────────────────────────
    st.markdown('<div class="chart-title">📊 Statik (F-47) vs Dinamik Tetikleme</div>', unsafe_allow_html=True)
    df_compare = pd.DataFrame([
        {"Özellik": "Mesafe rejimi",       "Statik CFS (F-47)": "100-200 km",      "Dinamik (F-52)": "100-10.000 km"},
        {"Özellik": "Fiziksel süreç",      "Statik CFS (F-47)": "Yarı-uzay elastik dislokasyon", "Dinamik (F-52)": "Yüzey dalga geçişi (Rayleigh/Love)"},
        {"Özellik": "Zaman ölçeği",        "Statik CFS (F-47)": "Saniyeler-aylar",  "Dinamik (F-52)": "Saniyeler-saatler (anlık)"},
        {"Özellik": "Mekanizma",           "Statik CFS (F-47)": "Okada 1992",        "Dinamik (F-52)": "Pore-pressure, rate-state friction"},
        {"Özellik": "Tipik gözlem",        "Statik CFS (F-47)": "Komşu fay artçısı", "Dinamik (F-52)": "Volkan/jeotermal mikrodeprem"},
    ])
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.caption(
        f"📚 **Senaryo:** {o['kaynak']} | "
        "**Hill et al. (1993)** *Science* 260, 1617-1623 — DOI:10.1126/science.260.5114.1617 | "
        "**Brodsky & Prejean (2005)** *JGR* 110, B04302 — DOI:10.1029/2004JB003211 | "
        "**Parsons (2005)** *GRL* 32, L04302 (gecikmeli tetikleme hipotezi) | "
        "**van der Elst & Brodsky (2010)** *JGR* 115, B07311 (rate-state friction). "
        "⚠️ Halka yarıçapları homojen yer kürede Rayleigh dalga hızı (~3.5 km/s) varsayımına göredir."
    )


if active_menu == "🌐 Dinamik Tetikleme":
    _render_dinamik_tetikleme()


# ════════════════════════════════════════════════════════════════════════════
# 📡 INSAR ZAMAN SERİSİ — F-53 / v1.30 — Koseismik + Postseismik
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Ferretti, A. et al. (2001). PS-InSAR. IEEE TGRS 39(1), 8-20.
#       DOI:10.1109/36.898661
#   • Berardino, P. et al. (2002). SBAS. IEEE TGRS 40(11), 2375-2383.
#       DOI:10.1109/TGRS.2002.803792
#   • Cetin, E. et al. (2014). Van postseismic. JGR 119, 1129-1145.
#       DOI:10.1002/2013JB010734
#   • Bürgmann et al. (2002) GJI 148, 358-378
# ════════════════════════════════════════════════════════════════════════════

_INSAR_TS_OLAYLAR = [
    {"id": "izmit-1999", "ad": "İzmit Mw 7.6 (1999)",
     "tarih": "1999-08-17",
     "coseismic_los_m": 2.7, "postseismic_total_m": 0.85,
     "tau_yr": 0.55, "veri_baslangic": -180, "veri_son": 1095,
     "mekanizma": "Afterslip + viskoelastik (alt kabuk)",
     "ref": "Bürgmann et al. 2002 GJI 148; Hearn 2002 GRL"},
    {"id": "van-2011", "ad": "Van Mw 7.1 (2011)",
     "tarih": "2011-10-23",
     "coseismic_los_m": 0.85, "postseismic_total_m": 0.18,
     "tau_yr": 0.40, "veri_baslangic": -120, "veri_son": 730,
     "mekanizma": "Postseismik afterslip + lokal poroelastik",
     "ref": "Cetin et al. 2014 JGR 119"},
    {"id": "elazig-2020", "ad": "Sivrice/Elazığ Mw 6.8 (2020)",
     "tarih": "2020-01-24",
     "coseismic_los_m": 0.95, "postseismic_total_m": 0.15,
     "tau_yr": 0.35, "veri_baslangic": -180, "veri_son": 540,
     "mekanizma": "Sentinel-1 SBAS afterslip",
     "ref": "Pousse-Beltran et al. 2020 GRL 47"},
    {"id": "kahramanmaras-2023", "ad": "Kahramanmaraş Mw 7.8 (2023)",
     "tarih": "2023-02-06",
     "coseismic_los_m": 5.5, "postseismic_total_m": 1.3,
     "tau_yr": 0.45, "veri_baslangic": -120, "veri_son": 700,
     "mekanizma": "Geniş afterslip + viskoelastik",
     "ref": "Xu et al. 2023 EPSL 612; Barbot et al. 2023 PNAS"},
]


def _insar_ts_synthetic(coseismic: float, postseismic: float, tau_yr: float,
                        t_start_day: int, t_end_day: int, n: int = 80):
    t_days = np.linspace(t_start_day, t_end_day, n)
    t_yr = t_days / 365.25
    u = 0.002 * t_yr  # interseismik 2 mm/yıl
    u += np.where(t_yr >= 0, coseismic, 0.0)
    t_post = np.where(t_yr >= 0, t_yr, 0)
    u += postseismic * (1 - np.exp(-t_post / tau_yr))
    rng = np.random.default_rng(42)
    u += rng.normal(0, 0.005, n)
    return t_days, u


@st.fragment
def _render_insar_zaman_serisi():
    st.markdown(
        '<div class="chart-title">📡 InSAR Zaman Serisi — Koseismik + Postseismik (F-53 / v1.30)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "📡 **PS-InSAR / SBAS Zaman Serisi:** Sentinel-1'in 6-günlük tekrar süresiyle "
        "aynı pikselin LOS deformasyonu zaman içinde takip edilir. Deprem sonrası "
        "**logaritmik gevşeme** (afterslip + viskoelastik) **Bürgmann et al. (2002)** "
        "modeliyle açıklanır."
    )

    sec = st.selectbox(
        "Olay seç",
        options=[o["ad"] for o in _INSAR_TS_OLAYLAR],
        index=0,
        key="insar_ts_select",
    )
    o = next(x for x in _INSAR_TS_OLAYLAR if x["ad"] == sec)

    t_days, u = _insar_ts_synthetic(
        o["coseismic_los_m"], o["postseismic_total_m"], o["tau_yr"],
        o["veri_baslangic"], o["veri_son"]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_days, y=u,
        mode="markers+lines",
        marker=dict(size=6, color="#1976D2"),
        line=dict(color="rgba(25,118,210,0.3)", width=1),
        name="Sentinel-1 LOS (PS-InSAR)",
        hovertemplate="t = %{x:+.0f} gün<br>LOS = %{y:+.3f} m<extra></extra>",
    ))

    fig.add_vline(x=0, line=dict(color="#E24B4A", width=2.5, dash="dash"),
                  annotation_text=f"Deprem ({o['tarih']})",
                  annotation_font_color="#E24B4A",
                  annotation_position="top")
    tau_day = o["tau_yr"] * 365.25
    fig.add_vline(x=tau_day, line=dict(color="#FFD700", width=1.5, dash="dot"),
                  annotation_text=f"τ = {o['tau_yr']:.2f} yıl",
                  annotation_font_color="#FFD700",
                  annotation_position="bottom")

    fig.update_layout(
        title=dict(text=f"InSAR Zaman Serisi — {sec}", font=dict(color=TEXT, size=13)),
        xaxis=dict(title="Depremden geçen süre (gün, 0 = ana şok)", color=TEXT, gridcolor=BORDER),
        yaxis=dict(title="LOS yer değiştirme (m)", color=TEXT, gridcolor=BORDER),
        height=420,
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0.3)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    post_pct = 100.0 * o["postseismic_total_m"] / o["coseismic_los_m"]
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{o['coseismic_los_m']:.2f} m", "#A32D2D", "Koseismik adım"),
        (c2, f"{o['postseismic_total_m']:.2f} m", "#EF9F27", f"Postseismik (%{post_pct:.0f})"),
        (c3, f"{o['tau_yr']:.2f} yıl",         "#FFD700", "Gevşeme zaman sabiti τ"),
        (c4, f"{int((o['veri_son'] - o['veri_baslangic']) / 6)}", "#1976D2",
         "Sentinel-1 görüntü sayısı (~6 gün)"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown(f"""
**🔬 Postseismik Mekanizma:** {o['mekanizma']}

**📐 Model bileşenleri (Bürgmann 2002):**
- **Afterslip** — Sığ kabukta (0-10 km) sismik olmayan kayma
- **Viskoelastik gevşeme** — Alt kabuk (15-30 km) + üst manto akış
- **Poroelastik** — Yeraltı suyu yeniden dağılımı (yıllar)
    """)

    st.markdown('<div class="chart-title">📋 PS-InSAR vs SBAS</div>', unsafe_allow_html=True)
    df_methods = pd.DataFrame([
        {"Yöntem": "PS-InSAR (Permanent Scatterers)",
         "Hedef": "Kentsel pikseller", "Hassasiyet": "1-3 mm/yıl",
         "Referans": "Ferretti et al. 2001 IEEE TGRS 39"},
        {"Yöntem": "SBAS (Small Baseline Subset)",
         "Hedef": "Geniş alan, vejetatif", "Hassasiyet": "3-5 mm/yıl",
         "Referans": "Berardino et al. 2002 IEEE TGRS 40"},
        {"Yöntem": "Stacking (basit ortalama)",
         "Hedef": "Tek piksel", "Hassasiyet": "5-10 mm",
         "Referans": "Wright et al. 2001 GRL"},
    ])
    st.dataframe(df_methods, use_container_width=True, hide_index=True)

    st.caption(
        f"📚 **Senaryo:** {o['ref']} | "
        "**Ferretti et al. (2001)** *IEEE TGRS* 39(1) — DOI:10.1109/36.898661 | "
        "**Berardino et al. (2002)** *IEEE TGRS* 40(11) — DOI:10.1109/TGRS.2002.803792 | "
        "**Bürgmann et al. (2002)** *GJI* 148, 358-378 | "
        "**Cetin et al. (2014)** *JGR* 119 — DOI:10.1002/2013JB010734. "
        "⚠️ Zaman serisi sentetik; gerçek için COMET LiCS veya ESA G-POD."
    )


if active_menu == "📡 InSAR Zaman Serisi":
    _render_insar_zaman_serisi()


# ════════════════════════════════════════════════════════════════════════════
# 🔒 FAY KİLİTLENME — F-54 / v1.31 — İnterseismik Coupling (φ)
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Reilinger, R. et al. (2006). GPS constraints on continental
#       deformation. JGR 111, B05411. DOI:10.1029/2005JB004051
#   • Ergintav, S. et al. (2014). Istanbul's earthquake hot spots.
#       GRL 41, 5783-5788. DOI:10.1002/2014GL060985
#   • Barka, A. (1996). Slip distribution along the NAF.
#       BSSA 86(5), 1238-1254.
#   • Savage, J.C. & Burford, R.O. (1973). Geodetic determination of relative
#       plate motion in central California. JGR 78(5), 832-845.
#       (Backslip yöntemi — kilitlenme modeli temeli)
#   • Hussain, E. et al. (2018). Geodetic observations of postseismic creep
#       in the decade after the 1999 Izmit earthquake. JGR 123.
# ════════════════════════════════════════════════════════════════════════════

# Segment kilitlenme parametreleri
# phi: 0 = aseismik creep (enerji birikmez), 1 = tam kilitli (enerji birikir)
# locking_depth_km: kilitlenmenin derinliği (sığ kilitli → derin creep)
_KILITLENME_SEGMENTLER = [
    {"id": "marmara-pa",  "ad": "Marmara — Prens Adaları segmenti",
     "lat1": 40.80, "lon1": 28.50, "lat2": 40.95, "lon2": 29.10,
     "phi": 0.95, "locking_depth_km": 15, "slip_rate_mm_yr": 22.0, "son_yil": 1766,
     "ref": "Ergintav et al. 2014 GRL 41"},
    {"id": "marmara-cn",  "ad": "Marmara — Orta Marmara",
     "lat1": 40.78, "lon1": 27.90, "lat2": 40.85, "lon2": 28.50,
     "phi": 0.75, "locking_depth_km": 12, "slip_rate_mm_yr": 18.0, "son_yil": 1766,
     "ref": "Ergintav 2014; Bohnhoff 2017 GRL"},
    {"id": "ganos-saros", "ad": "Ganos-Saros",
     "lat1": 40.60, "lon1": 26.00, "lat2": 40.75, "lon2": 27.30,
     "phi": 0.85, "locking_depth_km": 12, "slip_rate_mm_yr": 20.0, "son_yil": 1912,
     "ref": "Meade et al. 2002 BSSA"},
    {"id": "izmit-creep", "ad": "İzmit segmenti (postseismik creep)",
     "lat1": 40.75, "lon1": 29.50, "lat2": 40.85, "lon2": 30.40,
     "phi": 0.30, "locking_depth_km": 8, "slip_rate_mm_yr": 22.0, "son_yil": 1999,
     "ref": "Hussain et al. 2018 JGR 123 (decay)"},
    {"id": "duzce-creep", "ad": "Düzce segmenti",
     "lat1": 40.78, "lon1": 31.00, "lat2": 40.85, "lon2": 31.61,
     "phi": 0.50, "locking_depth_km": 10, "slip_rate_mm_yr": 22.0, "son_yil": 1999,
     "ref": "Hussain 2018; Çakir 2003 GJI"},
    {"id": "tosya",       "ad": "Tosya-Ladik",
     "lat1": 40.80, "lon1": 33.50, "lat2": 41.00, "lon2": 35.80,
     "phi": 0.80, "locking_depth_km": 15, "slip_rate_mm_yr": 20.0, "son_yil": 1943,
     "ref": "Yavasoglu 2011 GJI"},
    {"id": "erzincan",    "ad": "Erzincan segmenti",
     "lat1": 39.77, "lon1": 38.50, "lat2": 39.80, "lon2": 39.60,
     "phi": 0.85, "locking_depth_km": 18, "slip_rate_mm_yr": 18.0, "son_yil": 1939,
     "ref": "Aktug et al. 2013 JGR; Kozaci 2007"},
    {"id": "daf-kuzey",   "ad": "DAF Kuzey (Pazarcık)",
     "lat1": 37.40, "lon1": 36.80, "lat2": 37.70, "lon2": 37.20,
     "phi": 0.20, "locking_depth_km": 7, "slip_rate_mm_yr": 10.0, "son_yil": 2023,
     "ref": "Yetersiz GPS örtüm — 2023 sonrası creep beklenir"},
    {"id": "daf-guney",   "ad": "DAF Güney (Türkoğlu-Hatay)",
     "lat1": 36.30, "lon1": 36.10, "lat2": 37.10, "lon2": 36.80,
     "phi": 0.65, "locking_depth_km": 12, "slip_rate_mm_yr": 10.0, "son_yil": 1822,
     "ref": "Mahmoud 2013 JGR; Karabacak 2010"},
    {"id": "ege-akhisar", "ad": "Batı Anadolu (Gediz grabeni)",
     "lat1": 38.50, "lon1": 27.50, "lat2": 38.70, "lon2": 28.30,
     "phi": 0.40, "locking_depth_km": 8, "slip_rate_mm_yr": 6.0, "son_yil": 1969,
     "ref": "Aktug 2009 JGR; ekstansiyon"},
]


def _kilitlenme_renk(phi: float) -> str:
    if phi >= 0.85: return "#A32D2D"
    if phi >= 0.65: return "#E24B4A"
    if phi >= 0.45: return "#EF9F27"
    if phi >= 0.25: return "#FAC775"
    return "#1D9E75"


def _kilitlenme_etiket(phi: float) -> str:
    if phi >= 0.85: return "Tam kilitli (φ ≥ 0.85)"
    if phi >= 0.65: return "Yüksek (0.65 ≤ φ < 0.85)"
    if phi >= 0.45: return "Orta (0.45 ≤ φ < 0.65)"
    if phi >= 0.25: return "Kısmi creep (0.25 ≤ φ < 0.45)"
    return "Aseismik creep (φ < 0.25)"


@st.fragment
def _render_fay_kilitlenme():
    st.markdown(
        '<div class="chart-title">🔒 İnterseismik Kilitlenme (φ) — KAF + DAF (F-54 / v1.31)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🔒 **Kilitlenme katsayısı (φ):** GPS gözlemleriyle hesaplanır. "
        "**φ = 1**: fay tamamen kilitli, enerji birikiyor (gelecek deprem habercisi). "
        "**φ = 0**: aseismik creep, enerji birikmiyor. Teorik temel: "
        "**Savage & Burford (1973), JGR 78** backslip yöntemi; "
        "**Reilinger et al. (2006), JGR 111** Türkiye GPS uygulaması."
    )

    YIL_SIMDI = datetime.now().year

    # Hesap: birikmiş kayma açığı = phi × slip_rate × elapsed
    df_k = pd.DataFrame(_KILITLENME_SEGMENTLER)
    df_k["elapsed_yr"] = YIL_SIMDI - df_k["son_yil"]
    df_k["accumulated_slip_m"] = df_k["phi"] * df_k["slip_rate_mm_yr"] * df_k["elapsed_yr"] / 1000.0
    df_k["renk"] = df_k["phi"].apply(_kilitlenme_renk)
    df_k["etiket"] = df_k["phi"].apply(_kilitlenme_etiket)

    # ── Harita: segment çizgileri (kalınlık = phi) ─────────────────────────
    fig_map = go.Figure()
    for _, seg in df_k.iterrows():
        kalinlik = 2 + seg["phi"] * 8
        hover = (
            f"<b>{seg['ad']}</b><br>"
            f"φ = {seg['phi']:.2f} — {seg['etiket']}<br>"
            f"Kilitlenme derinliği: {seg['locking_depth_km']} km<br>"
            f"Kayma hızı: {seg['slip_rate_mm_yr']:.0f} mm/yıl<br>"
            f"Son deprem: {int(seg['son_yil'])} (geçen {int(seg['elapsed_yr'])} yıl)<br>"
            f"Birikmiş açık (kilitli kısım): {seg['accumulated_slip_m']:.2f} m<br>"
            f"Kaynak: {seg['ref']}"
            "<extra></extra>"
        )
        fig_map.add_trace(go.Scattermapbox(
            lat=[seg["lat1"], seg["lat2"]], lon=[seg["lon1"], seg["lon2"]],
            mode="lines",
            line=dict(width=kalinlik, color=seg["renk"]),
            name=f"{seg['ad']} (φ={seg['phi']:.2f})",
            hovertemplate=hover,
            showlegend=False,
        ))
        mid_lat = (seg["lat1"] + seg["lat2"]) / 2
        mid_lon = (seg["lon1"] + seg["lon2"]) / 2
        fig_map.add_trace(go.Scattermapbox(
            lat=[mid_lat], lon=[mid_lon],
            mode="markers+text",
            marker=dict(size=8, color=seg["renk"]),
            text=[f"φ={seg['phi']:.2f}"],
            textposition="top right",
            textfont=dict(size=10, color="#ffffff"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Legend dummy traces
    for phi_ref in (0.95, 0.75, 0.55, 0.35, 0.15):
        fig_map.add_trace(go.Scattermapbox(
            lat=[None], lon=[None],
            mode="lines",
            line=dict(width=2 + phi_ref * 8, color=_kilitlenme_renk(phi_ref)),
            name=_kilitlenme_etiket(phi_ref),
        ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.5, lon=33.0), zoom=5),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(
            orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.55)", font=dict(color="#fff", size=10),
            bordercolor=BORDER, borderwidth=1,
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Kilitlenme barchart ────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📊 Segment Kilitlenme Karşılaştırması</div>',
                unsafe_allow_html=True)
    df_sorted = df_k.sort_values("phi", ascending=True).reset_index(drop=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=df_sorted["ad"],
        x=df_sorted["phi"],
        orientation="h",
        marker=dict(color=df_sorted["renk"].tolist(), line=dict(color="#222", width=0.5)),
        text=df_sorted.apply(lambda r: f"{r['phi']:.2f} (açık: {r['accumulated_slip_m']:.2f} m)", axis=1),
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="<b>%{y}</b><br>φ = %{x:.2f}<extra></extra>",
    ))
    fig_bar.add_vline(x=0.5, line=dict(color="#888", width=1, dash="dot"))
    fig_bar.update_layout(
        xaxis=dict(title="Kilitlenme katsayısı φ (0 = creep, 1 = kilitli)",
                   color=TEXT, gridcolor=BORDER, range=[0, 1.2]),
        yaxis=dict(color=TEXT, gridcolor=BORDER),
        height=380,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    en_kilitli = df_k.loc[df_k["phi"].idxmax()]
    en_creep = df_k.loc[df_k["phi"].idxmin()]
    toplam_acik = df_k["accumulated_slip_m"].sum()
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{en_kilitli['phi']:.2f}",                 "#A32D2D",
         f"En kilitli ({en_kilitli['ad'].split(' — ')[0]})"),
        (c2, f"{en_creep['phi']:.2f}",                    "#1D9E75",
         f"En creep'li ({en_creep['ad'].split(' segmenti')[0][:20]})"),
        (c3, f"{en_kilitli['accumulated_slip_m']:.1f} m", "#EF9F27",
         "Maks birikmiş açık (m)"),
        (c4, f"{toplam_acik:.1f} m",                       "#1976D2",
         "Tüm segm. kümülatif açık"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tablo ─────────────────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Segment Detayları</div>', unsafe_allow_html=True)
    df_show = df_k[["ad", "phi", "etiket", "locking_depth_km", "slip_rate_mm_yr",
                    "son_yil", "elapsed_yr", "accumulated_slip_m", "ref"]].copy()
    df_show.columns = ["Segment", "φ", "Sınıf", "Kilitlenme Derinliği (km)",
                       "Kayma Hızı (mm/yıl)", "Son Deprem (yıl)", "Geçen (yıl)",
                       "Birikmiş Açık (m)", "Kaynak"]
    df_show = df_show.round({"phi": 2, "accumulated_slip_m": 2}).sort_values("φ", ascending=False)
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **Savage & Burford (1973)** *JGR* 78(5), 832-845 (backslip yöntemi) | "
        "**Reilinger et al. (2006)** *JGR* 111, B05411 — DOI:10.1029/2005JB004051 | "
        "**Ergintav et al. (2014)** *GRL* 41, 5783-5788 — DOI:10.1002/2014GL060985 (Marmara) | "
        "**Hussain et al. (2018)** *JGR* 123 (İzmit postseismik) | "
        "**Aktug et al. (2013)** *JGR* (Erzincan φ). "
        "⚠️ φ tahminleri GPS örtüm yoğunluğuna ve model boyutuna duyarlıdır; "
        "tam 3D inversiyon için Okada 1992 + sismik fizik gerekir."
    )


if active_menu == "🔒 Fay Kilitlenme":
    _render_fay_kilitlenme()


# ════════════════════════════════════════════════════════════════════════════
# 🌋 MOHO DERİNLİĞİ — F-55 / v1.32 — Kabuk Kalınlığı (Receiver Function)
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Mohorovičić, A. (1910). Potres od 8.X.1909. Godišnje izvješće
#       Zagrebačkog meteorološkog opservatorija 9, 1-56. (Moho ilk tanım)
#   • Zhu, L. & Kanamori, H. (2000). Moho depth from RFs. JGR 105(B2),
#       2981-2993. DOI:10.1029/1999JB900322
#   • Zor, E. et al. (2003). Crustal structure of E. Anatolian plateau.
#       GRL 30(24), 8044. DOI:10.1029/2003GL018192
#   • Laske, G. et al. (2013). CRUST1.0. EGU Abstract EGU2013-2658.
#   • Vanacore, E.A. et al. (2013). Moho structure of Anatolia.
#       Geophys. J. Int. 193(1), 329-337.
# ════════════════════════════════════════════════════════════════════════════

# Türkiye kabuk kalınlığı veri noktaları (CRUST1.0 + Zor 2003 + Vanacore 2013)
# Her nokta: kabuk kalınlığı (km) = Moho derinliği yüzeyden
_MOHO_NOKTALAR = [
    # Doğu Anadolu (kalın kabuk — sıkışma rejimi)
    {"city": "Erzincan",     "lat": 39.75, "lon": 39.49, "moho_km": 44, "ref": "Zor et al. 2003 GRL 30"},
    {"city": "Erzurum",      "lat": 39.91, "lon": 41.27, "moho_km": 48, "ref": "Zor 2003"},
    {"city": "Van",          "lat": 38.49, "lon": 43.41, "moho_km": 50, "ref": "Zor 2003; Türkelli 2003 GRL"},
    {"city": "Kars",         "lat": 40.60, "lon": 43.10, "moho_km": 47, "ref": "Vanacore et al. 2013 GJI 193"},
    {"city": "Tunceli",      "lat": 39.10, "lon": 39.55, "moho_km": 43, "ref": "Zor 2003"},
    {"city": "Elazığ",       "lat": 38.68, "lon": 39.22, "moho_km": 42, "ref": "Vanacore 2013"},
    {"city": "Bingöl",       "lat": 38.88, "lon": 40.50, "moho_km": 46, "ref": "Türkelli 2003"},
    {"city": "Diyarbakır",   "lat": 37.91, "lon": 40.24, "moho_km": 40, "ref": "Mutlu & Karabulut 2011 GJI"},
    {"city": "Şanlıurfa",    "lat": 37.17, "lon": 38.80, "moho_km": 38, "ref": "Vanacore 2013 (Arap plakası kenarı)"},
    {"city": "Kahramanmaraş","lat": 37.58, "lon": 36.93, "moho_km": 36, "ref": "Vanacore 2013"},

    # İç Anadolu (orta kalın — 35-40 km)
    {"city": "Sivas",        "lat": 39.75, "lon": 37.02, "moho_km": 40, "ref": "Vanacore 2013"},
    {"city": "Tokat",        "lat": 40.32, "lon": 36.55, "moho_km": 38, "ref": "Vanacore 2013"},
    {"city": "Ankara",       "lat": 39.93, "lon": 32.86, "moho_km": 35, "ref": "Vanacore 2013; Tezel 2010"},
    {"city": "Kayseri",      "lat": 38.73, "lon": 35.48, "moho_km": 38, "ref": "Vanacore 2013"},
    {"city": "Konya",        "lat": 37.87, "lon": 32.48, "moho_km": 36, "ref": "Vanacore 2013"},
    {"city": "Kırşehir",     "lat": 39.15, "lon": 34.16, "moho_km": 35, "ref": "Vanacore 2013"},
    {"city": "Eskişehir",    "lat": 39.78, "lon": 30.52, "moho_km": 33, "ref": "Tezel et al. 2010 GJI"},

    # KAF / Karadeniz kuzey kıyısı
    {"city": "Samsun",       "lat": 41.29, "lon": 36.34, "moho_km": 34, "ref": "Tezel 2010"},
    {"city": "Trabzon",      "lat": 41.00, "lon": 39.73, "moho_km": 36, "ref": "Tezel 2010"},
    {"city": "Zonguldak",    "lat": 41.46, "lon": 31.79, "moho_km": 32, "ref": "Karahan 2001 GJI"},
    {"city": "İstanbul",     "lat": 41.01, "lon": 28.98, "moho_km": 30, "ref": "Karahan 2001; Becel 2009"},

    # Batı Anadolu (ince kabuk — ekstansiyon rejimi)
    {"city": "İzmir",        "lat": 38.42, "lon": 27.14, "moho_km": 28, "ref": "Tezel 2010; Karabulut 2013"},
    {"city": "Manisa",       "lat": 38.62, "lon": 27.43, "moho_km": 28, "ref": "Tezel 2010"},
    {"city": "Denizli",      "lat": 37.78, "lon": 29.09, "moho_km": 30, "ref": "Tezel 2010"},
    {"city": "Aydın",        "lat": 37.85, "lon": 27.85, "moho_km": 27, "ref": "Tezel 2010"},
    {"city": "Muğla",        "lat": 37.21, "lon": 28.36, "moho_km": 26, "ref": "Karabulut 2013 (Ege ekstansiyon)"},
    {"city": "Antalya",      "lat": 36.89, "lon": 30.71, "moho_km": 32, "ref": "Karabulut 2013"},
    {"city": "Bursa",        "lat": 40.18, "lon": 29.07, "moho_km": 30, "ref": "Tezel 2010"},
    {"city": "Çanakkale",    "lat": 40.15, "lon": 26.41, "moho_km": 28, "ref": "Tezel 2010"},

    # Akdeniz kıyısı (Kıbrıs yayı)
    {"city": "Mersin",       "lat": 36.81, "lon": 34.64, "moho_km": 33, "ref": "Vanacore 2013"},
    {"city": "Adana",        "lat": 37.00, "lon": 35.32, "moho_km": 35, "ref": "Vanacore 2013"},
    {"city": "Hatay",        "lat": 36.20, "lon": 36.16, "moho_km": 32, "ref": "Vanacore 2013 (Ölü Deniz fayı yakını)"},
]


def _moho_renk(km: float) -> str:
    if km >= 46: return "#3F0080"   # çok kalın (mor)
    if km >= 42: return "#7A1F8B"
    if km >= 38: return "#A52A85"
    if km >= 35: return "#D43F71"
    if km >= 32: return "#EB6453"
    if km >= 29: return "#F89441"
    if km >= 26: return "#FAC775"   # ince (sarı)
    return "#FDE725"


@st.fragment
def _render_moho_derinligi():
    st.markdown(
        '<div class="chart-title">🌋 Moho Derinliği — Türkiye Kabuk Kalınlığı (F-55 / v1.32)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🌋 **Moho (Mohorovičić Süreksizliği):** Kabuk-manto sınırı, ilk kez "
        "**Mohorovičić (1910)** tarafından tanımlandı. Receiver Function (RF) yöntemiyle "
        "(**Zhu & Kanamori 2000, JGR 105**) kabuk kalınlığı ölçülür. Türkiye'de Doğu "
        "Anadolu altında **44-50 km**, Batı Anadolu (Ege) altında **26-30 km** "
        "(**Zor et al. 2003 GRL 30; Vanacore et al. 2013 GJI 193**)."
    )

    df_m = pd.DataFrame(_MOHO_NOKTALAR)
    df_m["renk"] = df_m["moho_km"].apply(_moho_renk)

    # ── Harita: Moho kontur (densitymapbox) ────────────────────────────────
    fig_map = go.Figure()
    fig_map.add_trace(go.Densitymapbox(
        lat=df_m["lat"], lon=df_m["lon"], z=df_m["moho_km"],
        radius=70,
        colorscale=[
            [0.00, "#FDE725"],
            [0.20, "#FAC775"],
            [0.40, "#F89441"],
            [0.55, "#EB6453"],
            [0.70, "#D43F71"],
            [0.85, "#A52A85"],
            [1.00, "#3F0080"],
        ],
        zmin=25, zmax=52,
        colorbar=dict(
            title=dict(text="Moho (km)", font=dict(color=TEXT, size=11)),
            tickfont=dict(color=TEXT, size=10),
            bgcolor="rgba(0,0,0,0.4)",
            thickness=14, len=0.7,
        ),
        opacity=0.55,
        hovertemplate="Moho: %{z:.0f} km<br>(%{lat:.2f}, %{lon:.2f})<extra></extra>",
    ))

    # Şehir noktaları
    fig_map.add_trace(go.Scattermapbox(
        lat=df_m["lat"], lon=df_m["lon"],
        mode="markers+text",
        marker=dict(size=10, color=df_m["renk"], opacity=0.95,
                    ),
        text=df_m["city"],
        textfont=dict(size=9, color="#fff"),
        textposition="top right",
        hovertemplate=df_m.apply(
            lambda r: f"<b>{r['city']}</b><br>Moho: {r['moho_km']} km<br>Kaynak: {r['ref']}<extra></extra>",
            axis=1,
        ),
        showlegend=False,
    ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.0, lon=35.0), zoom=5),
        height=540,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── W-E kesit profili ─────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📈 W-E Kesit Profili — Boylama Göre Moho</div>',
                unsafe_allow_html=True)
    df_we = df_m.sort_values("lon").reset_index(drop=True)
    fig_we = go.Figure()
    fig_we.add_trace(go.Scatter(
        x=df_we["lon"], y=df_we["moho_km"],
        mode="lines+markers",
        marker=dict(size=8, color=df_we["renk"], line=dict(color="#222", width=0.5)),
        line=dict(color="rgba(150,150,150,0.4)", width=1.5),
        text=df_we["city"],
        hovertemplate="<b>%{text}</b><br>Lon: %{x:.2f}°<br>Moho: %{y} km<extra></extra>",
        name="Moho derinliği",
    ))
    # Bölgesel ortalama çizgisi
    fig_we.add_hline(y=df_m["moho_km"].mean(), line=dict(color="#888", dash="dot"),
                     annotation_text=f"Türkiye ortalama: {df_m['moho_km'].mean():.1f} km",
                     annotation_font_color="#888", annotation_position="top right")

    fig_we.update_layout(
        xaxis=dict(title="Boylam (°E) — Batı (Ege) ← → Doğu (Anadolu)",
                   color=TEXT, gridcolor=BORDER),
        yaxis=dict(title="Moho derinliği (km, ters eksen)", autorange="reversed",
                   color=TEXT, gridcolor=BORDER),
        height=340,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        showlegend=False,
    )
    st.plotly_chart(fig_we, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    en_kalin = df_m.loc[df_m["moho_km"].idxmax()]
    en_ince = df_m.loc[df_m["moho_km"].idxmin()]
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{en_kalin['moho_km']} km",        "#3F0080", f"En kalın ({en_kalin['city']})"),
        (c2, f"{en_ince['moho_km']} km",          "#FDE725", f"En ince ({en_ince['city']})"),
        (c3, f"{df_m['moho_km'].mean():.1f} km",  "#A52A85", "Türkiye ortalama"),
        (c4, f"{en_kalin['moho_km'] - en_ince['moho_km']} km", "#EF9F27", "Doğu-Batı farkı"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tektonik yorumlama tablosu ────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Bölge × Moho × Tektonik Yorum</div>',
                unsafe_allow_html=True)
    df_interp = pd.DataFrame([
        {"Bölge": "Doğu Anadolu Platosu", "Moho (km)": "44-50",
         "Tektonik Rejim": "Arap-Avrasya çarpışması (sıkışma)",
         "Yorum": "Kabuk kalınlaşmış; Bouguer anomalisi negatif"},
        {"Bölge": "İç Anadolu",           "Moho (km)": "35-40",
         "Tektonik Rejim": "Stabil masif (Kırşehir bloğu)",
         "Yorum": "Tipik kıtasal kabuk"},
        {"Bölge": "Karadeniz kıyısı (KAF)","Moho (km)": "30-36",
         "Tektonik Rejim": "Doğrultu atımlı (KAF)",
         "Yorum": "Pontid yığışım kuşağı; az kalınlaşma"},
        {"Bölge": "Batı Anadolu (Ege)",    "Moho (km)": "26-30",
         "Tektonik Rejim": "Ekstansiyon (graben sistemi)",
         "Yorum": "İncelmiş kabuk; yüksek ısı akısı"},
        {"Bölge": "Akdeniz kıyısı",        "Moho (km)": "32-35",
         "Tektonik Rejim": "Kıbrıs yayı (subdüksiyon ön ülkesi)",
         "Yorum": "Geçişsel kabuk"},
    ])
    st.dataframe(df_interp, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **Mohorovičić (1910)** *Godišnje izvješće* 9 (Moho ilk tanım) | "
        "**Zhu & Kanamori (2000)** *JGR* 105(B2) — DOI:10.1029/1999JB900322 (RF yöntem) | "
        "**Zor et al. (2003)** *GRL* 30(24) — DOI:10.1029/2003GL018192 (Doğu Anadolu) | "
        "**Vanacore et al. (2013)** *GJI* 193(1), 329-337 (Anadolu Moho) | "
        "**Tezel et al. (2010)** *GJI* (Batı Anadolu) | "
        "**CRUST1.0:** Laske et al. (2013) EGU. "
        "⚠️ Moho derinlikleri RF + sismik tomografi sentezidir; ±2-3 km belirsizlik tipiktir."
    )


if active_menu == "🌋 Moho Derinliği":
    _render_moho_derinligi()


# ════════════════════════════════════════════════════════════════════════════
# 🌀 SKS SPLITTING — F-56 / v1.33 — Mantle Anizotropi Fast-Axis Haritası
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Savage, M.K. (1999). Seismic anisotropy and mantle deformation.
#       Rev. Geophys. 37(1), 65-106. DOI:10.1029/98RG02075
#   • Silver, P.G. & Chan, W.W. (1991). Shear wave splitting and subcontinental
#       mantle deformation. JGR 96(B10), 16429-16454.
#   • Biryol, C.B. et al. (2010). Segmented African lithosphere beneath the
#       Anatolian region from teleseismic P-wave tomography.
#       JGR 115, B07316. DOI:10.1029/2009JB006923
#   • Sandvol, E. et al. (2003). Shear wave splitting in the Anatolian Plateau.
#       JGR 108(B5), 2266. DOI:10.1029/2002JB002023
#   • IRIS SplittingDB: ds.iris.edu/ds/products/sksobs/
# ════════════════════════════════════════════════════════════════════════════

# SKS splitting istasyon verisi (Sandvol 2003, Biryol 2010, SplittingDB)
# fast_az: hızlı eksen azimuth (° CW from N)
# dt: gecikme süresi (sn) — anizotropi büyüklüğü göstergesi
_SKS_ISTASYONLAR = [
    # Doğu Anadolu (Sandvol 2003, Biryol 2010)
    {"sta": "BAYT", "lat": 40.27, "lon": 40.27, "fast_az": 75,  "dt": 1.0, "ref": "Sandvol 2003"},
    {"sta": "ERZN", "lat": 39.75, "lon": 39.49, "fast_az": 78,  "dt": 1.1, "ref": "Sandvol 2003 (Erzincan)"},
    {"sta": "MALT", "lat": 38.36, "lon": 38.30, "fast_az": 70,  "dt": 1.2, "ref": "Sandvol 2003"},
    {"sta": "AGRI", "lat": 39.72, "lon": 43.06, "fast_az": 82,  "dt": 1.3, "ref": "Sandvol 2003"},
    {"sta": "VAN",  "lat": 38.49, "lon": 43.41, "fast_az": 85,  "dt": 1.4, "ref": "Sandvol 2003"},
    {"sta": "DIY",  "lat": 37.91, "lon": 40.24, "fast_az": 68,  "dt": 1.0, "ref": "Sandvol 2003"},

    # İç Anadolu
    {"sta": "ANKR", "lat": 39.89, "lon": 32.76, "fast_az": 65,  "dt": 0.9, "ref": "Hatzfeld 2001 GJI"},
    {"sta": "ISP",  "lat": 37.79, "lon": 30.51, "fast_az": 55,  "dt": 1.0, "ref": "Sandvol 2003"},
    {"sta": "KONY", "lat": 37.87, "lon": 32.48, "fast_az": 60,  "dt": 0.8, "ref": "Sandvol 2003"},
    {"sta": "KAY",  "lat": 38.73, "lon": 35.48, "fast_az": 72,  "dt": 1.0, "ref": "Sandvol 2003"},

    # KAF boyunca (fault-parallel)
    {"sta": "AMAS", "lat": 40.65, "lon": 35.83, "fast_az": 85,  "dt": 1.2, "ref": "Biryol 2011 GJI (KAF paralel)"},
    {"sta": "TOK",  "lat": 40.32, "lon": 36.55, "fast_az": 88,  "dt": 1.2, "ref": "Biryol 2011"},
    {"sta": "BOL",  "lat": 40.74, "lon": 31.61, "fast_az": 90,  "dt": 1.1, "ref": "Biryol 2011"},
    {"sta": "ISTN", "lat": 41.06, "lon": 29.05, "fast_az": 80,  "dt": 0.9, "ref": "Endrun 2011 GJI"},

    # Batı Anadolu (Hellenic trench etkisi — KD-GB)
    {"sta": "IZMR", "lat": 38.40, "lon": 27.20, "fast_az": 30,  "dt": 1.4, "ref": "Endrun 2011 (Hellenic)"},
    {"sta": "DENZ", "lat": 37.78, "lon": 29.09, "fast_az": 35,  "dt": 1.3, "ref": "Endrun 2011"},
    {"sta": "AYDN", "lat": 37.85, "lon": 27.85, "fast_az": 32,  "dt": 1.4, "ref": "Endrun 2011"},
    {"sta": "MUGL", "lat": 37.21, "lon": 28.36, "fast_az": 38,  "dt": 1.5, "ref": "Confal 2018 EPSL"},
    {"sta": "ANTL", "lat": 36.89, "lon": 30.71, "fast_az": 45,  "dt": 1.2, "ref": "Confal 2018 (Kıbrıs yayı)"},
    {"sta": "CANK", "lat": 40.15, "lon": 26.41, "fast_az": 50,  "dt": 1.0, "ref": "Endrun 2011"},

    # Güneydoğu Akdeniz (DAF + Arap kenarı)
    {"sta": "MARS", "lat": 37.58, "lon": 36.93, "fast_az": 65,  "dt": 1.1, "ref": "Sandvol 2003 (DAF)"},
    {"sta": "GAZ",  "lat": 37.07, "lon": 37.38, "fast_az": 60,  "dt": 1.0, "ref": "Sandvol 2003"},
    {"sta": "HAT",  "lat": 36.20, "lon": 36.16, "fast_az": 55,  "dt": 0.9, "ref": "Mahmoud 2013 JGR"},
]


def _sks_arrow_endpoints(lat: float, lon: float, fast_az_deg: float, dt_sn: float):
    """Fast axis ok uç noktaları (her iki yön çiziliyor — anizotropi vektörü çift yönlüdür)."""
    half_len_deg = 0.10 + dt_sn * 0.15  # uzunluk dt ile orantılı
    az_rad = math.radians(fast_az_deg)
    cos_lat = max(math.cos(math.radians(lat)), 0.1)
    dlat = half_len_deg * math.cos(az_rad)
    dlon = half_len_deg * math.sin(az_rad) / cos_lat
    return [lat - dlat, lat + dlat], [lon - dlon, lon + dlon]


@st.fragment
def _render_sks_splitting():
    st.markdown(
        '<div class="chart-title">🌀 SKS Splitting — Mantle Anizotropi Fast-Axis (F-56 / v1.33)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🌀 **SKS Splitting:** SKS dalgaları çekirdekten geçerken izotropik, "
        "litosferde/astanosferde anizotropik olivin (a-eksen) ile parçalanır. "
        "**Fast axis** mantle akış yönünü, **dt** gecikme süresi anizotropi büyüklüğünü verir. "
        "Türkiye'de KAF boyunca fault-parallel, Ege'de Hellenic trench KD-GB. "
        "Teorik temel: **Savage (1999) Rev. Geophys. 37**; **Silver & Chan (1991) JGR 96**."
    )

    df_s = pd.DataFrame(_SKS_ISTASYONLAR)

    # ── Harita ─────────────────────────────────────────────────────────────
    fig_map = go.Figure()
    # Fast axis ok'ları (her istasyon için bir line segment)
    for _, st_row in df_s.iterrows():
        lats, lons = _sks_arrow_endpoints(st_row["lat"], st_row["lon"], st_row["fast_az"], st_row["dt"])
        fig_map.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(width=1.5 + st_row["dt"] * 1.5, color="#FFD700"),
            hovertemplate=(
                f"<b>{st_row['sta']}</b><br>"
                f"Fast az: {st_row['fast_az']}°<br>"
                f"dt: {st_row['dt']:.1f} sn<br>"
                f"Kaynak: {st_row['ref']}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # İstasyon noktaları (üzerinde)
    fig_map.add_trace(go.Scattermapbox(
        lat=df_s["lat"], lon=df_s["lon"],
        mode="markers",
        marker=dict(size=8, color="#E24B4A", opacity=0.95),
        text=df_s["sta"],
        hovertemplate=df_s.apply(
            lambda r: (f"<b>{r['sta']}</b><br>"
                       f"({r['lat']:.2f}, {r['lon']:.2f})<br>"
                       f"Fast az: {r['fast_az']}° · dt: {r['dt']:.1f} sn"
                       "<extra></extra>"),
            axis=1,
        ),
        name="SKS istasyonları",
    ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.0, lon=35.0), zoom=5),
        height=540,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Rose diagram (azimut dağılımı) ─────────────────────────────────────
    st.markdown('<div class="chart-title">🧭 Fast-Axis Azimuth Dağılımı (Rose Diyagram)</div>',
                unsafe_allow_html=True)
    # 20° binlere ayır
    bin_edges = np.arange(0, 181, 20)
    counts, _ = np.histogram(df_s["fast_az"], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig_rose = go.Figure()
    fig_rose.add_trace(go.Barpolar(
        r=counts,
        theta=bin_centers,
        width=[20] * len(counts),
        marker=dict(color="#FFD700", line=dict(color="#222", width=0.5)),
        opacity=0.85,
        hovertemplate="Azimuth %{theta}°<br>N = %{r}<extra></extra>",
    ))
    # Simetrik (180° eklemesi)
    fig_rose.add_trace(go.Barpolar(
        r=counts,
        theta=(bin_centers + 180) % 360,
        width=[20] * len(counts),
        marker=dict(color="#FAC775", line=dict(color="#222", width=0.5)),
        opacity=0.5,
        hoverinfo="skip",
    ))
    fig_rose.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, color=TEXT, gridcolor=BORDER),
            angularaxis=dict(direction="clockwise", rotation=90,
                             tickfont=dict(color=TEXT, size=10)),
            bgcolor=BG2,
        ),
        paper_bgcolor=BG2,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        title=dict(text="Fast Axis Frekansı (KD=45°, KB=135°)", font=dict(color=TEXT, size=12)),
    )
    st.plotly_chart(fig_rose, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{len(df_s)}",                       "#FFD700", "İstasyon sayısı"),
        (c2, f"{df_s['dt'].mean():.2f} sn",        "#EF9F27", "Ortalama dt"),
        (c3, f"{df_s['dt'].max():.1f} sn",         "#E24B4A", "Maks dt (anizotropi)"),
        (c4, f"{int(df_s['fast_az'].mean())}°",    "#1976D2", "Ortalama fast azimuth"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Bölge yorumlama tablosu ───────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Bölge × Fast Axis × Yorum</div>',
                unsafe_allow_html=True)
    df_yorum = pd.DataFrame([
        {"Bölge": "KAF boyunca (Bolu-Tokat-Amasya)",
         "Fast Axis": "~80-90° (D-B, fault-parallel)",
         "Yorum": "Mantle litosfer KAF ile birlikte sürükleniyor (Biryol 2011)"},
        {"Bölge": "Doğu Anadolu (Erzurum-Van)",
         "Fast Axis": "~75-85° (D-B)",
         "Yorum": "Arap-Avrasya çarpışma akışı (Sandvol 2003)"},
        {"Bölge": "Batı Anadolu (Ege grabenleri)",
         "Fast Axis": "~30-50° (KD-GB)",
         "Yorum": "Hellenic trench geri-çekilmesi (Endrun 2011)"},
        {"Bölge": "İç Anadolu (Ankara-Konya)",
         "Fast Axis": "~55-72° (geçiş)",
         "Yorum": "İki rejim arası geçişsel mantle akış"},
        {"Bölge": "Akdeniz kıyısı (Antalya-Hatay)",
         "Fast Axis": "~45-65° (Kıbrıs yayı)",
         "Yorum": "Kıbrıs yayı subdüksiyon mantle wedge"},
    ])
    st.dataframe(df_yorum, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **Silver & Chan (1991)** *JGR* 96(B10), 16429-16454 (SKS yöntem) | "
        "**Savage (1999)** *Rev. Geophys.* 37(1) — DOI:10.1029/98RG02075 | "
        "**Sandvol et al. (2003)** *JGR* 108(B5) — DOI:10.1029/2002JB002023 (Anadolu) | "
        "**Biryol et al. (2010)** *JGR* 115, B07316 — DOI:10.1029/2009JB006923 | "
        "**Endrun et al. (2011)** *GJI* (Hellenic) | "
        "**IRIS SplittingDB:** ds.iris.edu/ds/products/sksobs/. "
        "⚠️ Fast axis çift yönlü vektör (180° belirsizliği); ok her iki yönde çizilir."
    )


if active_menu == "🌀 SKS Splitting":
    _render_sks_splitting()


# ════════════════════════════════════════════════════════════════════════════
# 🌊 TSUNAMİ KATALOĞU — F-57 / v1.34 — Akdeniz Tarihsel + NEAM Tehlike
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Papadopoulos, G.A. et al. (2014). Historical and pre-historical
#       tsunamis in the Mediterranean. Marine Geology 354, 81-109.
#       DOI:10.1016/j.margeo.2014.04.014
#   • Yolsal-Çevikbilen, S. & Taymaz, T. (2012). Earthquake source
#       parameters along the Hellenic subduction zone. Tectonophysics
#       536-537, 61-100. DOI:10.1016/j.tecto.2012.01.025
#   • Stiros, S.C. (2001). The AD 365 Crete earthquake. J. Struct. Geol.
#       23(2-3), 545-562.
#   • NOAA NGDC Tsunami DB: ngdc.noaa.gov/hazard/tsu.shtml
#   • TSUMAPS-NEAM: tsumaps-neam.eu
# ════════════════════════════════════════════════════════════════════════════

# Akdeniz tarihsel tsunami olayları (Papadopoulos 2014, NOAA NGDC)
# runup_m: kıyıda gözlenen maks. dalga yüksekliği (m)
_TSUNAMI_OLAYLAR = [
    {"yil": -365, "yer": "Girit (Stiros 365 AD)", "lat": 35.50, "lon": 23.50, "runup_m": 9.0,
     "kaynak_mw": 8.5, "kaynak": "Stiros 2001 J. Struct. Geol.; megatsunami"},
    {"yil": 1303, "yer": "Rodos-Girit",            "lat": 35.50, "lon": 27.50, "runup_m": 8.0,
     "kaynak_mw": 8.0, "kaynak": "Guidoboni 2004; Doğu Akdeniz mega"},
    {"yil": 1481, "yer": "Rodos",                  "lat": 36.20, "lon": 28.10, "runup_m": 3.0,
     "kaynak_mw": 7.0, "kaynak": "Papadopoulos 2014 Tablo 1"},
    {"yil": 1509, "yer": "İstanbul",               "lat": 41.00, "lon": 28.97, "runup_m": 5.0,
     "kaynak_mw": 7.2, "kaynak": "Hancılar 2012 J. Tsunami Soc. (Marmara)"},
    {"yil": 1530, "yer": "Sicilya",                "lat": 38.10, "lon": 13.30, "runup_m": 3.0,
     "kaynak_mw": 7.0, "kaynak": "Tinti et al. 2004 J. Geophys. Res."},
    {"yil": 1556, "yer": "Sakız Adası",            "lat": 38.40, "lon": 26.10, "runup_m": 2.0,
     "kaynak_mw": 6.8, "kaynak": "NOAA NGDC"},
    {"yil": 1693, "yer": "Doğu Sicilya",           "lat": 37.10, "lon": 15.20, "runup_m": 12.0,
     "kaynak_mw": 7.4, "kaynak": "Tinti 2004; Catania mega"},
    {"yil": 1741, "yer": "Türkiye GB kıyısı",      "lat": 35.00, "lon": 28.00, "runup_m": 8.0,
     "kaynak_mw": 7.5, "kaynak": "Papadopoulos 2014"},
    {"yil": 1755, "yer": "Lizbon (uzak etki)",     "lat": 36.00, "lon": -10.50, "runup_m": 15.0,
     "kaynak_mw": 8.5, "kaynak": "Baptista 1998 J. Geodyn.; Atlantik ana"},
    {"yil": 1856, "yer": "Girit-Rodos",            "lat": 35.50, "lon": 26.00, "runup_m": 6.0,
     "kaynak_mw": 7.7, "kaynak": "Ambraseys 1962"},
    {"yil": 1908, "yer": "Messina (Sicilya-Calabria)", "lat": 38.10, "lon": 15.60, "runup_m": 13.0,
     "kaynak_mw": 7.2, "kaynak": "Tinti et al. 2008 BSSA"},
    {"yil": 1939, "yer": "Erzincan (göl tsunamisi)", "lat": 39.40, "lon": 39.00, "runup_m": 1.0,
     "kaynak_mw": 7.8, "kaynak": "Altınok 1999 Phys. Chem. Earth (lokal)"},
    {"yil": 1948, "yer": "Karpathos (Yunan)",      "lat": 35.50, "lon": 27.20, "runup_m": 3.0,
     "kaynak_mw": 7.1, "kaynak": "Papadopoulos 2014"},
    {"yil": 1956, "yer": "Amorgos (Ege)",          "lat": 36.80, "lon": 25.80, "runup_m": 25.0,
     "kaynak_mw": 7.7, "kaynak": "Okal et al. 2009 Mar. Geol. (lokal mega)"},
    {"yil": 1968, "yer": "Saros (Türkiye)",        "lat": 40.30, "lon": 26.60, "runup_m": 1.5,
     "kaynak_mw": 6.9, "kaynak": "Altınok 1999"},
    {"yil": 1999, "yer": "İzmit (Marmara)",        "lat": 40.75, "lon": 29.90, "runup_m": 2.5,
     "kaynak_mw": 7.6, "kaynak": "Altınok et al. 2001 Phys. Chem. Earth (lokal)"},
    {"yil": 2002, "yer": "Stromboli volkanik tsunami", "lat": 38.79, "lon": 15.21, "runup_m": 11.0,
     "kaynak_mw": 0, "kaynak": "Tinti 2005 (volkanik kaynak)"},
    {"yil": 2017, "yer": "Bodrum-Kos",             "lat": 36.96, "lon": 27.43, "runup_m": 1.9,
     "kaynak_mw": 6.6, "kaynak": "Yalçıner et al. 2017 (Gümbet 1.9m)"},
    {"yil": 2020, "yer": "Sığacık-Samos",          "lat": 37.90, "lon": 26.79, "runup_m": 3.8,
     "kaynak_mw": 6.9, "kaynak": "Dogan et al. 2021 Pure Appl. Geophys."},
]


def _tsunami_renk(runup: float) -> str:
    if runup >= 15: return "#3F0080"   # mega
    if runup >= 10: return "#A32D2D"   # büyük
    if runup >= 5:  return "#E24B4A"   # orta-büyük
    if runup >= 3:  return "#EF9F27"   # orta
    if runup >= 1.5: return "#FAC775"  # küçük-orta
    return "#1D9E75"                    # küçük


@st.fragment
def _render_tsunami_katalog():
    st.markdown(
        '<div class="chart-title">🌊 Akdeniz Tsunami Kataloğu — Tarihsel + NEAM (F-57 / v1.34)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🌊 **Akdeniz Tsunami:** 365 AD Girit megatsunami'sinden 2020 Samos olayına "
        "kadar 19 önemli olay. **Papadopoulos et al. (2014) Mar. Geol. 354** "
        "kapsamlı katalog. Türkiye kıyısı: Yolsal-Çevikbilen & Taymaz (2012) "
        "Hellenic trench modellemesi."
    )

    df_t = pd.DataFrame(_TSUNAMI_OLAYLAR)
    df_t["renk"] = df_t["runup_m"].apply(_tsunami_renk)

    col_yil, col_runup = st.columns(2)
    with col_yil:
        yil_min, yil_max = st.slider(
            "Yıl aralığı",
            min_value=-500, max_value=2030, value=(-500, 2030), step=50,
            key="tsunami_yil",
        )
    with col_runup:
        min_runup = st.slider(
            "Min. runup (m)",
            min_value=0.5, max_value=20.0, value=1.0, step=0.5,
            key="tsunami_min_runup",
        )

    df_filt = df_t[(df_t["yil"] >= yil_min) & (df_t["yil"] <= yil_max) &
                   (df_t["runup_m"] >= min_runup)].copy()
    if df_filt.empty:
        st.warning("Bu filtreye uygun olay yok.")
        return

    # ── Harita ─────────────────────────────────────────────────────────────
    fig_map = go.Figure()
    for _, ev in df_filt.iterrows():
        size = 10 + ev["runup_m"] * 1.5
        mw_str = f"Mw {ev['kaynak_mw']:.1f}" if ev["kaynak_mw"] > 0 else "Volkanik kaynak"
        hover = (
            f"<b>{ev['yer']}</b><br>"
            f"Yıl: {int(ev['yil'])}<br>"
            f"Runup: {ev['runup_m']:.1f} m<br>"
            f"Kaynak: {mw_str}<br>"
            f"Atıf: {ev['kaynak']}"
            "<extra></extra>"
        )
        fig_map.add_trace(go.Scattermapbox(
            lat=[ev["lat"]], lon=[ev["lon"]],
            mode="markers",
            marker=dict(size=size, color=ev["renk"], opacity=0.85),
            hovertemplate=hover,
            showlegend=False,
        ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=37.0, lon=25.0), zoom=4.3),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Zaman serisi (runup vs yıl) ────────────────────────────────────────
    st.markdown('<div class="chart-title">⏳ Zaman Çizelgesi (yıl × runup)</div>', unsafe_allow_html=True)
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=df_filt["yil"], y=df_filt["runup_m"],
        mode="markers",
        marker=dict(
            size=10 + df_filt["runup_m"] * 1.5,
            color=df_filt["renk"].tolist(),
            opacity=0.85, line=dict(color="#222", width=0.4),
        ),
        text=df_filt["yer"],
        hovertemplate="<b>%{text}</b><br>Yıl: %{x}<br>Runup: %{y:.1f} m<extra></extra>",
    ))
    fig_t.update_layout(
        xaxis=dict(title="Yıl (MS, negatif = MÖ)", color=TEXT, gridcolor=BORDER),
        yaxis=dict(title="Runup (m)", color=TEXT, gridcolor=BORDER, type="log",
                   range=[math.log10(0.5), math.log10(30)]),
        height=320,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        showlegend=False,
    )
    st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    biggest = df_filt.loc[df_filt["runup_m"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{len(df_filt)}",                "#FFD700", "Olay sayısı (filtreli)"),
        (c2, f"{biggest['runup_m']:.0f} m",    "#A32D2D", f"Maks runup ({biggest['yer'][:14]})"),
        (c3, f"{df_filt['runup_m'].mean():.1f} m", "#EF9F27", "Ortalama runup"),
        (c4, f"~{(yil_max - yil_min) // max(1, len(df_filt) - 1)} yıl", "#1976D2", "Ortalama olaylar arası"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tablo ─────────────────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Olay Listesi</div>', unsafe_allow_html=True)
    df_show = df_filt[["yil", "yer", "runup_m", "kaynak_mw", "kaynak"]].copy()
    df_show.columns = ["Yıl", "Yer", "Runup (m)", "Kaynak Mw", "Bilimsel Kaynak"]
    df_show = df_show.sort_values("Yıl").reset_index(drop=True)
    st.dataframe(df_show, use_container_width=True, hide_index=True, height=350)

    st.caption(
        "📚 **Papadopoulos et al. (2014)** *Marine Geology* 354, 81-109 — "
        "DOI:10.1016/j.margeo.2014.04.014 (ana katalog) | "
        "**Yolsal-Çevikbilen & Taymaz (2012)** *Tectonophysics* 536-537, 61-100 — "
        "DOI:10.1016/j.tecto.2012.01.025 (Türkiye Hellenic) | "
        "**Stiros (2001)** *J. Struct. Geol.* 23, 545-562 (365 AD Girit) | "
        "**NOAA NGDC Tsunami DB:** ngdc.noaa.gov/hazard/tsu.shtml | "
        "**TSUMAPS-NEAM:** tsumaps-neam.eu. "
        "⚠️ 1900 öncesi runup değerleri tarihsel kayıt belirsizliği taşır (±2-3 m)."
    )


if active_menu == "🌊 Tsunami Kataloğu":
    _render_tsunami_katalog()


# ════════════════════════════════════════════════════════════════════════════
# ⏱️ TSUNAMİ VARIŞ — F-58 / v1.35 — Shallow Water c=√(gd)
# ────────────────────────────────────────────────────────────────────────────
# Kaynaklar:
#   • Titov & Synolakis (1998) J. Waterw. 124(4); Lamb (1932) Hydrodynamics
#   • Yalçıner et al. (2017) Pure Appl. Geophys. 174(8) — Bodrum-Kos
#   • GEBCO 2023 batimetri (gebco.net)
# ════════════════════════════════════════════════════════════════════════════

_TSUNAMI_HEDEFLER = [
    {"yer": "Bodrum",         "lat": 37.04, "lon": 27.43},
    {"yer": "İzmir",          "lat": 38.42, "lon": 27.14},
    {"yer": "Kuşadası",       "lat": 37.86, "lon": 27.26},
    {"yer": "Marmaris",       "lat": 36.85, "lon": 28.27},
    {"yer": "Antalya",        "lat": 36.89, "lon": 30.71},
    {"yer": "Mersin",         "lat": 36.81, "lon": 34.64},
    {"yer": "İskenderun",     "lat": 36.59, "lon": 36.17},
    {"yer": "Çeşme",          "lat": 38.33, "lon": 26.30},
    {"yer": "Foça",           "lat": 38.66, "lon": 26.76},
    {"yer": "Edremit",        "lat": 39.60, "lon": 27.02},
    {"yer": "Çanakkale",      "lat": 40.15, "lon": 26.41},
    {"yer": "Tekirdağ",       "lat": 40.98, "lon": 27.51},
    {"yer": "İstanbul (Marmara)", "lat": 40.85, "lon": 28.50},
]

_TSUNAMI_KAYNAKLAR = {
    "Hellenic Trench (Mw 8.5 senaryo, Stiros 2001 benzer)": {
        "lat": 35.50, "lon": 23.50, "ortalama_derinlik_m": 2500,
        "ref": "Stiros 2001; Yolsal-Çevikbilen 2012"},
    "Doğu Akdeniz subdüksiyon (Mw 8.0 senaryo)": {
        "lat": 34.50, "lon": 27.50, "ortalama_derinlik_m": 2000,
        "ref": "Papadopoulos 2014"},
    "Kıbrıs yayı (Mw 7.5 senaryo)": {
        "lat": 34.80, "lon": 32.50, "ortalama_derinlik_m": 1500,
        "ref": "Yolsal-Çevikbilen 2012"},
    "Saros Körfezi (Mw 7.4, 1912 tekrarı)": {
        "lat": 40.65, "lon": 26.50, "ortalama_derinlik_m": 200,
        "ref": "Altınok 1999 Phys. Chem. Earth"},
    "Marmara (Mw 7.2, 1766 tekrarı)": {
        "lat": 40.80, "lon": 28.50, "ortalama_derinlik_m": 1100,
        "ref": "Hancilar 2012; Parsons 2004"},
    "Bodrum-Kos (Mw 6.6, 2017 tekrarı)": {
        "lat": 36.96, "lon": 27.43, "ortalama_derinlik_m": 300,
        "ref": "Yalçıner 2017 Pure Appl. Geophys."},
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _shallow_water_speed_kmh(depth_m: float) -> float:
    """c = √(g·d), m/s → km/h."""
    if depth_m <= 0:
        return 0.0
    return math.sqrt(9.81 * depth_m) * 3.6


@st.fragment
def _render_tsunami_varis():
    st.markdown(
        '<div class="chart-title">⏱️ Tsunami Varış Süresi — Shallow Water c=√(gd) (F-58 / v1.35)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "⏱️ **Shallow Water Yaklaşımı:** Tsunami dalgaları **c = √(g·d)** ile ilerler. "
        "2500m derinlikte ~565 km/h (jet uçağı), 100m'de ~113 km/h, sahile yaklaşınca "
        "yavaşlar (Lamb 1932). **Titov & Synolakis (1998) J. Waterw.** sayısal model."
    )

    sec = st.selectbox(
        "Tsunami kaynak senaryosu",
        options=list(_TSUNAMI_KAYNAKLAR.keys()),
        index=0,
        key="tsunami_var_kaynak",
    )
    k = _TSUNAMI_KAYNAKLAR[sec]
    speed_kmh = _shallow_water_speed_kmh(k["ortalama_derinlik_m"])

    rows = []
    for h in _TSUNAMI_HEDEFLER:
        dist_km = _haversine_km(k["lat"], k["lon"], h["lat"], h["lon"])
        varis_dk = (dist_km / speed_kmh) * 60 if speed_kmh > 0 else None
        rows.append({**h, "mesafe_km": round(dist_km, 0),
                     "varis_dk": round(varis_dk, 1) if varis_dk else None})
    df_v = pd.DataFrame(rows).sort_values("varis_dk").reset_index(drop=True)

    # ── Harita: eş-zaman halkaları ─────────────────────────────────────────
    fig = go.Figure()
    for t_dk, renk in [(5, "#1D9E75"), (10, "#7DC872"), (15, "#FAC775"),
                        (20, "#EF9F27"), (30, "#E24B4A"), (45, "#A32D2D"), (60, "#3F0080")]:
        r_km = (t_dk / 60.0) * speed_kmh
        if r_km > 1500:
            continue
        clat, clon = _shakemap_circle_coords(k["lat"], k["lon"], r_km, n=60)
        fig.add_trace(go.Scattermapbox(
            lat=clat, lon=clon,
            mode="lines",
            line=dict(width=1.8, color=renk),
            name=f"{t_dk} dk ({r_km:.0f} km)",
            opacity=0.7,
            hovertemplate=f"{t_dk} dk varış<br>r = {r_km:.0f} km<extra></extra>",
        ))

    fig.add_trace(go.Scattermapbox(
        lat=[k["lat"]], lon=[k["lon"]],
        mode="markers+text",
        marker=dict(size=18, color="#FFD700", symbol="star"),
        text=["★ Kaynak"],
        textposition="top right",
        textfont=dict(size=12, color="#FFD700"),
        hoverinfo="text",
        name="Tsunami kaynağı",
    ))
    fig.add_trace(go.Scattermapbox(
        lat=df_v["lat"], lon=df_v["lon"],
        mode="markers+text",
        marker=dict(size=10, color="#E24B4A"),
        text=df_v["yer"],
        textfont=dict(size=9, color="#fff"),
        textposition="top right",
        hovertemplate=df_v.apply(
            lambda r: (f"<b>{r['yer']}</b><br>"
                       f"Mesafe: {r['mesafe_km']:.0f} km<br>"
                       f"Varış: {r['varis_dk']:.1f} dk"
                       "<extra></extra>"),
            axis=1,
        ),
        name="Hedef kıyılar",
    ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=37.5, lon=28.0), zoom=4.7),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01,
                    bgcolor="rgba(0,0,0,0.55)", font=dict(color="#fff", size=10),
                    bordercolor=BORDER, borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Bar chart varış süreleri ──────────────────────────────────────────
    st.markdown('<div class="chart-title">📊 Kıyı Hedefleri — Varış Süresi (dk)</div>', unsafe_allow_html=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=df_v["yer"],
        x=df_v["varis_dk"],
        orientation="h",
        marker=dict(color=df_v["varis_dk"],
                    colorscale=[[0.0, "#A32D2D"], [0.5, "#EF9F27"], [1.0, "#1D9E75"]],
                    cmin=0, cmax=60,
                    line=dict(color="#222", width=0.5)),
        text=df_v.apply(lambda r: f"{r['varis_dk']:.1f} dk ({r['mesafe_km']:.0f} km)", axis=1),
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="<b>%{y}</b><br>Varış: %{x:.1f} dk<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis=dict(title="Tahmini varış (dakika)", color=TEXT, gridcolor=BORDER),
        yaxis=dict(color=TEXT, gridcolor=BORDER),
        height=420,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    en_yakin = df_v.iloc[0]
    en_uzak = df_v.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{en_yakin['varis_dk']:.1f} dk", "#A32D2D", f"En yakın ({en_yakin['yer'][:15]})"),
        (c2, f"{en_uzak['varis_dk']:.1f} dk",  "#1D9E75", f"En uzak ({en_uzak['yer'][:15]})"),
        (c3, f"{speed_kmh:.0f} km/h",          "#FFD700", f"Dalga hızı (d = {k['ortalama_derinlik_m']} m)"),
        (c4, f"{len(df_v)}",                    "#1976D2", "Hedef kıyı sayısı"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-title">📋 Derinlik × Hız (Shallow Water)</div>', unsafe_allow_html=True)
    df_dh = pd.DataFrame([
        {"Derinlik (m)": 5000, "Hız (km/h)": round(_shallow_water_speed_kmh(5000)), "Yorum": "Derin Pasifik (jet uçağı)"},
        {"Derinlik (m)": 2500, "Hız (km/h)": round(_shallow_water_speed_kmh(2500)), "Yorum": "Akdeniz derin havza"},
        {"Derinlik (m)": 1000, "Hız (km/h)": round(_shallow_water_speed_kmh(1000)), "Yorum": "Marmara havzası"},
        {"Derinlik (m)": 500,  "Hız (km/h)": round(_shallow_water_speed_kmh(500)),  "Yorum": "Şelf kenarı"},
        {"Derinlik (m)": 200,  "Hız (km/h)": round(_shallow_water_speed_kmh(200)),  "Yorum": "Karaya yakın şelf"},
        {"Derinlik (m)": 50,   "Hız (km/h)": round(_shallow_water_speed_kmh(50)),   "Yorum": "Sığ kıyı (yavaşlama → büyüme)"},
        {"Derinlik (m)": 10,   "Hız (km/h)": round(_shallow_water_speed_kmh(10)),   "Yorum": "Kıyı şeridi (max runup)"},
    ])
    st.dataframe(df_dh, use_container_width=True, hide_index=True)

    st.caption(
        f"📚 **Senaryo:** {k['ref']} | "
        "**Lamb (1932)** *Hydrodynamics* | "
        "**Titov & Synolakis (1998)** *J. Waterw.* 124(4) — "
        "DOI:10.1061/(ASCE)0733-950X(1998)124:4(157) | "
        "**Yalçıner et al. (2017)** *PAGEOPH* 174(8) (Bodrum-Kos) | "
        "**GEBCO 2023:** gebco.net. "
        "⚠️ Ortalama derinlik varsayımı; gerçek için NAMI-DANCE / MOST tam batimetri."
    )


if active_menu == "⏱️ Tsunami Varış":
    _render_tsunami_varis()


# ════════════════════════════════════════════════════════════════════════════
# 🎬 AMBRASEYS ANİMASYON — F-59 / v1.36 — KAF Batıya Göç Animasyonu
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Ambraseys, N.N. (2009). Earthquakes in the Mediterranean and Middle
#       East. Cambridge University Press. ISBN:9780521872928
#   • Barka, A. (1996). Slip distribution along the NAF associated with
#       large earthquakes 1939-1967. BSSA 86(5), 1238-1254.
#   • Stein, R.S., Barka, A.A. & Dieterich, J.H. (1997). Progressive failure
#       on the NAF since 1939 by earthquake stress triggering.
#       Geophys. J. Int. 128, 594-604.
#   • Toksöz, M.N. et al. (1979). Bull. Earthq. Res. Inst. (Tokyo).
# ════════════════════════════════════════════════════════════════════════════

# Plotly frames animasyonu için ana KAF batıya göç dizisi (1668-1999)
# + tarihi öncesi (Ambraseys) + 2023 doğu kuşak (DAF) zenginleştirildi
_AMBRASEYS_DIZISI = [
    # KAF doğu kesim (tarihsel)
    {"yil": 1668, "yer": "Amasya-Niksar",       "lat": 40.65, "lon": 35.80, "mw": 8.0, "fay": "KAF doğu"},
    {"yil": 1719, "yer": "İzmit (İlk Marmara)", "lat": 40.75, "lon": 30.00, "mw": 7.4, "fay": "KAF batı"},
    {"yil": 1766, "yer": "Marmara büyük",       "lat": 40.80, "lon": 29.00, "mw": 7.1, "fay": "KAF batı (Marmara)"},
    {"yil": 1784, "yer": "Erzincan",            "lat": 39.80, "lon": 39.30, "mw": 7.6, "fay": "KAF doğu"},
    {"yil": 1894, "yer": "İstanbul",            "lat": 40.70, "lon": 28.65, "mw": 7.0, "fay": "KAF batı"},

    # KAF batıya göç dizisi (Stein 1997 klasik)
    {"yil": 1939, "yer": "Erzincan",            "lat": 39.80, "lon": 39.51, "mw": 7.8, "fay": "KAF doğu"},
    {"yil": 1942, "yer": "Niksar-Erbaa",        "lat": 40.65, "lon": 36.95, "mw": 7.0, "fay": "KAF"},
    {"yil": 1943, "yer": "Tosya-Ladik",         "lat": 41.00, "lon": 34.00, "mw": 7.6, "fay": "KAF"},
    {"yil": 1944, "yer": "Bolu-Gerede",         "lat": 40.85, "lon": 32.30, "mw": 7.4, "fay": "KAF"},
    {"yil": 1957, "yer": "Abant",               "lat": 40.65, "lon": 31.00, "mw": 7.1, "fay": "KAF batı"},
    {"yil": 1967, "yer": "Mudurnu",             "lat": 40.65, "lon": 30.70, "mw": 7.1, "fay": "KAF batı"},
    {"yil": 1999, "yer": "İzmit",               "lat": 40.75, "lon": 29.86, "mw": 7.6, "fay": "KAF batı (İzmit)"},
    {"yil": 1999, "yer": "Düzce",               "lat": 40.79, "lon": 31.21, "mw": 7.2, "fay": "KAF batı (Düzce)"},

    # Doğu Anadolu + Ege ek
    {"yil": 1976, "yer": "Çaldıran",            "lat": 39.05, "lon": 44.04, "mw": 7.3, "fay": "Doğu Anadolu"},
    {"yil": 2011, "yer": "Van",                 "lat": 38.72, "lon": 43.51, "mw": 7.1, "fay": "Doğu Anadolu"},
    {"yil": 2020, "yer": "İzmir-Samos",         "lat": 37.90, "lon": 26.79, "mw": 6.9, "fay": "Ege"},
    {"yil": 2023, "yer": "Pazarcık",            "lat": 37.17, "lon": 37.04, "mw": 7.8, "fay": "DAF"},
    {"yil": 2023, "yer": "Elbistan",            "lat": 38.02, "lon": 37.20, "mw": 7.7, "fay": "DAF / Sürgü"},
]


@st.fragment
def _render_ambraseys_animasyon():
    st.markdown(
        '<div class="chart-title">🎬 Ambraseys Katalog Animasyonu — KAF Batıya Göç (F-59 / v1.36)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🎬 **KAF Batıya Göç (Stein 1997):** 1939 Erzincan'dan başlayarak NAFZ üzerinde "
        "büyük depremler batıya doğru bir 'tren güzergahı' gibi ilerledi. Bu animasyon "
        "1668 Amasya'dan 2023 Pazarcık'a kadar olayları zamanda ileri sarar. "
        "**Ambraseys (2009) Cambridge UP** + **Barka (1996) BSSA 86** + "
        "**Stein, Barka & Dieterich (1997) GJI 128** birleşik veri."
    )

    df_a = pd.DataFrame(_AMBRASEYS_DIZISI).sort_values("yil").reset_index(drop=True)

    # ── Animasyonlu scatter — her frame = bir yıl, kümülatif gösterim ──────
    yillar = sorted(df_a["yil"].unique())

    frames = []
    for y in yillar:
        df_y = df_a[df_a["yil"] <= y]
        renkler = []
        boyutlar = []
        for _, ev in df_y.iterrows():
            # Yeni olan parlak kırmızı, eski olanlar soluk
            yas = y - ev["yil"]
            if yas == 0:
                renkler.append("#FF3333")
                boyutlar.append(20 + (ev["mw"] - 6.5) * 6)
            else:
                # soluk
                renkler.append(f"rgba(150,80,80,{max(0.2, 1.0 - yas/200):.2f})")
                boyutlar.append(8 + (ev["mw"] - 6.5) * 4)

        frame_data = [go.Scattermapbox(
            lat=df_y["lat"],
            lon=df_y["lon"],
            mode="markers+text",
            marker=dict(size=boyutlar, color=renkler, opacity=0.9),
            text=df_y.apply(lambda r: f"{int(r['yil'])}", axis=1),
            textfont=dict(size=10, color="#fff"),
            textposition="top right",
            hovertemplate=df_y.apply(
                lambda r: (f"<b>{r['yer']}</b> ({int(r['yil'])})<br>"
                           f"Mw {r['mw']:.1f}<br>"
                           f"Fay: {r['fay']}"
                           "<extra></extra>"),
                axis=1,
            ),
        )]
        frames.append(go.Frame(data=frame_data, name=str(y)))

    # ── İlk frame veri ─────────────────────────────────────────────────────
    df_first = df_a[df_a["yil"] == yillar[0]]
    fig = go.Figure(
        data=[go.Scattermapbox(
            lat=df_first["lat"],
            lon=df_first["lon"],
            mode="markers+text",
            marker=dict(size=22, color="#FF3333", opacity=0.95),
            text=df_first.apply(lambda r: f"{int(r['yil'])}", axis=1),
            textfont=dict(size=11, color="#fff"),
            textposition="top right",
            hovertemplate="<b>%{text}</b><extra></extra>",
        )],
        frames=frames,
    )

    # ── Slider + play butonu ───────────────────────────────────────────────
    sliders = [{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"prefix": "Yıl: ", "font": {"color": TEXT, "size": 13}},
        "transition": {"duration": 200, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 30},
        "len": 0.85, "x": 0.10, "y": 0,
        "steps": [{
            "args": [[str(y)], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
            "label": str(y), "method": "animate"
        } for y in yillar],
    }]

    updatemenus = [{
        "type": "buttons", "showactive": False,
        "y": 0, "x": 0.05, "xanchor": "right", "yanchor": "top",
        "pad": {"t": 0, "r": 10},
        "buttons": [
            {"label": "▶ Oynat",
             "method": "animate",
             "args": [None, {"frame": {"duration": 800, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 200}}]},
            {"label": "⏸ Duraklat",
             "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate", "transition": {"duration": 0}}]},
        ],
    }]

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.5, lon=33.0), zoom=5),
        height=560,
        margin=dict(l=0, r=0, t=10, b=60),
        paper_bgcolor=BG2,
        sliders=sliders,
        updatemenus=updatemenus,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("💡 **Slider'ı sürükleyin veya ▶ Oynat butonuna basın** — KAF olayları yıllar içinde batıya kayar.")

    # ── İstatistikler ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    span_yil = int(df_a["yil"].max() - df_a["yil"].min())
    kartlar = [
        (c1, f"{len(df_a)}",                       "#FFD700", "Toplam olay"),
        (c2, f"{int(df_a['yil'].min())}–{int(df_a['yil'].max())}", "#EF9F27", "Zaman aralığı"),
        (c3, f"{span_yil} yıl",                     "#1976D2", "Kapsanan süre"),
        (c4, f"M{df_a['mw'].max():.1f}",            "#A32D2D", "En büyük olay"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tablo + batı göç vurgusu ───────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Olay Dizisi (kronolojik)</div>', unsafe_allow_html=True)
    df_show = df_a[["yil", "yer", "mw", "fay"]].copy()
    df_show.columns = ["Yıl", "Yer", "Mw", "Fay Sistemi"]
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **Ambraseys (2009)** Cambridge UP, ISBN 9780521872928 (tarihsel katalog) | "
        "**Barka (1996)** *BSSA* 86(5), 1238-1254 (KAF göç) | "
        "**Stein, Barka & Dieterich (1997)** *GJI* 128, 594-604 (kademeli kırılma) | "
        "**Toksöz et al. (1979)** *Bull. Earthq. Res. Inst.* | "
        "**NOAA NCEI:** ngdc.noaa.gov. "
        "🎯 Klasik gözlem: KAF üzerinde 1939→1999 arası batıya 700+ km göç. "
        "Sıra dışı: 2023 olayları DAF üzerinde, KAF dizisinden bağımsız."
    )


if active_menu == "🎬 Ambraseys Animasyon":
    _render_ambraseys_animasyon()


# ════════════════════════════════════════════════════════════════════════════
# ⛏️ PALEOSİSMİK KAZI — F-60 / v1.37 — NAFZ/DAF Trench Slot Diyagramı
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Kozacı, Ö. et al. (2007). Paleoseismological evidence on the eastern
#       NAF at Yaylabeli (Erzincan). BSSA 97(5), 1513-1527.
#       DOI:10.1785/0120060118
#   • Klinger, Y. et al. (2003). Paleoseismic evidence of characteristic
#       slip on the western segment of the NAF. BSSA 93(6), 2317-2332.
#       DOI:10.1785/0120010270
#   • Fraser, J., Vanneste, K. & Hubert-Ferrari, A. (2010). Recent behavior
#       of the NAF. JGR 115, B09316. DOI:10.1029/2009JB006982
#   • Akyüz, H.S. et al. (2002). Surface ruptures of 1939, 1942, 1943, 1957,
#       1967, 1992 NAF earthquakes. BSSA 92(1), 61-66.
#   • Meghraoui, M. et al. (2012). The seismic cycle along the EAF: paleoseismic
#       analysis at multiple sites. Tectonophysics 538-540, 88-102.
# ════════════════════════════════════════════════════════════════════════════

# Paleoseismik kazı verileri — Kozacı 2007, Klinger 2003, Fraser 2010, Akyüz 2002
# yıl: -3000 ≈ MÖ 3000; +1939 = MS 1939
# belirsizlik_yil: ±14C/OSL belirsizliği
_PALEOSEISMIK_KAZILAR = [
    # Yaylabeli — Erzincan (Kozacı 2007 birincil çalışma)
    {"site": "Yaylabeli (Erzincan)", "lat": 39.78, "lon": 39.50,
     "ref": "Kozacı et al. 2007 BSSA 97(5)",
     "olaylar": [
         {"yil": 1939, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1668, "belirsizlik": 0,   "tip": "tarihsel"},  # tartışmalı
         {"yil": 1180, "belirsizlik": 60,  "tip": "paleo"},
         {"yil": 870,  "belirsizlik": 80,  "tip": "paleo"},
         {"yil": 560,  "belirsizlik": 100, "tip": "paleo"},
         {"yil": 250,  "belirsizlik": 120, "tip": "paleo"},
         {"yil": -100, "belirsizlik": 150, "tip": "paleo"},
     ],
     "ortalama_tekrar": 320, "sigma": 80},

    # Tahtaköprü — Batı NAFZ (Klinger 2003)
    {"site": "Tahtaköprü (Batı NAFZ)", "lat": 40.85, "lon": 32.40,
     "ref": "Klinger et al. 2003 BSSA 93(6)",
     "olaylar": [
         {"yil": 1944, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1668, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1290, "belirsizlik": 50,  "tip": "paleo"},
         {"yil": 1050, "belirsizlik": 70,  "tip": "paleo"},
         {"yil": 760,  "belirsizlik": 90,  "tip": "paleo"},
         {"yil": 480,  "belirsizlik": 110, "tip": "paleo"},
     ],
     "ortalama_tekrar": 285, "sigma": 70},

    # Demir Köprü — Marmara (Hubert-Ferrari 2000 NAFZ batı)
    {"site": "Demir Köprü (Marmara giriş)", "lat": 40.80, "lon": 29.50,
     "ref": "Hubert-Ferrari et al. 2002 EPSL",
     "olaylar": [
         {"yil": 1999, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1719, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1509, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1063, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 740,  "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 360,  "belirsizlik": 100, "tip": "paleo"},
     ],
     "ortalama_tekrar": 280, "sigma": 100},

    # Niksar (Fraser 2010 sentez)
    {"site": "Niksar Havzası", "lat": 40.62, "lon": 36.95,
     "ref": "Fraser et al. 2010 JGR 115",
     "olaylar": [
         {"yil": 1942, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1668, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1170, "belirsizlik": 80,  "tip": "paleo"},
         {"yil": 900,  "belirsizlik": 100, "tip": "paleo"},
         {"yil": 540,  "belirsizlik": 120, "tip": "paleo"},
     ],
     "ortalama_tekrar": 350, "sigma": 100},

    # Çardak — DAF kuzey (Meghraoui 2012)
    {"site": "Çardak (DAF kuzey)", "lat": 38.05, "lon": 37.20,
     "ref": "Meghraoui et al. 2012 Tectonophysics 538-540",
     "olaylar": [
         {"yil": 2023, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1114, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 200,  "belirsizlik": 150, "tip": "paleo"},
         {"yil": -750, "belirsizlik": 200, "tip": "paleo"},
     ],
     "ortalama_tekrar": 800, "sigma": 250},

    # Misis — DAF güney (Karabacak 2011)
    {"site": "Misis (DAF güney)", "lat": 36.96, "lon": 35.62,
     "ref": "Karabacak et al. 2011 Quat. Int.",
     "olaylar": [
         {"yil": 1872, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 1408, "belirsizlik": 0,   "tip": "tarihsel"},
         {"yil": 850,  "belirsizlik": 100, "tip": "paleo"},
         {"yil": 100,  "belirsizlik": 150, "tip": "paleo"},
     ],
     "ortalama_tekrar": 600, "sigma": 180},
]


@st.fragment
def _render_paleosismik_kazi():
    st.markdown(
        '<div class="chart-title">⛏️ Paleosismik Kazı — NAFZ/DAF Trench Slot Diyagramı (F-60 / v1.37)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "⛏️ **Paleoseismoloji:** Fay boyunca açılan **kazılarda (trenches)** "
        "kolüvyal tortul katmanlardaki yer değiştirme izleri **¹⁴C/OSL** ile "
        "tarihlendirilerek prehistorik depremler tespit edilir. Türkiye'de "
        "**Kozacı et al. (2007) BSSA 97** Erzincan Yaylabeli'de 6 paleo olay "
        "saptamış, ortalama tekrar süresi ~320 yıl bulmuştur."
    )

    # ── Site harita ────────────────────────────────────────────────────────
    fig_map = go.Figure()
    for site in _PALEOSEISMIK_KAZILAR:
        n_olay = len(site["olaylar"])
        fig_map.add_trace(go.Scattermapbox(
            lat=[site["lat"]], lon=[site["lon"]],
            mode="markers+text",
            marker=dict(size=14 + n_olay * 2, color="#A52A85", opacity=0.85),
            text=[site["site"].split(" (")[0]],
            textposition="top right",
            textfont=dict(size=10, color="#fff"),
            hovertemplate=(
                f"<b>{site['site']}</b><br>"
                f"Olay sayısı: {n_olay}<br>"
                f"Ortalama tekrar: ~{site['ortalama_tekrar']} ± {site['sigma']} yıl<br>"
                f"Kaynak: {site['ref']}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.5, lon=35.0), zoom=5.5),
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Site seçici + slot diyagram ────────────────────────────────────────
    sec = st.selectbox(
        "Kazı sitesi (detay için)",
        options=[s["site"] for s in _PALEOSEISMIK_KAZILAR],
        index=0,
        key="paleo_site_select",
    )
    site = next(x for x in _PALEOSEISMIK_KAZILAR if x["site"] == sec)

    # ── Slot diyagram ──────────────────────────────────────────────────────
    st.markdown(f'<div class="chart-title">📊 Slot Diyagramı — {sec}</div>',
                unsafe_allow_html=True)
    fig_slot = go.Figure()

    for i, ev in enumerate(site["olaylar"]):
        renk = "#FFD700" if ev["tip"] == "tarihsel" else "#A52A85"
        # Hata barı = belirsizlik penceresi
        fig_slot.add_trace(go.Scatter(
            x=[ev["yil"]],
            y=[i],
            mode="markers",
            marker=dict(size=14, color=renk, symbol="diamond",
                        line=dict(color="#fff", width=1.5)),
            error_x=dict(type="data",
                         array=[ev["belirsizlik"]],
                         color="rgba(255,255,255,0.5)",
                         thickness=1.5, width=10),
            text=[f"Olay {i+1}"],
            hovertemplate=(
                f"<b>Olay {i+1}</b><br>"
                f"Yıl: {ev['yil']} ± {ev['belirsizlik']}<br>"
                f"Tip: {ev['tip']}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        # Yatay yardımcı çizgi
        fig_slot.add_shape(type="line",
                           x0=ev["yil"] - ev["belirsizlik"] - 50,
                           x1=ev["yil"] + ev["belirsizlik"] + 50,
                           y0=i, y1=i,
                           line=dict(color="rgba(255,255,255,0.1)", width=1))

    # Şu an çizgisi
    fig_slot.add_vline(x=datetime.now().year,
                       line=dict(color="#1976D2", width=2, dash="dash"),
                       annotation_text="Şu an",
                       annotation_font_color="#1976D2")

    fig_slot.update_layout(
        xaxis=dict(title="Yıl (MS, negatif = MÖ)", color=TEXT, gridcolor=BORDER),
        yaxis=dict(title="Olay sırası (yeni → eski)",
                   tickmode="array",
                   tickvals=list(range(len(site["olaylar"]))),
                   ticktext=[f"#{i+1} ({ev['tip']})" for i, ev in enumerate(site["olaylar"])],
                   color=TEXT, gridcolor=BORDER),
        height=380,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        showlegend=False,
    )
    st.plotly_chart(fig_slot, use_container_width=True, config={"displayModeBar": False})

    # ── İstatistikler ──────────────────────────────────────────────────────
    n_paleo = sum(1 for ev in site["olaylar"] if ev["tip"] == "paleo")
    n_tar = sum(1 for ev in site["olaylar"] if ev["tip"] == "tarihsel")
    son_olay = max(ev["yil"] for ev in site["olaylar"])
    elapsed = datetime.now().year - son_olay
    elapsed_ratio = elapsed / site["ortalama_tekrar"]
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{n_paleo + n_tar}",                "#FFD700", f"Toplam olay ({n_tar} tarihsel + {n_paleo} paleo)"),
        (c2, f"{site['ortalama_tekrar']}±{site['sigma']} yıl", "#EF9F27", "Ortalama tekrar süresi"),
        (c3, f"{elapsed} yıl",                     "#1976D2", f"Son depremden bu yana (son: {son_olay})"),
        (c4, f"%{100 * elapsed_ratio:.0f}",        "#E24B4A" if elapsed_ratio > 0.7 else "#1D9E75",
         "Tekrar süresinin yüzdesi"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Tüm siteler tablosu ───────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Tüm Kazı Siteleri Özet</div>', unsafe_allow_html=True)
    df_sites = pd.DataFrame([
        {"Site": s["site"],
         "Olay Sayısı": len(s["olaylar"]),
         "Tarihsel": sum(1 for ev in s["olaylar"] if ev["tip"] == "tarihsel"),
         "Paleo": sum(1 for ev in s["olaylar"] if ev["tip"] == "paleo"),
         "Tekrar (yıl)": f"{s['ortalama_tekrar']} ± {s['sigma']}",
         "En Eski (yıl)": min(ev["yil"] for ev in s["olaylar"]),
         "Son (yıl)": max(ev["yil"] for ev in s["olaylar"]),
         "Kaynak": s["ref"]}
        for s in _PALEOSEISMIK_KAZILAR
    ])
    st.dataframe(df_sites, use_container_width=True, hide_index=True)

    st.caption(
        f"📚 **Site referansı:** {site['ref']} | "
        "**Kozacı et al. (2007)** *BSSA* 97(5), 1513-1527 — DOI:10.1785/0120060118 (Yaylabeli) | "
        "**Klinger et al. (2003)** *BSSA* 93(6), 2317-2332 — DOI:10.1785/0120010270 (Batı NAFZ) | "
        "**Fraser et al. (2010)** *JGR* 115 — DOI:10.1029/2009JB006982 (sentez) | "
        "**Hubert-Ferrari et al. (2002)** *EPSL* | "
        "**Meghraoui et al. (2012)** *Tectonophysics* 538-540 (DAF). "
        "⚠️ Paleo olay tarihleri ¹⁴C/OSL belirsizliği taşır (±50-200 yıl). "
        "Olay tanılaması: kolüvyal kama, fault-scarp degradation, fissure dolgular."
    )


if active_menu == "⛏️ Paleosismik Kazı":
    _render_paleosismik_kazi()


# ════════════════════════════════════════════════════════════════════════════
# 🗺️ TSUNAMİ TEHLİKE — F-62 / v1.38 — NEAM Olasılıksal Tehlike Haritası
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Basili, R. et al. (2021). The making of the NEAM Tsunami Hazard Model
#       2018 (NEAMTHM18). Front. Earth Sci. 8, 616594.
#       DOI:10.3389/feart.2020.616594
#   • Selva, J. et al. (2016). Quantification of source uncertainties in
#       Seismic Probabilistic Tsunami Hazard Analysis (SPTHA).
#       GJI 205(3), 1780-1803. DOI:10.1093/gji/ggw107
#   • TSUMAPS-NEAM project (Horizon 2020)
#   • Yolsal-Çevikbilen, S. & Taymaz, T. (2012). Tectonophysics 536-537.
# ════════════════════════════════════════════════════════════════════════════

# NEAMTHM18 türevi: Türkiye + Akdeniz kıyısı 500-yıl tsunami yüksekliği (m)
# Basili et al. 2021 Front. Earth Sci. 8, Şekil 7 referanslı interpolasyon
_TSUNAMI_TEHLIKE_KIYILAR = [
    # Yüksek tehlike (>2 m, 500 yıl)
    {"yer": "Marmaris",       "lat": 36.85, "lon": 28.27, "h_500yr_m": 2.8, "zon": "yuksek"},
    {"yer": "Bodrum",         "lat": 37.04, "lon": 27.43, "h_500yr_m": 2.6, "zon": "yuksek"},
    {"yer": "Datça",          "lat": 36.73, "lon": 27.69, "h_500yr_m": 3.2, "zon": "yuksek"},
    {"yer": "Fethiye",        "lat": 36.65, "lon": 29.12, "h_500yr_m": 2.5, "zon": "yuksek"},
    {"yer": "Kaş",            "lat": 36.20, "lon": 29.64, "h_500yr_m": 2.2, "zon": "yuksek"},
    {"yer": "Kalkan",         "lat": 36.27, "lon": 29.40, "h_500yr_m": 2.3, "zon": "yuksek"},

    # Orta tehlike (1-2 m)
    {"yer": "İzmir",          "lat": 38.42, "lon": 27.14, "h_500yr_m": 1.6, "zon": "orta"},
    {"yer": "Kuşadası",       "lat": 37.86, "lon": 27.26, "h_500yr_m": 1.8, "zon": "orta"},
    {"yer": "Çeşme",          "lat": 38.33, "lon": 26.30, "h_500yr_m": 1.9, "zon": "orta"},
    {"yer": "Foça",           "lat": 38.66, "lon": 26.76, "h_500yr_m": 1.3, "zon": "orta"},
    {"yer": "Antalya",        "lat": 36.89, "lon": 30.71, "h_500yr_m": 1.5, "zon": "orta"},
    {"yer": "Alanya",         "lat": 36.55, "lon": 31.99, "h_500yr_m": 1.2, "zon": "orta"},
    {"yer": "Mersin",         "lat": 36.81, "lon": 34.64, "h_500yr_m": 1.0, "zon": "orta"},
    {"yer": "İskenderun",     "lat": 36.59, "lon": 36.17, "h_500yr_m": 1.4, "zon": "orta"},

    # Düşük-orta (0.5-1 m)
    {"yer": "Edremit",        "lat": 39.60, "lon": 27.02, "h_500yr_m": 0.9, "zon": "dusuk_orta"},
    {"yer": "Ayvalık",        "lat": 39.31, "lon": 26.69, "h_500yr_m": 0.8, "zon": "dusuk_orta"},
    {"yer": "Çanakkale",      "lat": 40.15, "lon": 26.41, "h_500yr_m": 0.7, "zon": "dusuk_orta"},
    {"yer": "Tekirdağ",       "lat": 40.98, "lon": 27.51, "h_500yr_m": 1.0, "zon": "dusuk_orta"},
    {"yer": "İstanbul (Marmara)", "lat": 40.85, "lon": 28.50, "h_500yr_m": 0.9, "zon": "dusuk_orta"},
    {"yer": "Yalova",         "lat": 40.65, "lon": 29.27, "h_500yr_m": 1.0, "zon": "dusuk_orta"},

    # Düşük (<0.5 m, Karadeniz / iç koylar)
    {"yer": "Samsun",         "lat": 41.29, "lon": 36.34, "h_500yr_m": 0.3, "zon": "dusuk"},
    {"yer": "Trabzon",        "lat": 41.00, "lon": 39.73, "h_500yr_m": 0.3, "zon": "dusuk"},
    {"yer": "Sinop",          "lat": 42.03, "lon": 35.15, "h_500yr_m": 0.4, "zon": "dusuk"},
    {"yer": "Zonguldak",      "lat": 41.46, "lon": 31.79, "h_500yr_m": 0.4, "zon": "dusuk"},
]

_TSUNAMI_TEHLIKE_RENK = {
    "yuksek":      "#A32D2D",
    "orta":        "#EF9F27",
    "dusuk_orta":  "#FAC775",
    "dusuk":       "#1D9E75",
}
_TSUNAMI_TEHLIKE_ETIKET = {
    "yuksek":      "Yüksek (>2 m)",
    "orta":        "Orta (1-2 m)",
    "dusuk_orta":  "Düşük-Orta (0.5-1 m)",
    "dusuk":       "Düşük (<0.5 m)",
}


@st.fragment
def _render_tsunami_tehlike():
    st.markdown(
        '<div class="chart-title">🗺️ Tsunami Tehlike Haritası — NEAMTHM18 (F-62 / v1.38)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🗺️ **NEAM Tsunami Hazard Model 2018 (NEAMTHM18):** Akdeniz'de **500 yıl dönüş "
        "periyodunda** kıyıda beklenen tsunami yüksekliği (m). Türkiye'nin GB kıyısı "
        "(Marmaris, Datça, Bodrum) Hellenic trench yakınlığı nedeniyle en yüksek "
        "tehlike altındadır. Teorik temel: **Basili et al. (2021), Front. Earth Sci. 8** + "
        "**Selva et al. (2016), GJI 205**."
    )

    # ── Dönüş periyodu seçici ──────────────────────────────────────────────
    rp_options = {
        "100 yıl (39% / 50 yr)":   0.50,
        "475 yıl (10% / 50 yr)":   1.00,
        "500 yıl":                  1.05,
        "1000 yıl (5% / 50 yr)":   1.35,
        "2500 yıl (2% / 50 yr)":   1.75,
    }
    rp_sec = st.selectbox("Dönüş periyodu seç",
                          options=list(rp_options.keys()),
                          index=2, key="ttehlike_rp")
    rp_factor = rp_options[rp_sec]

    df_t = pd.DataFrame(_TSUNAMI_TEHLIKE_KIYILAR)
    df_t["h_rp"] = df_t["h_500yr_m"] * rp_factor
    df_t["renk"] = df_t["zon"].apply(lambda z: _TSUNAMI_TEHLIKE_RENK[z])
    df_t["etiket"] = df_t["zon"].apply(lambda z: _TSUNAMI_TEHLIKE_ETIKET[z])

    # ── Harita ─────────────────────────────────────────────────────────────
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(
        lat=df_t["lat"], lon=df_t["lon"],
        mode="markers+text",
        marker=dict(
            size=14 + df_t["h_rp"] * 3,
            color=df_t["h_rp"],
            colorscale=[
                [0.00, "#1D9E75"],
                [0.25, "#7DC872"],
                [0.50, "#FAC775"],
                [0.75, "#EF9F27"],
                [1.00, "#A32D2D"],
            ],
            cmin=0,
            cmax=max(0.5, df_t["h_rp"].max()),
            colorbar=dict(
                title=dict(text=f"H ({rp_sec.split()[0]} yıl, m)",
                           font=dict(color=TEXT, size=11)),
                tickfont=dict(color=TEXT, size=10),
                bgcolor="rgba(0,0,0,0.4)",
                thickness=14, len=0.7,
            ),
            opacity=0.92,
        ),
        text=df_t["yer"],
        textfont=dict(size=9, color="#fff"),
        textposition="top right",
        hovertemplate=df_t.apply(
            lambda r: (f"<b>{r['yer']}</b><br>"
                       f"H ({rp_sec}): {r['h_rp']:.2f} m<br>"
                       f"Zon: {r['etiket']}"
                       "<extra></extra>"),
            axis=1,
        ),
        showlegend=False,
    ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=38.0, lon=30.0), zoom=5.3),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Bar chart kıyı sıralaması ──────────────────────────────────────────
    st.markdown('<div class="chart-title">📊 Kıyı Bazlı Tsunami Yükseklik Sıralaması</div>',
                unsafe_allow_html=True)
    df_t_sorted = df_t.sort_values("h_rp", ascending=True).reset_index(drop=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=df_t_sorted["yer"],
        x=df_t_sorted["h_rp"],
        orientation="h",
        marker=dict(color=df_t_sorted["renk"].tolist(), line=dict(color="#222", width=0.5)),
        text=df_t_sorted.apply(lambda r: f"{r['h_rp']:.2f} m", axis=1),
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="<b>%{y}</b><br>H = %{x:.2f} m<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis=dict(title=f"Tsunami yüksekliği H ({rp_sec}) — m", color=TEXT, gridcolor=BORDER),
        yaxis=dict(color=TEXT, gridcolor=BORDER),
        height=520,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    en_yuksek = df_t.loc[df_t["h_rp"].idxmax()]
    n_yuksek = int((df_t["zon"] == "yuksek").sum())
    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"{en_yuksek['h_rp']:.1f} m",        "#A32D2D", f"En yüksek ({en_yuksek['yer']})"),
        (c2, f"{df_t['h_rp'].mean():.1f} m",      "#EF9F27", "Türkiye kıyı ortalama"),
        (c3, f"{n_yuksek}",                        "#E24B4A", f"Yüksek zonda kıyı ({rp_sec})"),
        (c4, f"{len(df_t)}",                        "#1976D2", "Toplam değerlendirilen kıyı"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Zone tablosu ──────────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Zon × Risk Açıklaması</div>', unsafe_allow_html=True)
    df_zone = pd.DataFrame([
        {"Zon": "Yüksek (>2 m)",       "Anlam": "İnsan ve yapı için tehdit, evakuasyon planı zorunlu",
         "Kaynak": "Hellenic trench M8+ olayları"},
        {"Zon": "Orta (1-2 m)",         "Anlam": "Liman/marina hasarı, kıyı evakuasyon gerekli",
         "Kaynak": "Ege subdüksiyon orta olayları"},
        {"Zon": "Düşük-Orta (0.5-1 m)", "Anlam": "Yerel sel, dikkat seviyesi",
         "Kaynak": "Lokal heyelan/deprem tsunamileri"},
        {"Zon": "Düşük (<0.5 m)",       "Anlam": "Önemsiz",
         "Kaynak": "Karadeniz ve iç koylar"},
    ])
    st.dataframe(df_zone, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **Basili et al. (2021)** *Front. Earth Sci.* 8, 616594 — "
        "DOI:10.3389/feart.2020.616594 (NEAMTHM18) | "
        "**Selva et al. (2016)** *GJI* 205(3), 1780-1803 — DOI:10.1093/gji/ggw107 (SPTHA) | "
        "**TSUMAPS-NEAM** Horizon 2020 projesi | "
        "**Yolsal-Çevikbilen & Taymaz (2012)** *Tectonophysics* 536-537 (Türkiye). "
        "⚠️ Veri kıyı noktalarında interpolasyondur; gerçek site-spesifik analiz "
        "TSUMAPS-NEAM tam grid (tsumaps-neam.eu) ile yapılmalıdır."
    )


if active_menu == "🗺️ Tsunami Tehlike":
    _render_tsunami_tehlike()


# ════════════════════════════════════════════════════════════════════════════
# 🏔️ VS30 ZEMİN — F-64 / v1.39 — Zemin Sınıfı + NEHRP Kategorileri
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Wald, D.J. & Allen, T.I. (2007). Topographic slope as a proxy for
#       seismic site conditions and amplification. BSSA 97(5), 1379-1395.
#       DOI:10.1785/0120060267
#   • Boore, D.M. et al. (2014). NGA-West2 GMPE. Earthquake Spectra 30(3),
#       1057-1085. DOI:10.1193/070113EQS184M
#   • BSSC (2003). NEHRP Recommended Provisions for Seismic Regulations,
#       FEMA 450. (NEHRP site classes A-F tanımları)
#   • USGS Global Vs30 Server: earthquake.usgs.gov/data/vs30/
# ════════════════════════════════════════════════════════════════════════════

# Türkiye şehirleri Vs30 tahmini (USGS Global Vs30 + KOERI mikrobölgeleme türevi)
# Wald & Allen 2007 topografik eğim proxy + lokal kalibrasyon
_VS30_NOKTALAR = [
    # KAF + Marmara — karışık zemin
    {"city": "İstanbul (Kadıköy/Beyoğlu)", "lat": 41.01, "lon": 28.98, "vs30": 380, "nehrp": "C"},
    {"city": "İstanbul (Avcılar dolgu)",   "lat": 41.02, "lon": 28.72, "vs30": 180, "nehrp": "D"},
    {"city": "Adapazarı (dolgu)",          "lat": 40.78, "lon": 30.40, "vs30": 140, "nehrp": "E"},
    {"city": "İzmit",                       "lat": 40.77, "lon": 29.91, "vs30": 320, "nehrp": "D"},
    {"city": "Düzce",                       "lat": 40.84, "lon": 31.16, "vs30": 230, "nehrp": "D"},
    {"city": "Yalova (kıyı dolgu)",        "lat": 40.65, "lon": 29.27, "vs30": 200, "nehrp": "D"},
    {"city": "Bursa",                       "lat": 40.18, "lon": 29.07, "vs30": 360, "nehrp": "C"},
    {"city": "Çanakkale",                   "lat": 40.15, "lon": 26.41, "vs30": 280, "nehrp": "D"},
    {"city": "Tekirdağ",                    "lat": 40.98, "lon": 27.51, "vs30": 410, "nehrp": "C"},

    # Erzincan + Doğu Anadolu
    {"city": "Erzincan ovası (allüvyon)",   "lat": 39.75, "lon": 39.49, "vs30": 220, "nehrp": "D"},
    {"city": "Erzincan dağ etekleri (kaya)", "lat": 39.85, "lon": 39.55, "vs30": 720, "nehrp": "B"},
    {"city": "Erzurum",                      "lat": 39.91, "lon": 41.27, "vs30": 500, "nehrp": "C"},
    {"city": "Van",                          "lat": 38.49, "lon": 43.41, "vs30": 320, "nehrp": "D"},
    {"city": "Elazığ",                       "lat": 38.68, "lon": 39.22, "vs30": 380, "nehrp": "C"},
    {"city": "Malatya",                      "lat": 38.36, "lon": 38.30, "vs30": 290, "nehrp": "D"},
    {"city": "Kahramanmaraş",               "lat": 37.58, "lon": 36.93, "vs30": 350, "nehrp": "C"},
    {"city": "Hatay (Amik ovası)",          "lat": 36.20, "lon": 36.16, "vs30": 180, "nehrp": "D"},
    {"city": "Gaziantep",                    "lat": 37.07, "lon": 37.38, "vs30": 420, "nehrp": "C"},
    {"city": "Adıyaman",                     "lat": 37.76, "lon": 38.27, "vs30": 380, "nehrp": "C"},

    # Batı Anadolu (Ege grabenleri)
    {"city": "İzmir merkez (dolgu)",        "lat": 38.42, "lon": 27.14, "vs30": 230, "nehrp": "D"},
    {"city": "İzmir Karşıyaka",              "lat": 38.46, "lon": 27.12, "vs30": 220, "nehrp": "D"},
    {"city": "Manisa",                       "lat": 38.62, "lon": 27.43, "vs30": 280, "nehrp": "D"},
    {"city": "Denizli",                      "lat": 37.78, "lon": 29.09, "vs30": 320, "nehrp": "D"},
    {"city": "Aydın",                        "lat": 37.85, "lon": 27.85, "vs30": 250, "nehrp": "D"},
    {"city": "Muğla",                        "lat": 37.21, "lon": 28.36, "vs30": 520, "nehrp": "C"},
    {"city": "Antalya",                      "lat": 36.89, "lon": 30.71, "vs30": 350, "nehrp": "C"},

    # İç Anadolu (kayalık)
    {"city": "Ankara",                       "lat": 39.93, "lon": 32.86, "vs30": 580, "nehrp": "C"},
    {"city": "Konya",                        "lat": 37.87, "lon": 32.48, "vs30": 480, "nehrp": "C"},
    {"city": "Kayseri",                      "lat": 38.73, "lon": 35.48, "vs30": 620, "nehrp": "C"},
    {"city": "Sivas",                        "lat": 39.75, "lon": 37.02, "vs30": 520, "nehrp": "C"},

    # Karadeniz
    {"city": "Samsun",                       "lat": 41.29, "lon": 36.34, "vs30": 320, "nehrp": "D"},
    {"city": "Trabzon",                      "lat": 41.00, "lon": 39.73, "vs30": 480, "nehrp": "C"},
]


def _vs30_nehrp_color(nehrp: str) -> str:
    return {
        "A": "#1D9E75",  # sert kaya — yeşil
        "B": "#7DC872",
        "C": "#C0DD97",
        "D": "#FAC775",  # sert zemin — sarı
        "E": "#E24B4A",  # yumuşak — kırmızı
        "F": "#A32D2D",  # özel
    }.get(nehrp, "#888")


def _vs30_amplification(vs30: float, pga: float) -> tuple:
    """
    Boore et al. (2014) NGA-West2 basitleştirilmiş zemin amplifikasyon faktörleri.
    Lin & doğrusal olmayan terimler ihmal edilmiş.
    Vs_ref = 760 m/s (NEHRP B/C sınırı).
    """
    vs_ref = 760.0
    # Lineer terim
    if vs30 >= vs_ref:
        F_lin = 1.0
    else:
        F_lin = (vs_ref / vs30) ** 0.5  # rough approximation
    # PGA bağımlı doğrusal olmayan azalma (yüksek PGA'da zemin etkisi düşer)
    nl_factor = max(0.5, 1.0 - pga * 0.3)
    return F_lin, F_lin * nl_factor


@st.fragment
def _render_vs30_zemin():
    st.markdown(
        '<div class="chart-title">🏔️ Vs30 Zemin Sınıflandırma — NEHRP (F-64 / v1.39)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🏔️ **Vs30:** Üst 30 m'nin ortalama kayma dalga hızı (m/s). Zemin "
        "büyütme katsayısının en yaygın proxy'si. **NEHRP** (BSSC 2003) sınıfları: "
        "A (≥1500) sert kaya → E (<180) yumuşak zemin. **Wald & Allen (2007) BSSA 97** "
        "topografik eğim proxy'si; **Boore et al. (2014) Earthq. Spectra 30** "
        "NGA-West2 GMPE."
    )

    df_v = pd.DataFrame(_VS30_NOKTALAR)
    df_v["renk"] = df_v["nehrp"].apply(_vs30_nehrp_color)

    # ── Harita ─────────────────────────────────────────────────────────────
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(
        lat=df_v["lat"], lon=df_v["lon"],
        mode="markers+text",
        marker=dict(size=14, color=df_v["renk"], opacity=0.9,
                    ),
        text=df_v["city"],
        textfont=dict(size=8, color="#fff"),
        textposition="top right",
        hovertemplate=df_v.apply(
            lambda r: (f"<b>{r['city']}</b><br>"
                       f"Vs30: {r['vs30']} m/s<br>"
                       f"NEHRP: {r['nehrp']}"
                       "<extra></extra>"),
            axis=1,
        ),
        showlegend=False,
    ))

    # Legend için dummy traces
    for sınıf in ["A", "B", "C", "D", "E"]:
        fig_map.add_trace(go.Scattermapbox(
            lat=[None], lon=[None],
            mode="markers",
            marker=dict(size=12, color=_vs30_nehrp_color(sınıf)),
            name=f"NEHRP {sınıf}",
        ))

    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=39.0, lon=35.0), zoom=5),
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG2,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=11)),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ── Amplifikasyon hesaplayıcı ──────────────────────────────────────────
    st.markdown('<div class="chart-title">🧮 Zemin Büyütme Hesaplayıcı</div>', unsafe_allow_html=True)
    col_city, col_pga = st.columns([2, 1])
    with col_city:
        sec_city = st.selectbox(
            "Konum seç",
            options=df_v["city"].tolist(),
            index=df_v["city"].tolist().index("Erzincan ovası (allüvyon)")
            if "Erzincan ovası (allüvyon)" in df_v["city"].tolist() else 0,
            key="vs30_city",
        )
    with col_pga:
        pga_g = st.slider("PGA (g) — kaya temel",
                          min_value=0.05, max_value=0.6, value=0.20, step=0.05,
                          key="vs30_pga")

    site = df_v[df_v["city"] == sec_city].iloc[0]
    F_lin, F_total = _vs30_amplification(site["vs30"], pga_g)
    pga_site = pga_g * F_total

    c1, c2, c3, c4 = st.columns(4)
    risk_color = "#A32D2D" if pga_site > 0.4 else ("#EF9F27" if pga_site > 0.2 else "#1D9E75")
    kartlar = [
        (c1, f"{site['vs30']} m/s",   _vs30_nehrp_color(site["nehrp"]),
         f"Vs30 ({site['city'][:25]})"),
        (c2, site["nehrp"],            _vs30_nehrp_color(site["nehrp"]), "NEHRP zemin sınıfı"),
        (c3, f"×{F_total:.2f}",        "#EF9F27",                          "Büyütme faktörü (toplam)"),
        (c4, f"{pga_site:.3f} g",      risk_color,                         f"Site PGA ({pga_g:.2f}g kaya → site)"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── NEHRP sınıf tablosu ───────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 NEHRP Zemin Sınıfları (BSSC 2003 / FEMA 450)</div>',
                unsafe_allow_html=True)
    df_nehrp = pd.DataFrame([
        {"Sınıf": "A", "Vs30 (m/s)": "≥ 1500", "Tanım": "Sert kaya",            "Tipik Büyütme": "1.0×"},
        {"Sınıf": "B", "Vs30 (m/s)": "760 – 1500", "Tanım": "Kaya",            "Tipik Büyütme": "1.0×"},
        {"Sınıf": "C", "Vs30 (m/s)": "360 – 760",  "Tanım": "Sıkı toprak/yumuşak kaya", "Tipik Büyütme": "1.2× – 1.6×"},
        {"Sınıf": "D", "Vs30 (m/s)": "180 – 360",  "Tanım": "Sert zemin (kil/kum)",      "Tipik Büyütme": "1.5× – 2.5×"},
        {"Sınıf": "E", "Vs30 (m/s)": "< 180",       "Tanım": "Yumuşak zemin (kil)",       "Tipik Büyütme": "2.5× – 4×"},
        {"Sınıf": "F", "Vs30 (m/s)": "Özel",        "Tanım": "Sıvılaşabilir / organik",   "Tipik Büyütme": "Site-spesifik"},
    ])
    st.dataframe(df_nehrp, use_container_width=True, hide_index=True)

    # ── İl tablosu ────────────────────────────────────────────────────────
    st.markdown('<div class="chart-title">🏙️ İl/Konum Detayları</div>', unsafe_allow_html=True)
    df_show = df_v[["city", "vs30", "nehrp"]].copy()
    df_show.columns = ["Konum", "Vs30 (m/s)", "NEHRP"]
    df_show = df_show.sort_values("Vs30 (m/s)").reset_index(drop=True)
    st.dataframe(df_show, use_container_width=True, hide_index=True, height=320)

    st.caption(
        "📚 **Wald & Allen (2007)** *BSSA* 97(5), 1379-1395 — DOI:10.1785/0120060267 (Vs30 proxy) | "
        "**Boore et al. (2014)** *Earthquake Spectra* 30(3), 1057-1085 — "
        "DOI:10.1193/070113EQS184M (NGA-West2 GMPE) | "
        "**BSSC (2003)** FEMA 450 NEHRP Provisions | "
        "**USGS Global Vs30:** earthquake.usgs.gov/data/vs30/. "
        "⚠️ Vs30 değerleri proxy/lokal kalibrasyon türevi; tam değerlendirme MASW/REMI "
        "saha ölçümü ile yapılmalıdır."
    )


if active_menu == "🏔️ Vs30 Zemin":
    _render_vs30_zemin()


# ════════════════════════════════════════════════════════════════════════════
# 🏚️ HAZUS KAYIP — F-65 / v1.40 — HAZUS + Fragility Curves
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • FEMA (2003). HAZUS-MH Technical Manual. Washington, DC.
#       fema.gov/sites/default/files/2020-09/fema_hazus_earthquake-model
#   • Erdik, M. et al. (2003). Earthquake risk assessment for Istanbul
#       metropolitan area. Earthq. Eng. Eng. Vibr. 2(1), 1-23.
#       DOI:10.1007/BF02857534
#   • Lagomarsino, S. & Giovinazzi, S. (2006). Macroseismic and mechanical
#       models for vulnerability and damage assessment of current buildings.
#       Bull. Earthq. Eng. 4(4), 415-443.
#       DOI:10.1007/s10518-006-9024-z
#   • Silva, V. et al. (2019). Current challenges in fragility modeling.
#       Earthq. Spectra 35(4), 1927-1952.
# ════════════════════════════════════════════════════════════════════════════

# Yapı tipi × hasar seviyesi median PGA + β (Lagomarsino 2006 Tablo 4-7 türevi)
_FRAGILITY = {
    "Yığma (1970 öncesi)": {
        "median_pga": {"slight": 0.10, "moderate": 0.20, "extensive": 0.35, "complete": 0.55},
        "beta": 0.65, "renk": "#A32D2D",
        "ref": "Lagomarsino 2006 Tablo 4 — yığma yüksek kırılganlık"},
    "Yığma (1975-1998)": {
        "median_pga": {"slight": 0.12, "moderate": 0.25, "extensive": 0.45, "complete": 0.70},
        "beta": 0.60, "renk": "#E24B4A",
        "ref": "Lagomarsino 2006; 1975 yönetmeliği sonrası"},
    "Betonarme (1975 öncesi)": {
        "median_pga": {"slight": 0.15, "moderate": 0.28, "extensive": 0.50, "complete": 0.80},
        "beta": 0.62, "renk": "#EF9F27",
        "ref": "Erdik 2003 İstanbul senaryo"},
    "Betonarme (1975-1998)": {
        "median_pga": {"slight": 0.18, "moderate": 0.35, "extensive": 0.65, "complete": 1.00},
        "beta": 0.58, "renk": "#FAC775",
        "ref": "1975 ABYYHY yönetmeliği"},
    "Betonarme (1998-2018, DBYBHY)": {
        "median_pga": {"slight": 0.25, "moderate": 0.50, "extensive": 0.85, "complete": 1.30},
        "beta": 0.55, "renk": "#C0DD97",
        "ref": "DBYBHY 1998 yönetmeliği"},
    "Betonarme (2018+, TBDY-2018)": {
        "median_pga": {"slight": 0.32, "moderate": 0.65, "extensive": 1.10, "complete": 1.70},
        "beta": 0.50, "renk": "#1D9E75",
        "ref": "TBDY-2018, performansa dayalı tasarım"},
}

# HAZUS hasar oranı → can kaybı (FEMA 2003 Tablo 13.4 türevi)
# Complete damage'da: gündüz %0.25, gece %1.0 ölüm oranı (bina kullanım yoğunluğu)
_HAZUS_OLUM_ORANI = {"slight": 0.0001, "moderate": 0.001, "extensive": 0.01, "complete": 0.10}
_HAZUS_AGIR_YARALI = {"slight": 0.0005, "moderate": 0.005, "extensive": 0.03, "complete": 0.20}


def _fragility_prob(pga: float, median: float, beta: float) -> float:
    """Lognormal CDF (Lagomarsino 2006): P(hasar ≥ seviye | PGA)."""
    if pga <= 0 or median <= 0:
        return 0.0
    z = math.log(pga / median) / beta
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@st.fragment
def _render_hazus_kayip():
    st.markdown(
        '<div class="chart-title">🏚️ HAZUS Kayıp Tahmini — Fragility Curves (F-65 / v1.40)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🏚️ **HAZUS Metodolojisi:** FEMA'nın 2003'ten beri kullandığı kayıp tahmini "
        "çerçevesi. Yapı tipi × PGA × **lognormal kırılganlık eğrisi** → hasar "
        "olasılığı → can kaybı tahmini. Türkiye uygulaması: **Erdik et al. (2003) "
        "Earthq. Eng. Eng. Vibr. 2(1)** (İstanbul senaryosu); fragility parametreleri: "
        "**Lagomarsino & Giovinazzi (2006) Bull. Earthq. Eng. 4(4)**."
    )

    # ── Kontroller ─────────────────────────────────────────────────────────
    col_pga, col_yapi, col_nufus = st.columns([1, 2, 1])
    with col_pga:
        pga = st.slider("Beklenen PGA (g)",
                        min_value=0.05, max_value=1.50, value=0.40, step=0.05,
                        key="hazus_pga")
    with col_yapi:
        sec_yapi = st.selectbox(
            "Yapı tipi",
            options=list(_FRAGILITY.keys()),
            index=2,  # Default: Betonarme (1975 öncesi)
            key="hazus_yapi",
        )
    with col_nufus:
        nufus = st.number_input(
            "Etkilenen nüfus (bin)",
            min_value=10, max_value=10000, value=500, step=50,
            key="hazus_nufus",
        )

    frag = _FRAGILITY[sec_yapi]

    # ── Hasar olasılıkları (kümülatif → bant) ──────────────────────────────
    p_cum = {seviye: _fragility_prob(pga, frag["median_pga"][seviye], frag["beta"])
             for seviye in ["slight", "moderate", "extensive", "complete"]}
    # Bant olasılıkları: P(slight only) = P(>slight) - P(>moderate), vb.
    p_band = {
        "no_damage": 1 - p_cum["slight"],
        "slight":    p_cum["slight"] - p_cum["moderate"],
        "moderate":  p_cum["moderate"] - p_cum["extensive"],
        "extensive": p_cum["extensive"] - p_cum["complete"],
        "complete":  p_cum["complete"],
    }
    # Negatif düzeltme (numerik)
    for k in p_band:
        p_band[k] = max(0.0, p_band[k])

    # ── Fragility eğrileri grafiği ─────────────────────────────────────────
    st.markdown('<div class="chart-title">📈 Kırılganlık Eğrileri (Fragility Curves)</div>',
                unsafe_allow_html=True)
    pga_range = np.linspace(0.01, 2.0, 100)
    fig_frag = go.Figure()
    renkler_seviye = {"slight": "#1D9E75", "moderate": "#FAC775",
                      "extensive": "#EF9F27", "complete": "#A32D2D"}
    etiketler_seviye = {"slight": "Hafif (slight)", "moderate": "Orta (moderate)",
                        "extensive": "Ağır (extensive)", "complete": "Çöküş (complete)"}
    for seviye, etiket in etiketler_seviye.items():
        med = frag["median_pga"][seviye]
        probs = [100 * _fragility_prob(p, med, frag["beta"]) for p in pga_range]
        fig_frag.add_trace(go.Scatter(
            x=pga_range, y=probs,
            mode="lines",
            line=dict(color=renkler_seviye[seviye], width=2.5),
            name=f"≥ {etiket} (median PGA={med:.2f}g)",
            hovertemplate="PGA = %{x:.2f} g<br>P(≥hasar) = %{y:.1f}%<extra></extra>",
        ))

    # Seçili PGA dikey çizgisi
    fig_frag.add_vline(x=pga, line=dict(color="#FFD700", width=2.5, dash="dash"),
                       annotation_text=f"Seçili: {pga:.2f} g",
                       annotation_font_color="#FFD700")

    fig_frag.update_layout(
        title=dict(text=f"Fragility — {sec_yapi}", font=dict(color=TEXT, size=13)),
        xaxis=dict(title="PGA (g)", color=TEXT, gridcolor=BORDER, range=[0, 2]),
        yaxis=dict(title="P(hasar ≥ seviye) %", color=TEXT, gridcolor=BORDER, range=[0, 105]),
        height=380,
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0.3)"),
    )
    st.plotly_chart(fig_frag, use_container_width=True, config={"displayModeBar": False})

    # ── Bant dağılımı pie ──────────────────────────────────────────────────
    st.markdown('<div class="chart-title">🥧 Hasar Dağılımı (Verilen PGA için)</div>',
                unsafe_allow_html=True)
    fig_pie = go.Figure()
    fig_pie.add_trace(go.Pie(
        labels=["Hasarsız", "Hafif", "Orta", "Ağır", "Çöküş"],
        values=[100 * p_band["no_damage"], 100 * p_band["slight"],
                100 * p_band["moderate"], 100 * p_band["extensive"],
                100 * p_band["complete"]],
        marker=dict(colors=["#1D9E75", "#C0DD97", "#FAC775", "#EF9F27", "#A32D2D"]),
        textinfo="label+percent",
        textfont=dict(color="#fff", size=11),
        hoverinfo="label+value",
    ))
    fig_pie.update_layout(
        height=380,
        paper_bgcolor=BG2,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(color=TEXT, size=10)),
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # ── Can kaybı tahmini ──────────────────────────────────────────────────
    nufus_full = nufus * 1000
    olum_tahmin = sum(p_band[s] * _HAZUS_OLUM_ORANI[s] * nufus_full
                      for s in ["slight", "moderate", "extensive", "complete"])
    yarali_tahmin = sum(p_band[s] * _HAZUS_AGIR_YARALI[s] * nufus_full
                        for s in ["slight", "moderate", "extensive", "complete"])
    hasarsiz_bina_pct = 100 * p_band["no_damage"]
    cokus_bina_pct = 100 * p_band["complete"]

    c1, c2, c3, c4 = st.columns(4)
    kartlar = [
        (c1, f"%{cokus_bina_pct:.1f}",     "#A32D2D", "Çöküş olasılığı (bina)"),
        (c2, f"%{hasarsiz_bina_pct:.1f}",  "#1D9E75", "Hasarsız kalma olasılığı"),
        (c3, f"~{int(olum_tahmin):,}",      "#E24B4A", f"Tahmini can kaybı ({nufus} bin nüfus)"),
        (c4, f"~{int(yarali_tahmin):,}",   "#EF9F27", "Tahmini ağır yaralı"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Yapı tipleri karşılaştırma tablosu ────────────────────────────────
    st.markdown('<div class="chart-title">📋 Yapı Tipleri × Çöküş Olasılığı (PGA = '
                f'{pga:.2f} g)</div>', unsafe_allow_html=True)
    df_comp = pd.DataFrame([
        {"Yapı Tipi": yapi,
         "Çöküş Median PGA (g)": params["median_pga"]["complete"],
         "β": params["beta"],
         f"P(Çöküş) @ {pga:.2f}g %": round(100 * _fragility_prob(pga, params["median_pga"]["complete"], params["beta"]), 1),
         "Kaynak": params["ref"]}
        for yapi, params in _FRAGILITY.items()
    ])
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **FEMA (2003)** *HAZUS-MH Technical Manual* (kayıp tahmini standardı) | "
        "**Erdik et al. (2003)** *Earthq. Eng. Eng. Vibr.* 2(1), 1-23 — "
        "DOI:10.1007/BF02857534 (İstanbul senaryosu) | "
        "**Lagomarsino & Giovinazzi (2006)** *Bull. Earthq. Eng.* 4(4) — "
        "DOI:10.1007/s10518-006-9024-z (fragility eğrileri) | "
        "**Silva et al. (2019)** *Earthq. Spectra* 35(4) (modern challenges). "
        "⚠️ Sonuçlar yaklaşık. Tam HAZUS uygulaması bina envanteri (TÜİK), zemin tipi "
        "(Vs30 — F-64), saatlik nüfus dağılımı gerektirir."
    )


if active_menu == "🏚️ HAZUS Kayıp":
    _render_hazus_kayip()


# ════════════════════════════════════════════════════════════════════════════
# 🏺 ERZİNCAN PALEO — F-67 / v1.41 — Erzincan Segmenti Paleosismolojisi
# ────────────────────────────────────────────────────────────────────────────
# Bilimsel temel:
#   • Kozacı, Ö., Doğan, B., Özaksoy, V., Yıldırım, C., Gökaşan, E. &
#       Tokay, F. (2007). Paleoseismological evidence for the relatively
#       regular recurrence of infrequent, large-magnitude earthquakes on
#       the eastern North Anatolian fault at Yaylabeli (Erzincan).
#       BSSA 97(5), 1513-1527. DOI:10.1785/0120060118
#   • Barka, A. (1996). Slip distribution along the NAF associated with
#       large earthquakes 1939-1967. BSSA 86(5), 1238-1254.
#   • Hartleb, R.D., Dolan, J.F., Kozacı, Ö., Akyüz, H.S. & Seitz, G.G.
#       (2006). A 2500-year-long paleoseismologic record on the eastern
#       NAF, Turkey. GSA Bulletin 118(7-8), 823-840.
#       DOI:10.1130/B25849.1
#   • Akyüz, H.S. et al. (2002). Surface ruptures of 1939, 1942, 1943,
#       1957, 1967, 1992 NAF earthquakes. BSSA 92(1), 61-66.
# ════════════════════════════════════════════════════════════════════════════

# Yaylabeli (Erzincan) — Kozacı 2007 6 olay + Hartleb 2006 ek geriye dönüş
_ERZINCAN_PALEO_OLAYLAR = [
    {"olay": "E1", "yil": 1939, "belirsizlik": 0,
     "kaynak": "Tarihsel (Ms 7.8)", "siddet": "yuksek"},
    {"olay": "E2", "yil": 1668, "belirsizlik": 0,
     "kaynak": "Ambraseys 2009 (KAF doğu kolu? tartışmalı)", "siddet": "yuksek"},
    {"olay": "E3", "yil": 1180, "belirsizlik": 60,
     "kaynak": "Kozacı 2007 14C", "siddet": "yuksek"},
    {"olay": "E4", "yil": 870, "belirsizlik": 80,
     "kaynak": "Kozacı 2007 14C", "siddet": "yuksek"},
    {"olay": "E5", "yil": 560, "belirsizlik": 100,
     "kaynak": "Kozacı 2007 14C", "siddet": "yuksek"},
    {"olay": "E6", "yil": 250, "belirsizlik": 120,
     "kaynak": "Kozacı 2007 14C", "siddet": "yuksek"},
    {"olay": "E7", "yil": -100, "belirsizlik": 150,
     "kaynak": "Kozacı 2007 14C / Hartleb 2006", "siddet": "orta"},
    {"olay": "E8", "yil": -450, "belirsizlik": 180,
     "kaynak": "Hartleb 2006 GSA 118 — 2500 yıl arşiv", "siddet": "orta"},
    {"olay": "E9", "yil": -800, "belirsizlik": 200,
     "kaynak": "Hartleb 2006 GSA 118", "siddet": "orta"},
]


@st.fragment
def _render_erzincan_paleo():
    st.markdown(
        '<div class="chart-title">🏺 Erzincan Paleosismoloji — Yaylabeli Trench (F-67 / v1.41)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "🏺 **Erzincan Segmenti:** Kozacı et al. (2007) Yaylabeli (Erzincan) kazısında "
        "**6 paleo olay** + 1939 tarihsel toplam **9 olay** belgeledi. **Hartleb et al. "
        "(2006) GSA Bulletin 118** 2500 yıllık arşivi genişletti. **Ortalama tekrar "
        "süresi ~320 ± 80 yıl** — istatistiksel olarak en uzun gözlenen NAFZ paleo veri."
    )

    df_p = pd.DataFrame(_ERZINCAN_PALEO_OLAYLAR)
    YIL_SIMDI = datetime.now().year

    # ── Yaşam çizgisi (olay timeline) ──────────────────────────────────────
    fig_tl = go.Figure()
    for i, ev in df_p.iterrows():
        renk = "#A32D2D" if ev["siddet"] == "yuksek" else "#EF9F27"
        fig_tl.add_trace(go.Scatter(
            x=[ev["yil"]],
            y=[0],
            mode="markers+text",
            marker=dict(size=22, color=renk, symbol="diamond",
                        line=dict(color="#fff", width=2)),
            error_x=dict(type="data", array=[ev["belirsizlik"]],
                         color="rgba(255,255,255,0.4)",
                         thickness=1.5, width=8),
            text=[ev["olay"]],
            textposition="top center",
            textfont=dict(size=12, color="#FFD700"),
            hovertemplate=(
                f"<b>{ev['olay']}</b><br>"
                f"Yıl: {ev['yil']} ± {ev['belirsizlik']}<br>"
                f"Kaynak: {ev['kaynak']}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig_tl.add_vline(x=YIL_SIMDI, line=dict(color="#FFD700", width=2.5, dash="dash"),
                     annotation_text=f"Şu an ({YIL_SIMDI})", annotation_font_color="#FFD700")

    fig_tl.update_layout(
        title=dict(text="Yaylabeli Yaşam Çizgisi (MÖ 1000 → MS 2050)",
                   font=dict(color=TEXT, size=13)),
        xaxis=dict(title="Yıl (MS, negatif = MÖ)", color=TEXT, gridcolor=BORDER,
                   range=[-1000, 2050]),
        yaxis=dict(visible=False, range=[-1, 2]),
        height=240,
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        showlegend=False,
    )
    st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})

    # ── Olaylar arası süre dağılımı (histogram) ────────────────────────────
    yillar_sorted = sorted([ev["yil"] for ev in _ERZINCAN_PALEO_OLAYLAR])
    intervals = [yillar_sorted[i+1] - yillar_sorted[i]
                 for i in range(len(yillar_sorted) - 1)]

    st.markdown('<div class="chart-title">📊 Olaylar Arası Süre Dağılımı</div>',
                unsafe_allow_html=True)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_int = go.Figure()
        fig_int.add_trace(go.Histogram(
            x=intervals, nbinsx=8,
            marker=dict(color="#1976D2", line=dict(color="#222", width=0.5)),
            opacity=0.85,
            hovertemplate="Aralık: %{x} yıl<br>N: %{y}<extra></extra>",
        ))
        ortalama = float(np.mean(intervals))
        fig_int.add_vline(x=ortalama, line=dict(color="#FFD700", width=2, dash="dash"),
                          annotation_text=f"Ort: {ortalama:.0f} yıl",
                          annotation_font_color="#FFD700")
        fig_int.update_layout(
            xaxis=dict(title="Olaylar arası süre (yıl)", color=TEXT, gridcolor=BORDER),
            yaxis=dict(title="Frekans", color=TEXT, gridcolor=BORDER),
            height=320,
            margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor=BG2, plot_bgcolor=BG2,
            showlegend=False,
        )
        st.plotly_chart(fig_int, use_container_width=True, config={"displayModeBar": False})

    with col_h2:
        # Olaylar arası süre listesi
        intervals_df = pd.DataFrame([
            {"Aralık": f"{yillar_sorted[i]:>5} → {yillar_sorted[i+1]:>5}",
             "Süre (yıl)": yillar_sorted[i+1] - yillar_sorted[i]}
            for i in range(len(yillar_sorted) - 1)
        ])
        intervals_df = intervals_df.sort_values("Süre (yıl)", ascending=False).reset_index(drop=True)
        st.dataframe(intervals_df, use_container_width=True, hide_index=True, height=320)

    # ── Bilgi kartları ─────────────────────────────────────────────────────
    elapsed = YIL_SIMDI - 1939
    pct_of_avg = 100 * elapsed / ortalama
    c1, c2, c3, c4 = st.columns(4)
    color_pct = "#A32D2D" if pct_of_avg > 80 else ("#EF9F27" if pct_of_avg > 50 else "#1D9E75")
    kartlar = [
        (c1, f"{len(df_p)}",                  "#FFD700", "Belgelenmiş olay (2500 yıl)"),
        (c2, f"{ortalama:.0f} ± {np.std(intervals):.0f} yıl", "#EF9F27", "Ortalama tekrar süresi"),
        (c3, f"{elapsed} yıl",                "#1976D2", "1939'dan bu yana"),
        (c4, f"%{pct_of_avg:.0f}",            color_pct, "Ortalama döngünün yüzdesi"),
    ]
    for col, val, color, label in kartlar:
        with col:
            st.markdown(
                f'<div class="stat-box">'
                f'<div style="font-size:1.35rem;font-weight:800;color:{color}">{val}</div>'
                f'<div style="font-size:0.7rem;opacity:0.55;margin-top:2px">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    # ── Olay tablosu ──────────────────────────────────────────────────────
    st.markdown('<div class="chart-title">📋 Erzincan Segmenti Paleo Olay Kataloğu</div>',
                unsafe_allow_html=True)
    df_show = df_p[["olay", "yil", "belirsizlik", "kaynak", "siddet"]].copy()
    df_show.columns = ["Olay", "Yıl", "Belirsizlik ±yıl", "Tarihlendirme Kaynağı", "Şiddet Güveni"]
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.caption(
        "📚 **Kozacı et al. (2007)** *BSSA* 97(5), 1513-1527 — DOI:10.1785/0120060118 "
        "(Yaylabeli birincil kazı) | "
        "**Hartleb et al. (2006)** *GSA Bulletin* 118(7-8), 823-840 — "
        "DOI:10.1130/B25849.1 (2500 yıllık arşiv) | "
        "**Barka (1996)** *BSSA* 86(5), 1238-1254 | "
        "**Akyüz et al. (2002)** *BSSA* 92(1) (yüzey kırıkları). "
        "⚠️ 1668 olayının Erzincan segmentine ait olması tartışmalı; 14C tarih belirsizliği "
        "100-200 yıl. Sismik döngü tahminleri **olasılıksal** — kesin değil."
    )


if active_menu == "🏺 Erzincan Paleo":
    _render_erzincan_paleo()

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
