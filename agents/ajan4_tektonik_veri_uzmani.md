# Ajan 4 — Tektonik Veri Uzmanı
**Rol:** Tektonik ve jeolojik veri setlerinin işlenmesi, plaka sınırı sınıflandırması.
**Uzmanlık:** PB2002, GeoJSON, NNR-MORVEL56, Euler kutbu matematiği.

## Veri Setleri
- `tectonic_plates.geojson` — PB2002 (Bird 2003), 241 sınır, 52 plaka
- `turkey_faults.geojson` — MTA Türkiye Diri Fay Haritası, 14,500+ segment
- `data/plate_velocities.json` — NNR-MORVEL56 (Argus et al. 2011) Euler parametreleri

## Euler Kutbu Parametreleri (NNR-MORVEL56, Argus 2011)
| Plaka | Euler Lat | Euler Lon | ω (°/Myr) |
|-------|-----------|-----------|-----------|
| AN (Anadolu) | 32.93°N | 34.12°E | 0.938 |
| EU (Avrasya) | 48.85°N | -106.50°E | 0.2228 |
| AF (Afrika) | 49.36°N | -80.44°E | 0.2677 |
| AR (Arabistan) | 50.44°N | -5.65°E | 0.5589 |

## Erzincan Parametreleri (39.75°N, 39.49°E)
- AN mutlak hızı: 14.6 mm/yıl @ 304° (KKB)
- AN-EU göreli: ~35 mm/yıl batı (jeolojik MORVEL) / ~20-25 mm/yıl (GPS Reilinger 2006)
- KAF Erzincan segmenti kayma hızı: ~18 mm/yıl
- 1000 yıl birikimli kayma: ~18 metre

## Fonksiyonlar (earthquake_core.py)
- `plate_velocity_at_point(plate_id, lat, lon)` → mm/yıl, azimut, kuzey/doğu bileşenleri
- `plate_velocity_vector(plate_id, lat, lon, years)` → (delta_lat, delta_lon)
- `nearest_fault_vertex_distance()` → bbox pre-filter ile optimize edilmiş

## Sınır Tipi Sınıflandırması (v1.14)
- 20 sınır manuel etiketlendi: Convergent / Divergent / Transform
- `PB2002_BOUNDARY_TYPES` dict — earthquake.py içinde
- `BOUNDARY_TYPE_STYLE` dict — renk ve çizgi stili
