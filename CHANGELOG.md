# Changelog

## v1.15a - 2026-05-25 — UI Uzmanı: ANA MENÜ üst horizontal pill
- 🎨 UI Uzmanı kararı (PLAN_v2.md): ANA MENÜ sidebar altından **ana içerik üstündeki horizontal pill bar**'a taşındı.
- `streamlit-option-menu>=0.3.13` bağımlılığı eklendi (`requirements.txt`).
- 8 panel (Canlı Radar / İstatistik & Analiz / Fay / Astronomi / Erken Uyarı / Bilgi / Sistem / Rapor) Bootstrap ikonları ile pill bar olarak metrik kartlarının üstünde.
- Gerekçe: F-shaped reading + Fitts's Law — sidebar-alt konumu kullanıcıyı 800-1200 px scroll'a zorluyordu; üst pill bar mesafesi ~60 px sabit.
- Sismolog gözüyle: panel geçişlerinde göz "yo-yo" hareketi ortadan kalktı; bilişsel yük filtre kararlarına ayrılabilir hale geldi.
- v1.15b (sıradaki): sidebar 720-751 bölgesi 3 `st.expander` (Görünüm / Veri Kaynakları / Sistem) altına toplanacak.
- features/BACKLOG.md eklendi (Tasarım Master / Ajan 9): 42 feature'lık detaylı yol haritası, peer-reviewed sismolojik atıflarla.

## v1.14 - 2026-05-25 — Tektonik Veri Uzmanı (Ajan 4 — Adım A)
- PB2002 plaka sınırı tipi sınıflandırma sistemi eklendi (convergent / divergent / transform / unknown).
- 8 Türkiye + 12 dünya sınırı manuel etiketlendi: NAFZ, EAFZ, Helenik Yay, Ölüdeniz, Bitlis-Zagros, Mid-Atlantic, San Andreas, Himalaya, Andlar vb.
- Render artık tip bazında 4 ayrı trace + legend: kırmızı (subduction/collision), mavi (rift), sarı (strike-slip), gri (sınıflandırılmamış).
- Plaka sınırları varsayılan AÇIK (önceden kapalıydı, kullanıcı keşfetmiyordu).
- Hover tooltip'te plaka çifti + Türkçe sınır adı + tip açıklaması.

## v1.13 - 2026-05-25 — Vektörleştirme Uzmanı (Ajan 3)
- `fetch_all` içinde `df.apply(haversine, axis=1)` → NumPy vektörleştirilmiş radyan-matematik (~28× hızlanma, 1000 satır 8.4ms → 0.3ms).
- Dedup loop O(n²) → O(n × k) sliding-window (descending-sorted zaman array'i üzerinde 120s pencereli, doğruluk testinde 200 satır eski/yeni birebir aynı 150 unique).
- Vektörleştirme + dedup birlikte: cache-miss anında yaşanan saniye-seviyesi takılma kalkar.

## v1.12 - 2026-05-25 — Fragment Mimarı (Ajan 2)
- 3 ağır panel @st.fragment ile sarıldı: Canlı Radar, İstatistik & Analiz (üst + alt iki blok), Astronomik Analiz.
- Panel-içi widget etkileşimleri ("Çalıştır" butonu, sub-tab selectbox) sadece o fragment'i re-run eder; üst script ve diğer panel state'leri korunur.
- `load_fault_lines` ve `load_tectonic_plates` `@st.cache_data` → `@st.cache_resource` (her menü geçişindeki ~40-60ms hash overhead ortadan kalktı).
- `fetch_all` TTL 120s → 600s (cache-miss frekansı 5× azaldı).

## v1.11 - 2026-05-25
- Added `🚨 Erken Uyarı` sidebar menu with P/S wave countdown simulator.
- 15 major Turkish cities dropdown + manual coordinate input as observer location.
- Computes hypocenter distance (3D) and arrival times for P (6.0 km/s), S (3.5 km/s), Rayleigh (2.5 km/s) waves.
- Highlighted **S − P warning window** card with 4-tier color coding and action message.
- MMI intensity estimate using simplified GMPE `I = 1.5·M − 1.5·log10(R) − 3.5` (5 intensity bands).
- Plotly timeline visualization: arrival markers, warning band, danger band.
- 3 educational "Bu Süre İçinde Ne Yapmalı?" cards (Drop-Cover-Hold / Avoid windows / Vehicle-outdoor).
- Edge case warnings: epicenter (<5 km), deep focus (>200 km), far-field (>500 km).
- Clear disclaimer that this is a concept demo only, not a replacement for AFAD-EQE.

## v1.10 - 2026-05-25
- Added `🔭 Astronomik Analiz` sidebar menu with 5-component panel exploring lunar/solar/planetary correlations with seismicity.
- Component 1: Real-time celestial state cards (Moon phase, distance, altitude; Sun, Jupiter, Venus; syzygy/quadrature alignment).
- Component 2: Moon phase vs earthquake magnitude scatter plot with spring-tide bands.
- Component 3: Lunar gravitational influence time series overlaid with earthquake events.
- Component 4: FFT periodogram of daily earthquake counts with reference lines at lunar (29.5d), semi-lunar M2 (14.8d), annual (365.25d) periods.
- Component 5: Planetary gravitational influence (Jupiter + Venus, mass-weighted 1/d²) time series with event overlay.
- Heavy `ephem` calculations gated behind explicit "Çalıştır" button to protect CPU.
- Scientific honesty: explicit non-causality warnings citing Cochran 2004 and Métivier 2009.

## v1.9 - 2026-05-25
- Refactored top navigation: tabs → sidebar radio menu for true lazy loading (only active section renders).
- Sub-tabs (η Clustering / RTL Silence / AMR Power-Law / b-Value Map) under İstatistik & Analiz converted to selectbox lazy loading.
- Pre-computed bounding boxes (`min_lat`, `max_lat`, `min_lon`, `max_lon`) for fault segments in `load_fault_lines()`.
- `nearest_fault_vertex_distance()` now uses bbox pre-filter with safe-margin (111 km/° latitude, 80 km/° longitude) — typical 5-10× speedup on large fault catalogs.
- `calc_b_grid_cache()` haversine inner loop vectorized with NumPy (replacing `df.apply` Python loop) — ~100× speedup on dense spatial grids.

## v1.8 - 2026-04-30
- UX & performance: explicit "Çalıştır" button on heavy `İstatistik & Analiz` computations to prevent autorefresh-triggered freezes.
- API client timeouts tuned to keep UI responsive when one source is slow.

## v1.7 - 2026-04-29
- Added astronomical features to correlation matrix via the `ephem` Python library (Moon distance/gravity proxy, Sun distance, Jupiter + Venus mass-weighted influence).
- Added `mevsim` (DoY sinusoid), `sicaklik` (climate-model temperature proxy), and `haftalik_aktivite` environmental features.
- General performance optimizations across the radar and analysis tabs.
- Fix: pulled API request timeouts down to 4.0s and added a Run button to `stats_tab` to prevent UI locking.

## v1.6 - 2026-04-28
- Improved `Bilgi Havuzu` wave education with a clearer observer station, 3D subsurface cross-section, P/S particle-motion arrows, and Rayleigh surface-particle ellipses.
- Rebuilt the Erzincan virtual impact view as an oblique 3D basin scene with terrain deformation, fault trace, real event markers, and animated wavefronts.

## v1.5 - 2026-04-28
- Fixed Streamlit hot-reload/cache issue that could keep an older `earthquake_core` module in memory and break `event_signature` imports.
- Restarted the main local app on port 8560 with the current code path.

## v1.4 - 2026-04-28
- Fixed `Olay Detayı` event selection so the chosen earthquake remains stable across refreshes.
- Rebuilt `Bilgi Havuzu` with 3D fault-mechanics, P/S/Rayleigh wave, and Erzincan virtual impact-map simulations.
- Added explicit educational-use caveats to separate schematic simulations from scientific analysis outputs.

## v1.3 - 2026-04-28
- Added `Bilgi Havuzu` education screen with schematic fault-motion simulations.
- Added `Veri Kalitesi` screen documenting source counts, deduplication tolerances, and analysis-catalog caveats.
- Strengthened auto-refresh fade suppression for Streamlit stale elements.

## v1.2 - 2026-04-28
- Added dedicated screens for source health, event detail, fault analysis, activity/alarm, and reporting.
- Improved refresh comfort by suppressing Streamlit stale element fade during auto-refresh.
- Added version tracking with `VERSION`, `APP_VERSION`, and visible app labels.

## v1.1 - 2026-04-28
- Added USGS Fast Feed source.
- Added flexible time range controls.
- Split radar, scientific analysis, and data table into clearer tabs.
- Added source coverage and cumulative energy analysis.
