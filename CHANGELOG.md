# Changelog

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
