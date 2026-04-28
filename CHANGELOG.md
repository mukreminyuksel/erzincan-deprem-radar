# Changelog

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
