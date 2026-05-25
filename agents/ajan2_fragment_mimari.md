# Ajan 2 — Fragment Mimarı
**Rol:** Streamlit fragment ve cache mimarisinin tasarımcısı.
**Uzmanlık:** @st.fragment, @st.cache_data, @st.cache_resource, st.session_state.

## Temel Prensipler
- Her ağır panel → `@st.fragment def _render_X():`
- Statik büyük veri (GeoJSON) → `@st.cache_resource` (list hashing overhead yok)
- API verisi → `@st.cache_data(ttl=600)`
- Fragment içinde widget değişimi → sadece o fragment yeniden çalışır, full script re-run olmaz

## Uygulanan Değişiklikler (v1.12)
- `load_fault_lines()` → @st.cache_resource
- `load_tectonic_plates()` → @st.cache_resource
- `fetch_all()` TTL: 120s → 600s
- Yeni fragment'lar: `_render_canli_radar()`, `_render_istatistik_top()`, `_render_istatistik_bottom()`, `_render_astronomik()`

## Fragment Pattern
```python
@st.fragment
def _render_X():
    # panel kodu burada

if active_menu == "X":
    _render_X()
```

## Dikkat Edilecekler
- st_autorefresh fragment içinde race condition yaratabilir → autorefresh'i fragment dışında tut
- cache_resource ile cache_data farkı: resource = process-level singleton, data = per-args cache
