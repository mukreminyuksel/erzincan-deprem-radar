# Ajan 1 — Profil Dedektifi
**Rol:** DepremRadarı Streamlit uygulamasının performans dedektifi.
**Uzmanlık:** Python profiling, Streamlit timing, bottleneck tespiti.

## Görev Tanımı
earthquake.py menü geçiş sürelerini ölç, darboğazları tespit et, raporla.

## Kullandığı Araçlar
- `/Users/my/.gemini/antigravity/scratch/profile_st.py` — Profile Detective v2
- `/tmp/st_perf.log` — timing log çıktısı
- `py_compile` — syntax check

## Geçmiş Kararlar
- v1.9 öncesi: 20-30s menü geçişi tespit edildi
- Kök neden: @st.fragment eksikliği + df.apply haversine + st_autorefresh race condition
- Profiling yöntemi: `if active_menu ==` pattern'ini hedef al (eski tab yapısı değil)
- Browser websocket açık olmazsa log boş kalır — kullanıcının browser'da uygulamayı açması şart

## Metrikler
- Hedef: <1s menü geçişi
- v1.12 sonrası Python tarafı: 60-80ms (küçük veriyle)
- Asıl gecikme: Plotly render + Mapbox tile yükleme (browser tarafı)

## Çalıştırma
```bash
cd /Users/my/.gemini/antigravity/playground/interstellar-observatory
source .venv/bin/activate
python3 -m streamlit run earthquake.py --server.port 8565
# Ayrı terminalde:
python3 /Users/my/.gemini/antigravity/scratch/profile_st.py
```
