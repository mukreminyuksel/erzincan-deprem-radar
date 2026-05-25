# DepremRadarı Feature Backlog
Tarih: 2026-05-25 | Sahibi: Tasarım Master (Ajan 9) | İlgili plan: [PLAN_v2.md](../PLAN_v2.md) | Sürüm: v1.14 → v1.15+

---

## Özet
- **Toplam özellik:** 38 (26 mevcut + 12 önerilen)
- **Tamamlanmış (v1.0-v1.14):** 26
- **Aktif geliştirme (v1.15 sprint):** 3
- **Backlog (v1.15-v1.18 hedefli):** 9
- **Donmuş / iptal:** 3

Tablo iki bloktan oluşur: (1) **MEVCUT** — `earthquake.py` içinde halen çalışan paneller; her satıra `file:line` referansı verildi. (2) **ÖNERİLEN** — PLAN_v2.md misyonlarından + peer-reviewed sismolojik literatürden türetilmiş; her satıra atıf verildi.

---

## Backlog Tablosu — MEVCUT (v1.0 - v1.14 commitli)

| ID | Özellik | Durum | Öncelik | Hedef sürüm | Menü/Yerleşim | Sorumlu ajan(lar) | Bağımlılık | Bilimsel arka plan |
|---|---|---|---|---|---|---|---|---|
| F-01 | 9 kaynaklı paralel deprem çekme (USGS, USGS-Fast, EMSC, AFAD, AFAD-Web, Kandilli, GFZ, IRIS, INGV) | Tamamlandı | 🔴 Yüksek | v1.1 | Veri katmanı (arka plan) | Veri Mühendisi | requests, ThreadPool | Multi-source aggregation; FDSN web services (IRIS 2019) |
| F-02 | Sliding-window dedup + NumPy haversine | Tamamlandı | 🔴 Yüksek | v1.13 | Veri katmanı | Vektörleştirme Uzmanı (Ajan 3) | NumPy O(n log n) | event-pairing window (Schorlemmer 2007) |
| F-03 | Canlı Radar — Plotly Scattermapbox (M-bantları renkli) | Tamamlandı | 🔴 Yüksek | v1.0 | Canlı Radar [earthquake.py:1147](../earthquake.py#L1147) | Render Uzmanı | Plotly + ESRI tiles | USGS magnitude color scale |
| F-04 | MTA Diri Fay 14.500 segment overlay (kayma türü rengi: SAD/SOD/T/N/AÇ) | Tamamlandı | 🔴 Yüksek | v1.9 | Canlı Radar [earthquake.py:836-887](../earthquake.py#L836) | Tektonik Veri Uzmanı (Ajan 4) | turkey_faults.geojson | MTA 2013 Diri Fay Haritası — Emre vd. 2013 |
| F-05 | PB2002 plaka sınırları + tip sınıflandırması (convergent/divergent/transform) | Tamamlandı | 🔴 Yüksek | v1.14 | Canlı Radar [earthquake.py:894-924](../earthquake.py#L894) | Tektonik Veri Uzmanı (Ajan 4) | tectonic_plates.geojson | Bird 2003, PB2002 model |
| F-06 | Bbox view-filter + lat/lon margin (fay perf) | Tamamlandı | 🟡 Orta | v1.9 | Canlı Radar [earthquake.py:1194-1205](../earthquake.py#L1194) | Vektörleştirme Uzmanı | — | spatial pre-indexing |
| F-07 | Kayan deprem listesi (3D virtualized) | Tamamlandı | 🟡 Orta | v1.0 | Canlı Radar sağ kolon [earthquake.py:1359-1410](../earthquake.py#L1359) | UI Uzmanı (Ajan 7) | — | — |
| F-08 | Günlük/Saatlik/Kümülatif aktivite grafiği | Tamamlandı | 🟡 Orta | v1.1 | Canlı Radar [earthquake.py:1531](../earthquake.py#L1531) | Render Uzmanı | — | Earthquake catalog visualization |
| F-09 | Büyüklük + Derinlik dağılımı | Tamamlandı | 🟡 Orta | v1.0 | İstatistik & Analiz [earthquake.py:1625](../earthquake.py#L1625) | Render Uzmanı | — | Catalog completeness viz |
| F-10 | Aktivite/Alarm skor kartı (4 bileşen: 24h yoğunluk, max-M, kaynak sayısı, Erzincan yakınlığı) | Tamamlandı | 🟡 Orta | v1.2 | İstatistik üst [earthquake.py:1759-1802](../earthquake.py#L1759) | Bilim Profesörü + UI Uzmanı | — | Composite seismic activity index — heuristic |
| F-11 | η Kümeleme Analizi (Zaliapin & Ben-Zion 2013) | Tamamlandı | 🔴 Yüksek | v1.3 | İstatistik altsekmesi [earthquake.py:2946-3052](../earthquake.py#L2946) | Bilim Profesörü (Ajan 8) | calc_etas_cache | Zaliapin & Ben-Zion JGR 2013 |
| F-12 | RTL Sismik Sessizlik (Sobolev & Tyupkin 1997) | Tamamlandı | 🔴 Yüksek | v1.3 | İstatistik altsekmesi [earthquake.py:3057-3145](../earthquake.py#L3057) | Bilim Profesörü (Ajan 8) | calc_rtl_cache | Sobolev & Tyupkin Volc Geol 1997 |
| F-13 | AMR Güç Yasası Hızlanma (Bowman 1998) | Tamamlandı | 🔴 Yüksek | v1.3 | İstatistik altsekmesi [earthquake.py:3151-3222](../earthquake.py#L3151) | Bilim Profesörü (Ajan 8) | calc_amr_cache | Bowman et al. JGR 1998 |
| F-14 | Uzamsal b-Değeri Haritası (MLE grid) | Tamamlandı | 🔴 Yüksek | v1.3 | İstatistik altsekmesi [earthquake.py:3224-3315](../earthquake.py#L3224) | Bilim Profesörü (Ajan 8) | calc_b_grid_cache | Wiemer & Wyss BSSA 2000; Aki 1965 |
| F-15 | Fay Sistemleri paneli (en yakın fay tablosu + histogram) | Tamamlandı | 🟡 Orta | v1.2 | Fay Sistemleri [earthquake.py:1717-1757](../earthquake.py#L1717) | Render Uzmanı | nearest_fault_vertex_distance | spatial join |
| F-16 | Astronomik panel — Anlık gök durumu kartları (Ay/Güneş/Jüpiter/Venüs + syzygy) | Tamamlandı | 🟢 Düşük | v1.10 | Astronomik [earthquake.py:3373-3423](../earthquake.py#L3373) | Bilim Profesörü (Ajan 8) | ephem | Cochran 2004; Métivier 2009 (zayıf korelasyon) |
| F-17 | Ay fazı vs deprem büyüklüğü scatter + spring-tide bantları | Tamamlandı | 🟢 Düşük | v1.10 | Astronomik [earthquake.py:3433+](../earthquake.py#L3433) | Bilim Profesörü | ephem | Tanaka 2012 (Tohoku tidal stress) |
| F-18 | Ay/gezegen çekim zaman serisi (ay_cekim, gezegen_cekim proxy) + deprem overlay | Tamamlandı | 🟢 Düşük | v1.10 | Astronomik | Bilim Profesörü | compute_environmental_features [L2279](../earthquake.py#L2279) | 1/d² gravitational proxy |
| F-19 | Günlük deprem sayısı FFT periodogramı (lunar 29.5d, M2 14.8d, yıllık 365.25d) | Tamamlandı | 🟢 Düşük | v1.10 | Astronomik | Bilim Profesörü | numpy.fft | Métivier 2009 spectral analysis |
| F-20 | Erken Uyarı Simülatörü — P/S/Rayleigh geri sayım | Tamamlandı | 🔴 Yüksek | v1.11 | Erken Uyarı [earthquake.py:3630-3915](../earthquake.py#L3630) | Bilim Profesörü + UI Uzmanı | haversine, math | Allen & Kanamori 2003 (EEW concept); GMPE Wald 1999 |
| F-21 | MMI Şiddet tahmini (basit GMPE I=1.5M−1.5log10(R)−3.5) | Tamamlandı | 🔴 Yüksek | v1.11 | Erken Uyarı [earthquake.py:3729-3735](../earthquake.py#L3729) | Bilim Profesörü | — | Wald et al. EQ Spectra 1999 (instrumental MMI) |
| F-22 | 15 şehir + manuel koordinat gözlemci seçici | Tamamlandı | 🟡 Orta | v1.11 | Erken Uyarı [earthquake.py:3651-3714](../earthquake.py#L3651) | UI Uzmanı | — | — |
| F-23 | "Bu süre içinde ne yapmalı?" eğitici 3 kart | Tamamlandı | 🟡 Orta | v1.11 | Erken Uyarı [earthquake.py:3871-3901](../earthquake.py#L3871) | UI Uzmanı | — | FEMA Drop-Cover-Hold |
| F-24 | Bilgi Havuzu — 3D Fay Mekaniği (5 fay tipi animasyonu) | Tamamlandı | 🟡 Orta | v1.4 | Bilgi Havuzu [earthquake.py:1857-2010](../earthquake.py#L1857) | Animasyon Uzmanı (Ajan 6) | Plotly Mesh3d + frames | Anderson 1905 fault classification |
| F-25 | Bilgi Havuzu — P/S/Rayleigh dalga eğitim animasyonu | Tamamlandı | 🟡 Orta | v1.4-v1.6 | Bilgi Havuzu [earthquake.py:2011+](../earthquake.py#L2011) | Animasyon Uzmanı | — | Lay & Wallace 1995 (seismology textbook) |
| F-26 | Erzincan Sanal Etki — 3D havza + dalga propagasyonu (4 senaryo) | Tamamlandı | 🟡 Orta | v1.6 | Bilgi Havuzu [earthquake.py:2137+](../earthquake.py#L2137) | Animasyon Uzmanı | Plotly Scatter3d | tarihsel Erzincan 1939 katalog |
| F-27 | @st.fragment refactor — 8 panel | Tamamlandı | 🔴 Yüksek | v1.12 | tüm paneller | Fragment Mimarı (Ajan 2) | Streamlit 1.33+ | menu-isolated rerun |
| F-28 | "Çalıştır" buton gating (heavy panel CPU koruması) | Tamamlandı | 🟡 Orta | v1.8 | İstatistik + Astronomik | UI Uzmanı | session_state | — |
| F-29 | Sidebar — 9 kaynak checkbox + zaman/yarıçap/M filtre | Tamamlandı | 🔴 Yüksek | v1.0 | Sidebar [earthquake.py:648-751](../earthquake.py#L648) | UI Uzmanı | — | — |
| F-30 | Raporlar paneli — TXT indir | Tamamlandı | 🟢 Düşük | v1.2 | Raporlar [earthquake.py:1807-1825](../earthquake.py#L1807) | UI Uzmanı | — | — |

---

## Backlog Tablosu — ÖNERİLEN (v1.15+ hedefli)

| ID | Özellik | Durum | Öncelik | Hedef sürüm | Menü/Yerleşim | Sorumlu ajan(lar) | Bağımlılık | Bilimsel arka plan |
|---|---|---|---|---|---|---|---|---|
| F-31 | ANA MENÜ üst horizontal pill (streamlit-option-menu) + sidebar expander reorganizasyonu | Aktif sprint | 🔴 Yüksek | v1.15 | Üst banner (header altı) — sidebar yerine [PLAN_v2.md:200-256](../PLAN_v2.md#L200) | UI Uzmanı (Ajan 7) | streamlit-option-menu paketi, requirements.txt | Nielsen Norman F-pattern; Fitts's Law (Fitts 1954) |
| F-32 | Plaka sınırı stilizasyonu — convergent için üçgen markerlar, divergent kesik çizgi, transform için ok glyphleri | Backlog | 🟡 Orta | v1.15 | Canlı Radar harita stili [PLAN_v2.md:115](../PLAN_v2.md#L115) | Render Uzmanı (Ajan 5) | PB2002 type alanı (F-05 tamam) | USGS plate boundary cartographic convention |
| F-33 | LOD (Level-of-Detail) — Douglas-Peucker zoom-bazlı fay simplification | Backlog | 🟡 Orta | v1.15 | Canlı Radar [PLAN_v2.md:114](../PLAN_v2.md#L114) | Render Uzmanı (Ajan 5) | shapely/rdp | Douglas & Peucker Cartographica 1973 |
| F-34 | Plotly figure cache @st.cache_data hash-bazlı | Backlog | 🟡 Orta | v1.15 | Tüm grafikler | Render Uzmanı + Fragment Mimarı | Streamlit cache | — |
| F-35 | Türkiye mikroplaka detayı: Anadolu/Ege/Karadeniz blok GPS hız vektörleri | Backlog | 🔴 Yüksek | v1.16 | Canlı Radar + Fay Sistemleri | Tektonik Veri Uzmanı (Ajan 4) | yeni plate_velocities.geojson | Reilinger et al. JGR 2006; Aktug et al. JGR 2009 |
| F-36 | Marmara denizi NAFZ alt-segment detayı | Backlog | 🔴 Yüksek | v1.16 | Canlı Radar | Tektonik Veri Uzmanı (Ajan 4) | yeni marmara_faults.geojson | Le Pichon et al. EPSL 2001; Şengör vd. 2005 |
| F-37 | Coulomb stress transfer (CFS) haritası — büyük olay sonrası komşu fay yüklenmesi | Backlog | 🔴 Yüksek | v1.17 | Fay Sistemleri (yeni altsekme) veya yeni "Stres" menüsü | Bilim Profesörü (Ajan 8) + Render | numpy linear algebra (Okada 1992 dislocation modeli) | King, Stein & Lin BSSA 1994; Stein Nature 1999 |
| F-38 | ETAS modeli ile artçı tahmini (intensity rate λ(t)=μ+Σ aftershock kernel) | Backlog | 🔴 Yüksek | v1.17 | İstatistik altsekmesi (η yanına) | Bilim Profesörü (Ajan 8) | scipy.optimize, mevcut calc_etas_cache | Ogata JASA 1988; Helmstetter & Sornette JGR 2002 |
| F-39 | Magnitude of Completeness (Mc) zaman serisi + b-değer evolution | Backlog | 🟡 Orta | v1.17 | İstatistik altsekmesi | Bilim Profesörü (Ajan 8) | mevcut calc_b_mle [L2934](../earthquake.py#L2934) | Wiemer & Wyss BSSA 2000; Mignan & Woessner CORSSA 2012 |
| F-40 | P/S/Rayleigh particle motion — fiziksel doğru (P longitudinal, S transverse, Rayleigh retrograde elliptik) | Backlog | 🟡 Orta | v1.16 | Bilgi Havuzu — dalga animasyonunun yenisi [PLAN_v2.md:128-132](../PLAN_v2.md#L128) | Animasyon Uzmanı (Ajan 6) | mevcut Plotly Cone/Scatter3d | Aki & Richards 2002 quantitative seismology |
| F-41 | SRTM 30m topografya — Erzincan sanal etki 3D terrain overlay | Backlog | 🟢 Düşük | v1.18 | Bilgi Havuzu — Erzincan simülasyonu | Animasyon Uzmanı (Ajan 6) | SRTM tile fetch, ~30MB lazy | NASA SRTM v3 (Farr et al. RG 2007) |
| F-42 | Foreshock pattern recognition — M2+ swarm + spatial clustering uyarısı | Backlog | 🟡 Orta | v1.18 | Erken Uyarı altsekmesi veya yeni "Öncü Tespit" | Bilim Profesörü (Ajan 8) | DBSCAN, sklearn | Bouchon vd. Science 2013; Brodsky & Lay Science 2014 |

---

## v1.15 Sprint (yakın hedef — sırayla)

| Sıra | ID | Özellik | Süre tahmini | Engelleyici |
|---|---|---|---|---|
| 1 | **F-31** | ANA MENÜ üst horizontal pill + sidebar reorganizasyonu | 1-2 commit | requirements.txt güncellemesi (streamlit-option-menu); ardından `earthquake.py:812-828` silinip `:794` öncesi yerleştirilecek (PLAN_v2.md kararı) |
| 2 | **F-32** | Plaka sınırı stilizasyonu (üçgen/kesik/ok) | 1 commit | F-05 zaten v1.14'te tamamlandı, sadece glyph layer eklenecek |
| 3 | **F-34** | Plotly figure cache | 1 commit | F-31 sonrası fragment'ler stabil olduğunda — figure_hash key'i kararlı olmalı |

**Sprint sonu kullanıcı doğrulaması:** Menü geçişi <500ms (Profile Detective ölçümünden gelecek) + plaka renklerinin görsel olarak tip ayırması.

---

## Bilim Profesörü'ne (Ajan 8) sorulacaklar

| Soru-ID | Özellik | Soru |
|---|---|---|
| Q-SCI-1 | F-37 (Coulomb stress) | Okada 1992 dislocation modeli yerine basitleştirilmiş 2D CFS (King, Stein & Lin BSSA 1994 formülasyonu) yeterli mi? Friction coefficient µ' için Türkiye için tipik 0.4 mi yoksa NAFZ için ölçülmüş özel değer var mı? |
| Q-SCI-2 | F-38 (ETAS) | Ogata 1988 ETAS parametreleri (K, c, p, α) Türkiye için kalibre edilmiş bir literatür var mı (örn. Yıldırım vd.), yoksa generic California değerleri (Helmstetter & Sornette 2002) mi kullanılsın? |
| Q-SCI-3 | F-39 (Mc evolution) | Mc tahmini için MAXC (Wiemer & Wyss 2000) mi yoksa GFT (Wiemer & Wyss 2000) mi yoksa b-positive (van der Elst 2021) mi? Hangi yöntem Erzincan kataloğu için daha stabil? |
| Q-SCI-4 | F-21 (MMI) | Mevcut `I = 1.5M − 1.5log10(R) − 3.5` formülünün kaynağı belirsiz; Wald 1999 ile değiştirmeli mi? Türkiye için Akkar & Bommer 2010 GMPE'sine geçiş gerekçesi var mı? |
| Q-SCI-5 | F-42 (Foreshock) | Bouchon 2013 öncü swarm tanımı (Mmax−1.0 birim altı kümeler) operasyonel demosu için minimum N=? eşik ve maksimum yarıçap (km) seçimi? False positive oranını nasıl raporlayalım? |
| Q-SCI-6 | F-16/17/18/19 (Astro) | Astronomik korelasyon paneli "keşfedici" disclaimer ile durabilir mi, yoksa Schuster testi (Schuster 1897) + p-değer eklenmeli mi? Métivier 2009 metodolojisi uygulanabilir mi? |
| Q-SCI-7 | F-35 (mikroplaka GPS) | Reilinger 2006 hız alanı en güncel mi yoksa daha yeni bir IGS+TUSAGA çözümü tercih edilsin mi (Aktug 2013, Özdemir 2020)? Hız vektörünün lokal jeodezik referansı ne olmalı (Eurasia-fixed mi)? |

---

## UI Uzmanı'na (Ajan 7) sorulacaklar

| Soru-ID | Özellik | Soru |
|---|---|---|
| Q-UI-1 | F-37 (Coulomb stress) | CFS haritası Fay Sistemleri menüsünde altsekme mi olmalı, yoksa yeni bir "Stres & Yüklenme" menüsü mü? Mevcut 8-pill bar 9. öğeyi kaldırır mı (Hick's law cezası)? |
| Q-UI-2 | F-38 (ETAS aftershock rate) | ETAS λ(t) tahmini İstatistik & Analiz altsekmesi olarak mı, yoksa Erken Uyarı'nın yanına "Artçı Beklentisi" olarak mı? Hangi panel zihinsel olarak yakın? |
| Q-UI-3 | F-32 (plaka glyph) | Üçgen/kesik/ok glyphleri Plotly Scattermapbox layer'ında zoom-bağımlı görünmeli mi (zoom<4 gizle), yoksa her zaman mı? Mobil (375px) viewport'ta üçgen markerlar çakışıyor mu? |
| Q-UI-4 | F-31 (üst pill) | `streamlit-option-menu` 80KB ek bağımlılık — alternatif olarak `st.tabs` yatay tab kullanmak yeterli mi (PLAN_v2.md (C+E) önerdi ama trade-off değerlendir)? |
| Q-UI-5 | F-39 (Mc time-series) | b-değer + Mc zaman serisi grafiği için optimum pencere boyutu (gün cinsinden) sidebar slider mı yoksa otomatik (sqrt-N kural) mu olmalı? Slider yorucu olur mu? |
| Q-UI-6 | F-41 (SRTM terrain) | 30MB lazy download progress nasıl gösterilmeli? `st.spinner` mi `st.progress` mi? İlk çağrıda 5-10s gecikme kabul edilebilir mi yoksa background prefetch mi? |
| Q-UI-7 | Genel | Mevcut renkler: M-bantları, fay kayma türleri, plaka tipleri — 3 ayrı renk dilini kullanıcı karıştırır mı? Renk-kör (deuteranopia) testi yapıldı mı? |

---

## Donmuş / İptal

| ID | Özellik | Neden donduruldu |
|---|---|---|
| FZ-01 | Mirror üst+alt navigasyon (PLAN_v2.md (B) seçeneği) | UI Uzmanı 3/10 verdi: Hick's Law cezası + görsel kirlilik. F-31 ile yerine üst-pill (C+E) seçildi [PLAN_v2.md:208](../PLAN_v2.md#L208) |
| FZ-02 | `st.navigation` multi-page mimarisi (PLAN_v2.md (E) seçeneğine alternatif) | Multi-page'e geçiş tüm `session_state` paylaşımını ve `df` global'ini kırar; refactor maliyeti yüksek. `option_menu` daha hafif [PLAN_v2.md:211](../PLAN_v2.md#L211) |
| FZ-03 | Astronomik panel "deprem tahmini" iddiası | Cochran 2004 ve Métivier 2009 zayıf korelasyon raporladı ama nedensellik yok. Panel "keşfedici" disclaimer ile bırakıldı; tahmin iddiası açıkça reddedildi [earthquake.py:3346-3350](../earthquake.py#L3346) |

---

## Eklenmesi düşünülmeyen ama proaktif öneriler (Tasarım Master notu — Q-DM)

Aşağıdakiler kullanıcı henüz talep etmediği için **tabloya eklenmedi**, fakat Bilim Profesörü'ne sorulup F-43+ olarak eklenebilir:

- **Slip deficit / interseismic coupling haritası** — Reilinger 2006 GPS verisinden NAFZ kilitlenme oranları; b-haritasının tektonik karşılığı.
- **Paleosismik veri layer** — son 2000 yılın büyük Erzincan/Bingöl/Marmara depremleri timeline'ı (Hubert-Ferrari 2002 NAFZ trench data).
- **Tsunami uyarı paneli** — Akdeniz subduction olaylarında lokal tsunami varış süresi (Yolsal & Taymaz 2010).
- **Magnitude conversion ML↔MW↔Mb** — Scordilis 2006 global regresyon; Kandilli'nin verdiği ML ile USGS MW arasında köprü.

**Karar bekleniyor:** Bu 4 öneri kullanıcıya (orkestrasyon şefi) bir kerelik tanıtılsın, hangileri v1.18'e girer onaylansın.

---

## Versiyonlama notu

PLAN_v2.md'nin "Önerilen Sıralama" bölümü (satır 261-282) v1.12 (fragment), v1.13 (vektörleştirme), v1.14 (tektonik veri) başarıyla commitledi. Bu backlog **Adım 9** (Tasarım Master: feature backlog + spec dokümanları) çıktısıdır. Sonraki adımlar her feature için ayrı `features/SPEC_<id>.md` dokümanı (PLAN_v2.md:196).

**Co-Authored-By:** Claude Opus 4.7 (1M context) — Tasarım Master / Orkestrasyon Şefi rolünde
