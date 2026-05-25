# DepremRadarı — Performans + Veri Kalitesi + Animasyon Misyonu (v1.12+)

**Tarih:** 25 Mayıs 2026
**Sebep:** Menüler arası geçişlerde 20-30 saniyelik gecikme. Kullanıcı önceliği: hız + animasyon gerçekçiliği + harita doğruluğu + zengin plaka/fay detayı.

---

## 🔬 Mevcut Durum Analizi

### Geçmişte yapılmış performans önlemleri (kullanıcı geçmişi)
| Commit | Ne yapıldı |
|---|---|
| `0f09369` | Heavy scientific calc cache |
| `d713e0a` | fetch_all cache bust |
| `c502850` | 3D kamera zoom + heatmap |
| `53a5c70` | Plotly native auto-play animations |
| `c20827f` (v1.7) | ephem + perf opt |
| `70a0e78` (v1.8) | API timeout + Run buton |
| `7a77817` (v1.9) | sidebar lazy + bbox + B-grid vektör |

→ **Örüntü:** Cache + lazy loading + gerçek veri (MTA Diri Fay, haversine) + Run-button-gating.

### Mevcut bottleneck adayları (tahminim, ölçülmeli)

| Sıra | Bottleneck | Konum | Tahmini Etki |
|---|---|---|---|
| 1 | **Fragment yok** — menü değişimi tüm script'i re-run ediyor | `if active_menu ==` blokları | 🔥🔥🔥 EN BÜYÜK kazanç |
| 2 | `df.apply(haversine)` cache miss'te tüm depremler için | [earthquake.py:432](earthquake.py#L432) | 🔥🔥 |
| 3 | `st_autorefresh` her N saniyede tam re-run | [earthquake.py:738](earthquake.py#L738) | 🔥🔥 (menü etkileşimi sırasında çakışır) |
| 4 | `compute_environmental_features` ephem, korelasyon matrisinde tüm depremler için | [earthquake.py:2172](earthquake.py#L2172) | 🔥🔥 (zaten butonla gated ama korelasyon hâlâ yavaş) |
| 5 | Fay render: 14,500 segment view-bazlı filter ama hâlâ binlerce çizgi | [earthquake.py:1116-1142](earthquake.py#L1116-L1142) | 🔥 |
| 6 | Plotly figure cache yok — aynı veri için figure yeniden üretilir | tüm panellerdeki `go.Figure` | 🔥 |
| 7 | `load_fault_lines` 7.9 MB GeoJSON parse (session başına 1 kez) | [earthquake.py:832](earthquake.py#L832) | düşük (cache'li) |

### Veri kalitesi mevcut durumu
- ✅ MTA Diri Fay (Türkiye): 14,500 segment, kayma türü renkleri (sağ/sol/normal/ters/SAD/SOD)
- ✅ PB2002 Plaka sınırları: 52 plaka, 241 sınır (global; Peter Bird 2001-2002 + Mueller 1987 + Lemaux 2002)
- ⚠️ **Plaka sınırları varsayılan KAPALI** — kullanıcı keşfetmiyor
- ❌ Türkiye'nin mikroplakaları (Anadolu Bloğu, Ege Bloğu, Karadeniz Bloğu) ayrıntısı yetersiz
- ❌ Marmara denizi fay detayı az
- ❌ Ege denizi açılma çatlağı detayı yok
- ❌ Levha sınırlarının tipi (convergent/divergent/transform) gösterilmiyor
- ⚠️ Animasyonlar: Erzincan 3D simülasyon var, ama P/S/Rayleigh particle motion mevcut. Genelleştirilebilir.

---

## 🎯 Misyon Önceliği

Kullanıcının üç önceliği (sıralanmış):
1. **🚀 Hız** — menü geçişleri anında olmalı (mevcut: 20-30s)
2. **🗺️ Doğruluk** — fay/plaka sınırlarının gerçek yerlerinde olması, mikroplakalar dahil detay
3. **🎬 Gerçekçilik** — animasyonların fiziksel doğruluğu (P-pull, S-shear, Rayleigh-elliptic)

---

## 👥 Ajan Rosterı (9 alt-ajan — 6 teknik + 3 yeni)

### 🔍 Ajan 1 — Profil Dedektifi (Profile Detective)
**Görev:** Gerçek bottleneck'i ÖLÇ. Tahminle çalışma.

- [profile_st.py](../scratch/profile_st.py) instrumentation'ı tekrar uygula
- Streamlit'i başlat, 3 farklı menü geçişi yap, `/tmp/st_perf.log` topla
- 5 ağır işlem'i ms cinsinden raporla
- Çıktı: **`PROFILE_REPORT.md`** — net "X ms şu yerden geliyor, Y ms şu yerden" tablosu

**Çıktı türü:** Salt rapor (kod değiştirmez). Sonraki ajanlar bu rapora göre karar verir.

---

### ⚡ Ajan 2 — Fragment Mimarı (Lazy Loading Architect) — EN BÜYÜK KAZANÇ
**Görev:** Her panel için `@st.fragment` refactor.

- Streamlit'te `@st.fragment` dekoratörlü fonksiyonlar **rerun'u o fragment'e limitler** — üst script çalışmaz
- Şu an sadece Bilgi Havuzu'nda var; 8 panele yayılacak:
  - Canlı Radar, İstatistik & Analiz (üst), İstatistik (alt-bilimsel), Fay Sistemleri,
    Astronomik Analiz, Erken Uyarı, Bilgi Havuzu (zaten var), Sistem & Veri, Raporlar
- Her panel: `@st.fragment` ile sarılmış fonksiyon, `active_menu` if'inde fonksiyon çağrısı
- Cross-fragment state için `st.session_state` kullan
- Risk: `df` ve global state'in fragment içinden erişimi — dikkatli refactor

**Çıktı türü:** Tek büyük commit (~v1.12). [earthquake.py](earthquake.py) ~50-100 satır değişikliği. Bu adımdan sonra menü geçişi <500ms olmalı.

---

### 🚀 Ajan 3 — Vektörleştirme Uzmanı (Vectorization Engineer)
**Görev:** Kalan `df.apply` Python döngülerini NumPy'a çevir.

- [earthquake.py:432](earthquake.py#L432) `df["uzaklik_km"] = df.apply(haversine, axis=1)` → NumPy vektörleştir (zaten B-grid'de yapıldı, kopyala)
- [earthquake.py:2384](earthquake.py#L2384) ve [earthquake.py:3321](earthquake.py#L3321) `astro_df.apply(compute_environmental_features)` → batch ephem (zor; ephem.Observer state'li)
- [earthquake.py:456](earthquake.py#L456) `df["event_id"] = df.apply(...)` → string concat vektörleştir
- Dedup loop (line 437-444) → KDTree veya pandas merge_asof ile O(n²) → O(n log n)

**Çıktı türü:** Bir veya iki commit (~v1.13). Vektörleştirilebilir her döngü için ölçülmüş hızlanma raporu.

---

### 🗺️ Ajan 4 — Tektonik Veri Uzmanı (Tectonics Data Scientist)
**Görev:** Daha detaylı, daha doğru plaka & fay verisi.

- **Plaka sınırı tipleri:** PB2002 [GeoJSON'a](tectonic_plates.geojson) `Type` alanı boş — convergent/divergent/transform sınıflandırması yapıp renklendir (kırmızı=collision, mavi=divergent, sarı=transform)
- **Türkiye mikroplakalar:** Anadolu, Ege, Karadeniz, Kuzey Arap blok detayı için Reilinger 2006 + Aktug 2009 GPS hız vektörü tabanlı sınırlar
- **Marmara denizi fay detayı:** NAFZ Marmara segment kırıkları için MTA + Le Pichon 2001 datası — ekstra GeoJSON
- **Plaka sınırlarını varsayılan AÇIK yap** — kullanıcı keşfetmiyor
- **Plaka hareket vektörleri** (oklarla): UNAVCO veya NUVEL-1A datası ile hız vektör layer
- **Çıktı dosyaları:** `tectonic_plates_v2.geojson` (Tip alanı dolu), `marmara_faults.geojson`, `plate_velocities.geojson`

**Çıktı türü:** 1 büyük commit (~v1.14). Yeni veri dosyaları + render güncellemeleri.

---

### 🎨 Ajan 5 — Harita Render Uzmanı (Map Render Specialist)
**Görev:** Plotly harita performansı + görsel kalite.

- **LOD (Level of Detail) zoom-bazlı simplification:** Geniş zoom'da fay segmentleri Douglas-Peucker ile %90 azalt, dar zoom'da tam çöz
- **Plaka sınırı stilizasyonu:** Convergent (kırmızı, çift çizgi + üçgen marker), Divergent (mavi, kesik çizgi), Transform (sarı, oklar)
- **Fay rengi netleştirme:** Mevcut neon renkler ama legend'da `kayma_aciklama` ile birlikte (kullanıcı renge bakıp anlamayabilir)
- **Plotly figure cache:** `@st.cache_data` ile figure üretimini cache'le (figure boyut hash'ine göre)
- **Hover tooltip iyileştir:** Fay adı + segment + son depremin tarihi + uzunluk
- **Mapbox style daha kontrastlı:** ESRI World Imagery + ek labels layer

**Çıktı türü:** 1-2 commit (~v1.15). Görsel/render kalitesi gözle anlaşılır artış.

---

### 🌊 Ajan 6 — Animasyon Uzmanı (Wave Physics Animator)
**Görev:** P/S/Rayleigh dalga animasyon gerçekçiliği.

- **Particle motion** (parçacık hareketi):
  - P-dalga: longitudinal (sıkışma-genleşme okları)
  - S-dalga: transverse (yatay/dikey shear okları)
  - Rayleigh: elliptical retrograde (saat tersi elips)
- **3D wave propagation:** Hipocenter'dan iki yarım küre dalga (radial expanding shells) — gerçek hızlarda
- **Topografya:** SRTM 30m elevation (Türkiye için ~30MB) ile 3D terrain overlay
- **Genişletilebilir Erzincan senaryosu:** Şehir seçilebilir (Erzincan, Bingöl, Düzce, Van — gerçek geçmiş depremler ile)
- **Animasyon kontrolü:** Hız (1x/2x/5x), pause/resume, frame slider

**Çıktı türü:** 1-2 commit (~v1.16). Bilgi Havuzu paneli ve Erken Uyarı timeline'ı iyileştirme.

---

### 🎨 Ajan 7 — UI Uzmanı (UX/UI Specialist) — YENİ
**Görev:** Kullanıcı arayüzü hijyeni — menü konumu, sidebar düzeni, bilgi mimarisi, F-shaped reading, Fitts's Law, mobil responsive. Veri-yoğun dashboard ve Streamlit/Plotly tabanlı analitik uygulamalarında uzman (Tufte + Nielsen Norman ilkeleri).

**Sorumluluk alanı:**
- Menü/navigation konumu (sidebar vs. top horizontal vs. tab)
- Sidebar filtre hiyerarşisi (live filter vs. expander vs. popover)
- Renk paleti, tipografi, beyaz alan, ikon tutarlılığı
- Mobil viewport davranışı (375-768px breakpoint)
- Cognitive load ölçümü — bir sismolog gün boyu kullanırken hangi etkileşim yorucu?

**Karar verme yetkisi:** Konum/yerleşim/etkileşim deseni kararlarında **belirleyici oy**. Diğer ajanlar yeni özellik eklediğinde UI Uzmanı yerleşim onayı verir.

**İlk çıktı:** ANA MENÜ konum kararı (aşağıda § UI Uzmanı — ANA MENÜ Konum Kararı).

**Çıktı türü:** Rapor + spesifik kod patch'leri (Streamlit primitive seçimi dahil).

---

### 🔬 Ajan 8 — Bilim Profesörü (Earthquake Engineer & Geology Prof.) — YENİ
**Görev:** Uygulamadaki bilimselliği en üst düzeyde kontrol etsin — bilimsel ve yenilikçi yaklaşımlar önersin, yeni özellikler eklenecekse bunları önerip uygulatsın, hesaplamaları ve bilimselliği denetlesin.

**Sorumluluk alanı:**
- **Sismolojik doğruluk:** Büyüklük dönüşümleri (ML↔MW↔Mb), Gutenberg-Richter b-değer hesabı, Omori yasası (artçı sönümleme), ETAS modeli, RTL/AMR/η istatistikleri
- **Tektonik doğruluk:** Plaka sınırı sınıflandırması (convergent/divergent/transform), GPS hız vektörleri (Reilinger, Aktug), seismic gap teorisi
- **Astronomik korelasyon bilimi:** Ay/güneş çekiminin gerçek deprem korelasyonu var mı (Tanaka 2012, Métivier 2009) — sözde-bilimden ayır
- **Erken uyarı fiziği:** P-S dalga hız farkı, S-P warning window hesabı, magnitude estimation (Pd/Pv tabanlı)
- **Yeni özellik önerileri:** b-grid time evolution, slip deficit haritası, Coulomb stress transfer (CFS), foreshock pattern recognition (M2+ kümeleri)

**Karar verme yetkisi:** Bilimsel formül/algoritma doğruluğunda **veto yetkisi**. Bir özellik bilimsel olarak yanlışsa veya yanıltıcıysa engelleyebilir, atıf/uyarı ekletebilir.

**İlk çıktı görevi (önerilir):** Mevcut [earthquake.py](earthquake.py) içindeki bilimsel hesaplamaların (b-değer, η, RTL, AMR, ephem-deprem korelasyonu) doğruluğunu denetle — bulguları `SCIENCE_AUDIT.md`'ye yaz.

**Çıktı türü:** Bilimsel denetim raporu + yeni özellik tasarım dokümanı + atıf listesi (peer-reviewed kaynaklar).

---

### 🎭 Ajan 9 — Tasarım Master / Orkestrasyon Şefi (Application Design Master) — YENİ
**Görev:** Uygulamaya önerilen özelliklerin **nereye, nasıl** ekleneceğine karar versin. Yeni özellik talep edip diğer ajanlara sorsun. Gelen önerileri değerlendirip uygulamaya soksun.

**Sorumluluk alanı:**
- **Feature routing:** Yeni özellik hangi menüye girer? (örn. Coulomb stress → Fay Sistemleri mi, Bilim Profesörü Paneli mi yeni mi?)
- **Coordination:** UI Uzmanı + Bilim Profesörü + teknik ajanlar (Vektörleştirme, Render, vs.) arasında roundtrip
- **Backlog yönetimi:** Hangi özellik bir sonraki sürümde? Hangi v1.16'da? Hangi v2.0?
- **Trade-off karar:** Bir özelliğin bilimsel değeri yüksek ama UI yükü ağır → ya pas geç ya da farklı menüye yerleştir
- **Dilek listesi yönetimi:** Kullanıcının "şunu da ekleyelim" demediği ama proaktif öneriler

**Karar verme yetkisi:** Feature placement (nerede görünecek) ve **prioritization** (ne zaman). UI Uzmanı ile mimari konum, Bilim Profesörü ile bilimsel öncelik konuşur.

**İş akışı:**
1. Bilim Profesörü "X özelliği eklensin" der → Tasarım Master değerlendirir
2. Tasarım Master UI Uzmanı'na "X nereye konsun?" diye danışır
3. UI Uzmanı yerleşim önerir → Tasarım Master onaylar
4. Teknik ajana (Render/Vektör/Animasyon) "şu satırlardan, şu Streamlit primitive'iyle uygula" tasklar
5. Tasarım Master + orkestrasyon şefi (= kullanıcı) son onay verir

**Çıktı türü:** Feature spec dokümanları (`features/SPEC_<name>.md`), her özellik için 1 sayfa: gerekçe + UI yerleşimi + bilimsel arka plan + teknik uygulama bağlantısı.

---

## 🎨 UI Uzmanı — ANA MENÜ Konum Kararı (Ajan 7'nin ilk raporu — 25 May 2026)

**Karar: (C) + (E) hibrit** — `streamlit_option_menu` ile **ana içerik alanının ÜSTÜNDE horizontal pill bar**; sidebar tamamen filtre/ayar paneline dönüşür. Navigation ve configuration iki ayrı zihinsel görevdir; aynı dikey eksende sıkıştırılırsa hem F-shaped tarama bozulur hem de Fitts's Law'a göre menü hedefi pahalı hale gelir.

### Konum değerlendirmesi (1=zayıf, 10=güçlü)
| Seçenek | Skor | Not |
|---|---|---|
| (A) Sidebar üstü | 6/10 | Hızlı kazanç ama ayar/menü zihinsel karışıklığı sürer; mobilde sidebar default kapalı |
| (B) Mirror üst+alt | 3/10 | Hick's Law cezası, görsel kirlilik |
| **(C) Üst horizontal pill** | **9/10** | F-pattern uyumu, header altı, metrik üstü |
| (D) Üst sidebar + breadcrumb | 5/10 | 8 kardeş panel için fazladan yük |
| **(E) `option_menu` ile (C)'yi uygula** | **9/10** | `st.navigation` multi-page'e zorlar (refaktör pahalı), `option_menu` daha hafif |
| (F) Alternatif | — | Yok, (C+E) yeterli |

### Sidebar reorganizasyonu (yan ürün)
Şu an sidebar "ayar çöplüğü + gizli menü" — UI Uzmanı önerisi (görsel yük ~%60↓):
- **Üstte sabit (live filter):** Yarıçap, Min. Büyüklük, Zaman Aralığı
- **`st.expander("Görünüm")`:** Tema, Harita Stili, Fay/Plaka checkbox
- **`st.expander("Veri Kaynakları (N/9)")`** veya **`st.popover("Kaynaklar")`:** 9 checkbox + "Hepsini seç/bırak"
- **`st.expander("Otomatik Yenileme & Sistem")`:** refresh interval + changelog notu

### Uygulama planı (concrete)
```python
# requirements.txt → streamlit-option-menu ekle
from streamlit_option_menu import option_menu

# earthquake.py — 812-828 SİL (mevcut sidebar radio bloğu)
# earthquake.py — 794'ten ÖNCE (metrik kartları öncesi) ekle:
active_menu = option_menu(
    None,
    ["Canlı Radar","İstatistik","Fay","Astronomi","Erken Uyarı","Bilgi","Sistem","Rapor"],
    icons=["globe","bar-chart","compass","moon-stars","exclamation-triangle","mortarboard","gear","file-text"],
    orientation="horizontal", default_index=0,
    styles={"container":{"padding":"0","background-color":"transparent"},
            "nav-link-selected":{"background-color":"#1976d2"}})

# earthquake.py — 720-751 → 3 st.expander içine sar
```

### Risk/Kazanç
| Kriter | Mevcut (sidebar-alt) | (C) üst pill |
|---|---|---|
| F-shaped okuma | Kötü (F dışında) | İyi (üst banda hizalı) |
| Fitts's Law mesafesi | 800-1200 px scroll | ~60 px sabit |
| Mobil (sidebar gizli) | 2 tıklama + scroll | 1 tıklama, görünür |
| Cognitive load | Yüksek (ayar/menü karışık) | Düşük (ayrık eksen) |
| Risk | — | `option_menu` 80KB bağımlılık + CSS ince ayar |

### Sismolog gözüyle
Sidebar-alt en yorucu seçenek: her panel değişiminde göz 800px dikey kayar, fare yo-yo yapar, saatte yüzlerce kez. **Üst horizontal pill**'de menü + içerik aynı bakış alanında — sakkad mesafesi 60-100px. Navigasyon "otomatikleşir", bilişsel yük filtre kararlarına ayrılabilir.

### Uygulama Sırası (v1.15 commit hedefi)
1. `pip install streamlit-option-menu` + requirements.txt güncelle
2. earthquake.py:812-828 sil; 794 öncesi `option_menu` bloğunu yerleştir
3. earthquake.py:720-751 → 3 `st.expander` (Görünüm / Veri Kaynakları / Sistem)
4. Mobil test: Chrome DevTools 375px viewport, pill bar wrap davranışı
5. Commit: `v1.15(ui): ANA MENÜ üst horizontal pill + sidebar expander reorganizasyonu`

---

## 📋 Önerilen Sıralama

```
Adım 1 — Profil Dedektifi raporu (1 oturum)          [✅ v1.12 fragment ile çoğu bitti]
            ↓
Adım 2 — Fragment Mimarı refactor                    [✅ v1.12 commitlendi]
            ↓
        Kullanıcı doğrulama: "menü geçişi hızlandı mı?"
            ↓
Adım 3 — Vektörleştirme                              [✅ v1.13 commitlendi]
            ↓
Adım 4 — Tektonik Veri (plaka tipi + stilizasyon)    [✅ v1.14 commitlendi]
            ↓
Adım 5 — UI Uzmanı: ANA MENÜ üst pill + sidebar expander  ◀ ŞIMDI BURADA (v1.15 hedef)
            ↓
Adım 6 — Harita Render (LOD + figure cache + plaka stili)
            ↓
Adım 7 — Animasyon Uzmanı (Particle motion + 3D)
            ↓
Adım 8 — Bilim Profesörü: SCIENCE_AUDIT.md (b-değer, η, RTL, AMR, ephem doğrulama)
            ↓
Adım 9 — Tasarım Master: feature backlog + spec dokümanları
```

**Toplam tahmin:** 8-10 commit, v1.12 → v1.18+

---

## ⚠️ Risk Notları

- **Fragment refactor (Ajan 2)** en büyük refactor'dır — global state (`df`, `FAULT_LINES`, vs.) fragment içine geçirilmeli. Yanlış yapılırsa veri sync sorunları doğar
- **SRTM topografya 30MB+** dosya — repo'ya commit etmek yerine lazy download / .gitignore stratejisi
- **Plotly cache'in dezavantajı:** figure objeleri büyük, RAM tüketir — TTL ile sınırla
- **Yeni veri dosyalarının lisansı:** PB2002 (CC0), MTA (kaynak göster), GEM-GAF (CC-BY) — atıfları korumak gerek

---

## ❓ Onay Gerekli Sorular

1. **9 ajan rosterı uygun mu?** UI Uzmanı + Bilim Profesörü + Tasarım Master eklenmiş hali ile.
2. **UI Uzmanı'nın ANA MENÜ konum kararı (C+E hibrit):** Onaylanıyor mu? `streamlit-option-menu` bağımlılığı kabul mü?
3. **Sidebar reorganizasyonu** (3 expander + üst sabit live filter): Aynı commit'te mi (v1.15) yoksa ayrı v1.15a/v1.15b commit'lere mi bölünsün?
4. **Veri eklenmesi:** Yeni dosyalar repo'ya commit edilsin mi (boyut artar) yoksa runtime'da download edilsin mi?
5. **Bilim Profesörü ilk görev:** `SCIENCE_AUDIT.md` denetimi v1.15'ten önce mi (UI'dan önce denetim) yoksa sonra mı?
6. **Sürüm stratejisi:** Her ajan bir commit + sürüm (v1.12, v1.13, ...) mi, yoksa "Performans Misyonu v2" tek büyük versiyon olarak mı?
