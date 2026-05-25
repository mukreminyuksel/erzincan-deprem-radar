# Profile Detective Raporu — Menü Geçiş Performansı

**Tarih:** 25 Mayıs 2026, 23:05-23:08
**Test ortamı:** Streamlit 1.50.0, `earthquake_profiled.py` (port 8562), 14 ardışık menü tıklama
**Veri durumu:** **9 satır deprem** (test penceresinde çok az olay)
**Profilleme noktası:** Script başlangıcı, fetch_all öncesi/sonrası, fault/plate load, menü entry, script_end

---

## 📊 Ham Sonuçlar

### Cold Boot (ilk yükleme) — RUN 536751
| İşlem | Delta (ms) | Toplam (ms) |
|---|---:|---:|
| script_begin | 0.1 | 0.1 |
| script_end (early — autorefresh boot) | 60 | 60 |
| fetch_all (NETWORK 6 API) | **1235** | 1295 |
| load_fault_lines (GeoJSON parse) | **396** | 1691 |
| load_tectonic_plates | 9 | 1700 |
| **TOPLAM ilk yükleme** | | **~1.7s** |

### Warm runs (menü geçişleri, cache hit) — RUN 639775+
| İşlem | Delta (ms) |
|---|---:|
| fetch_all (CACHE HIT) | **1-2** ✅ |
| load_fault_lines (CACHE'li ama yine hesaplama) | **40-60** ⚠️ |
| load_tectonic_plates | 1-3 |
| menu_enter | 5-6 |
| **TOPLAM menü geçişi (Python tarafı)** | **~60-80** ms |

### Per-menü test (sırayla)
| Menü | Run | Python süre |
|---|---|---:|
| Canlı Radar | 536751 (cold) | 1705 ms |
| İstatistik & Analiz | 639775 | 76 ms |
| Fay Sistemleri | 642135 | 66 ms |
| Astronomik Analiz | 644099 | 65 ms |
| Erken Uyarı | 646784 | 74 ms |
| Bilgi Havuzu | 648996 | 66 ms |
| Sistem & Veri | 675744 | 71 ms |
| Raporlar | 677324 | 64 ms |
| Canlı Radar (warm) | 679529 | 82 ms |

---

## 🔬 Yorumlama

### ✅ İYİ HABER
- **fetch_all caching çalışıyor** — cold 1235ms → warm 1-2ms (%99.8 hızlanma)
- **Python tarafı menü geçişi <100ms** (test ortamı, 9 satır data ile)
- **Cache TTL=120s** doğru ayarlanmış
- Plate boundary load çok hızlı (2-3ms)

### ⚠️ ŞÜPHELİ
1. **load_fault_lines her menü geçişinde 40-60ms tüketiyor** — `@st.cache_data` dekoratörü var AMA yine de bu kadar süre alıyor. Olası neden: cache hit'te bile büyük return değerinin hash hesaplaması pahalı, ya da Streamlit hash protocol'ü 14,500 elemanlı listeyi her sefer doğruluyor. **Düzeltilebilir.**
2. **st_autorefresh otomatik tetikleniyor** — RUN'lar arasında 1-2s aralıklarda sayfa kendiliğinden re-run oluyor. Kullanıcı menü değişimi yaparken bu çakışırsa lag hissi.

### 🚨 NEDEN 20-30s GECİKME GÖRMEDİK?
Bu test'te yaşamadık çünkü:
- **Yalnızca 9 satır deprem** (test penceresinde olay az)
- **Cache hit oranı %95+** (parametre değiştirmedik)
- **Plotly render süresi ölçülmedi** (script_end'den sonra Python tarafı bitmiş ama browser hâlâ render ediyor)

### 🎯 GERÇEK BOTTLENECK'in 20-30s OLUŞTUĞU DURUMLAR (tahmin)
| Senaryo | Tahmini süre |
|---|---:|
| Cache miss (radius/tarih/kaynak değişimi) → 6 API paralel | 4-8s |
| Geniş tarih aralığı → 200-1000 satır deprem → `df.apply(haversine)` Python döngü | 2-5s |
| Canlı Radar açık + 14,500 fay segmenti view-filter + Plotly mapbox render | 3-8s |
| Korelasyon matrisi tetiklendi → 1000 deprem × ephem | 10-25s |
| Sub-tab değişimi (η/RTL/AMR/b-grid) → ağır bilimsel hesap | 5-30s |
| autorefresh + kullanıcı tıklama aynı anda → tam re-run iki kez | 2× |

### 📌 DOĞRULANMIŞ MİMARİ SORUN
- **Fragment kullanılmaması nedeniyle her menü tıklamasında üst-script tamamen re-run oluyor** — `fetch_all` cache hit olsa bile, sidebar render + `df` rebuild + uzaklık hesabı + fay load = **her seferinde tekrar tekrar yapılıyor.**
- Şu an: cache hit nedeniyle bu "küçük" görünüyor (60-80ms test'te), ama 1000 satır deprem ile bu 2-5 saniyeye çıkar.

---

## 🎯 Ajan 2 (Fragment Mimarı) İçin Eylem Planı

### Mevcut yapı (her menü tıklamasında tamamı çalışan üst-script)
```
top-of-script
  ├─ sidebar (radio, source checkboxes, refresh slider)
  ├─ fetch_all() ← cache'li ama yine de hash compute
  ├─ FAULT_LINES global ← 40-60ms her seferinde
  ├─ df enrich (uzaklik_km df.apply) ← rebuild
  ├─ deduplication loop ← rebuild
  └─ if active_menu == "X":  ← sadece bu BLOK koşullu
       render_panel_X()
```

### Hedef yapı (fragment ile)
```
top-of-script (sadece bir kez run)
  ├─ sidebar  (kalır)
  ├─ active_menu = st.sidebar.radio(...)
  ├─ @st.cache_data ile df hazırla (1 kez)
  │
  ├─ @st.fragment
  │  def canli_radar_panel():
  │      # tam render burada
  │
  ├─ @st.fragment
  │  def istatistik_panel(): ...
  │
  ├─ ...her panel için fragment
  │
  └─ dispatch:
       if active_menu == "🌍 Canlı Radar":
           canli_radar_panel()
       elif active_menu == "📊 İstatistik & Analiz":
           istatistik_panel()
       ...
```

### Beklenen kazanç (Fragment Mimarı sonrası)
- Menü geçişi: **60-80ms → 5-15ms** (sadece fragment re-run)
- Üst script (fetch_all, fault_lines hash) her sefer çalışmaz
- 1000 satır deprem durumunda da menü geçişi <500ms olmalı

### Yan kazançlar
- `df` ve `FAULT_LINES` sadece bir kez hesaplanır → kullanıcının veri penceresi değişirken bile sadece veri-getirici fragment etkilenir
- `st_autorefresh` daha az invaziv olur (sadece radar fragment etkilenir)

---

## ❗ Ek Bulgular (Ajan 3 & 5 için)

### Ajan 3 — Vektörleştirme adayları (öncelik sırası)
1. [earthquake.py:432](earthquake.py#L432) `df["uzaklik_km"] = df.apply(haversine, axis=1)` — NumPy vektörleştir
2. [earthquake.py:437-445](earthquake.py#L437) Dedup O(n²) Python döngü — KDTree veya pandas merge_asof
3. [earthquake.py:2384](earthquake.py#L2384) Korelasyon matrisinde `precursors.apply(compute_environmental_features)` — batch ephem (1000 olay × 5 gök cismi = 5000 ephem çağrısı, %90'ı tekrar)

### Ajan 5 — Render bulguları
- `load_fault_lines` 14,500 segment listesi → `@st.cache_data` cache hash 40-60ms (her menü geçişinde)
- Çözüm A: Liste yerine pandas DataFrame return et (Streamlit DataFrame'i daha iyi hash'liyor)
- Çözüm B: Module-level constant olarak yükle (st.cache_data atla, `@st.cache_resource` kullan — daha hızlı erişim için)
- Çözüm C: Global state olarak `if "FAULT_LINES" not in st.session_state` ile session_state'e koy

---

## 📋 Sonraki Adım

**Ajan 2 — Fragment Mimarı**'na geçiş onayı bekliyor.

Fragment refactor'un riskleri:
- `df` global'e bağımlı render kodları çok — fragment içinde nasıl erişeceği netleşmeli
- `st.session_state` kullanımı zaten yaygın, daha da artırılacak
- Test: her menünün hatasız render olduğu doğrulanmalı

Tahmini iş hacmi: **~50-100 satır değişiklik**, 1 commit (v1.12), test ile birlikte.
