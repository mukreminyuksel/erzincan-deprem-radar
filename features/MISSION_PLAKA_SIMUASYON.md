# 🌍 Misyon Brifingi — Tektonik Plaka Hareketi Simülasyonu

**Feature ID:** F-43  
**Versiyon hedefi:** v1.17  
**Talep eden:** Kullanıcı (doğrudan talep — 2026-05-26)  
**Hazırlayan:** Ajan 9 (Tasarım Master)  
**Routing:** Ajan 8 → Ajan 4 → Ajan 5 → Ajan 9  
**Öncelik:** 🔴 Yüksek  
**Durum:** Backlog — bilimsel fizibilite soruları yanıtlanmadan Ajan 4/5/9 görevlerine başlanmayacak

---

## 🎯 Özellik Tanımı

Kullanıcı, tektonik plakaların mm/yıl ölçeğindeki gerçek GPS hız vektörlerini kullanarak **10 / 100 / 1.000 yıllık zaman adımlarında** Dünya'nın yüzey konfigürasyonunu animasyonlu olarak simüle eden bir panel görmek istemektedir. Simülasyon yalnızca görsel/eğitimsel amaçlıdır; 1 My+ jeolojik ölçeğe iddia taşımaz.

**Kullanıcının göreceği şey:**

1. Mevcut PB2002 plaka poligonlarını başlangıç konfigürasyonu olarak alan bir Mapbox haritası.
2. Bir zaman kaydırıcı (slider): `0 – 10 – 50 – 100 – 500 – 1.000 yıl` adımları (ya da "Oynat" butonu ile otomatik ilerleme).
3. Her adımda plakaların GNSS hız vektörlerine (mm/yıl) göre kaydırılmış sınır poligonları — küçük ama birikimli yer değiştirme görülür.
4. **Erzincan pin'i (39.75°N, 39.49°E)** sabit kalır; plakanın ona göre nasıl yaklaşıp/uzaklaştığı ya da çevresindeki fay geriliminin yıllara göre nasıl biriktiği gösterilir.
5. Bilgi kutusu: seçili yılda Erzincan'ın Avrasya / Anadolu plakasına göre tahmini kümülatif yer değiştirmesi (mm → m → km cinsinden).

**Etkileşim akışı:**

```
Panel açılır → Varsayılan: 0 yıl (bugün)
      ↓
Slider ilerletilir veya ▶ Oynat tıklanır
      ↓
Plaka poligonları frame-frame kayar, Erzincan pin'i sabit
      ↓
Erzincan bilgi kutusu güncellenir: "Bu sürede ~X m / X km kaydı"
      ↓
"Bilimsel Not" expander: "Bu simülasyon lineer ekstrapolasyon kullanır;
 gerçek jeolojik süreçler (fay kilitleme, viskoelastik relaksasyon) daha karmaşıktır."
```

---

## 🔬 Ajan 8'e Sorular (Bilimsel Fizibilite)

**Adres:** Bilim Profesörü (Ajan 8)  
**Yanıt bekleme:** Ajan 4 ve 5'in göreve başlaması için Q-SCI-8.1–8.6 yanıtlanmalı.

| Soru-ID | Konu | Soru |
|---|---|---|
| Q-SCI-8.1 | GPS hız veri seti | Bu simülasyon için en uygun GNSS hız alanı hangisi: **Reilinger et al. JGR 2006** (642 istasyon, Anadolu+Orta Doğu), **Aktug et al. JGR 2009** (Türkiye odaklı, denser), yoksa UNAVCO/IGS'den indirilebilen en güncel çözüm mü? Referans çerçeve: Eurasia-fixed mi sabit tutulmalı? |
| Q-SCI-8.2 | Erzincan bölgesi mm/yıl değerleri | Erzincan'ın (~39.75°N, 39.49°E) Anadolu plakası içindeki doğrusal GNSS hızı (Doğu+Kuzey bileşenleri, mm/yıl) nedir? NAFZ'ın bu kesimindeki sağ-yönlü yanal kayma oranı (18–27 mm/yıl bant?) ne kadar güvenilir? |
| Q-SCI-8.3 | 10/100/1.000 yıl jeolojik gerçekçilik sınırları | Hangi zaman ufkuna kadar *lineer ekstrapolasyon* (v × t) savunulabilir, hangi noktadan itibaren fay kilitleme döngüleri, büyük deprem atlamaları ve viskoelastik relaksasyon baskın olur? "10 yıl makul, 1.000 yıl yalnızca eğitimsel" gibi bir sınır çizilebilir mi? |
| Q-SCI-8.4 | Erzincan kümülatif yer değiştirme ve tarihsel korelasyon | 1939 Ms 7.8 depremi NAFZ'da ~350–400 km uzunluğunda yüzey kırığı ve ~4–7 m yatay kayma üretmiştir. Bu değer, 1939'dan günümüze (87 yıl × ~22 mm/yıl ≈ ~1.9 m) interseismik birikimle tutarlı mı? Simülasyonun paleoseismik geçmiş ile kalibrasyonu mümkün mü? |
| Q-SCI-8.5 | Hangi plakaların hız vektörü gösterilmeli | Tam 52 plaka için ekstrapolasyon hesaplamak performansı öldürür. Türkiye çevresinde anlamlı olan **Anadolu, Avrasya, Arabistan, Afrika (Nubya), Ege** plakaları için sınırlasak bilimsel açıdan yeterli mi? |
| Q-SCI-8.6 | Disclaimer ve hata bandı | Simülasyonun gösterdiği yer değiştirme değerlerinin ±% kaç hata bandı içinde sunulması gerekir? "Bu bir öngörü değil, mevcut hız alanının ekstrapolasyonudur" ibaresi yeterli mi, yoksa daha güçlü uyarı mı lazım? |

---

## 🗺️ Ajan 4'e Görev (Tektonik Veri)

**Adres:** Tektonik Veri Uzmanı (Ajan 4)  
**Ön koşul:** Q-SCI-8.1 ve 8.5 yanıtlandıktan sonra başla.

### Mevcut veri durumu

| Dosya | İçerik | Simülasyona hazır mı? |
|---|---|---|
| `tectonic_plates.geojson` | PB2002, 52 plaka poligonu, `plate` özelliği | ✅ Poligon geometrisi var; hız yok |
| `turkey_faults.geojson` | MTA 14,500+ fay segmenti | ⬜ Animasyonda statik tutulabilir (ekstra hareket hesabı gerekmez) |

### Görevler

1. **GNSS hız vektörü dosyası oluştur** (`plate_velocities.json`):
   - Kaynak: Reilinger 2006 veya Ajan 8'in onayladığı veri seti.
   - Format: Her plaka için `{ "plate": "AN", "ve_mm_yr": 25.3, "vn_mm_yr": 6.1 }` (ITRF Eurasia-fixed).
   - Minimum kapsam: Anadolu (AN), Avrasya (EU), Arabistan (AR), Afrika-Nubya (NU), Ege (AE) + diğer görünür plakalar.
   - Kaynakta yer almayan plakalar için `null` değer; animasyonda bu plakalar statik kalır.

2. **Simülasyon geometri fonksiyonu prototipi** (`plate_simulation.py`):
   - Girdi: `plates_geojson`, `velocities_dict`, `years` (int).
   - Her poligon köşesi için: `Δlon = ve × years / (111_320 × cos(lat))`, `Δlat = vn × years / 111_320`.
   - Çıktı: Kaydırılmış GeoJSON (mevcut ile aynı yapı, koordinatlar güncellenmiş).
   - **Dikkat:** Küresel projeksiyon hatası — 1.000 yıl ufkunda (max ~25 m kayma) hata ihmal edilebilir; belgelensin.

3. **Bağımlılık analizi**: Mevcut `F-35` (Türkiye mikroplaka GPS vektörleri) bu görevle örtüşüyor mu? Varsa `plate_velocities.geojson`'ı ortak kullanacak şekilde tasarla.

---

## 🖥️ Ajan 5'e Görev (Render & Animasyon)

**Adres:** Harita Render Uzmanı (Ajan 5)  
**Ön koşul:** Ajan 4'ün `plate_simulation.py` prototipi hazır olduktan sonra başla.

### Plotly Animasyon Mimarisi

```python
# Hedef yapı
fig = go.Figure(
    data=[initial_plate_traces, fault_traces, erzincan_pin],
    frames=[
        go.Frame(
            data=[shifted_plate_traces(year=y)],
            name=str(y),
            layout=go.Layout(title_text=f"Plaka Hareketi — {y} Yıl")
        )
        for y in [0, 10, 50, 100, 250, 500, 1000]
    ],
    layout=go.Layout(
        updatemenus=[play_pause_button],
        sliders=[year_slider]
    )
)
```

### Slider parametresi

```python
sliders = [{
    "active": 0,
    "steps": [{"args": [[str(y)], {"frame": {"duration": 600}, "mode": "immediate"}],
               "label": f"{y}y", "method": "animate"}
              for y in [0, 10, 50, 100, 250, 500, 1000]],
    "x": 0.05, "xanchor": "left", "y": 0, "yanchor": "top",
    "currentvalue": {"prefix": "Simüle edilen süre: ", "suffix": " yıl", "visible": True}
}]
```

### Performans bütçesi

| Metrik | Hedef | Yöntem |
|---|---|---|
| İlk render | < 2.5 s | Sadece 5 plaka için frame hesapla (Ajan 8 → Q-SCI-8.5) |
| Frame geçiş | < 400 ms | Pre-compute all frames, JSON ön yükleme |
| Toplam frame boyutu | < 2 MB | 5 plaka × 7 zaman adımı × ~50 KB/poligon |
| Fay segmentleri | Statik | Animasyonda hareket etmez; tek trace olarak bırakılır |

### Erzincan pin'i

- Sabit `go.Scattermapbox` marker: `lat=39.75`, `lon=39.49`, `symbol="star"`, `color="#FF4136"`.
- Hover text (frame'e bağlı güncellenir): `"Erzincan | {year} yıl sonra tahmini kayma: ~{displacement:.1f} m"`.
- Displacement = `sqrt((ve × year)² + (vn × year)²)` mm → m dönüşümü, Ajan 4'ün AN plaka vektöründen.

---

## 🎨 Ajan 9 — Yerleşim Kararı

**Gerekçeli karar:** Özellik hangi menüye/panele yerleştirilecek?

### Adaylar

| Seçenek | Artıları | Eksileri | Skor (1–5) |
|---|---|---|---|
| 🌍 **Canlı Radar** altına alt-sekme | Harita zaten açık, bağlam tutarlı | Canlı Radar aktif sismisiteye odaklı; jeolojik ölçek uyumsuz | 2 |
| 🧭 **Fay Sistemleri** altına alt-sekme | Tektonik bağlam doğru | Fay Sistemleri türkiye-lokal odaklı; global plaka animasyonu büyük | 3 |
| 🌐 **Yeni bağımsız panel: "Tektonik Simülasyon"** | Kendi mental modelini taşıyor; overload yok | Pill bar'a 9. öğe ekler (Hick's Law cezası); F-31 pill mimarisi test edilmeli | 4 |
| 🧭 **Fay Sistemleri paneli içinde üst altsekme** | Pill sayısı sabit kalır | "Fay Sistemleri" ismi global simülasyonu yeterince karşılamıyor | 3 |

### **Karar: Yeni Bağımsız Panel — "🌐 Tektonik Simülasyon"**

**Gerekçe:**

1. **Mental model bütünlüğü:** Kullanıcı 10–1.000 yıl ufkunda dünya ölçeğinde bir simülasyon istiyor. Bu, mevcut hiçbir panelin kapsamına girmiyor; kendi sekmesini hak ediyor.
2. **Fay Sistemleri kirlenmemeli:** F-37 (Coulomb) ve F-36 (Marmara) zaten Fay Sistemleri'ne ekleniyor; üçüncü büyük özellik o paneli aşırı yükler.
3. **Canlı Radar'ın canlılığı korunmalı:** Radar, gerçek zamanlı sismisiteye adanmış; jeolojik zaman ölçeği orada yabancı.
4. **Pill bar sınırı:** F-31 sprint'inde pill bar test edilecek. Eğer 9. öğe UX testini geçerse (Ajan 7 onayı ile) Tektonik Simülasyon eklenir; geçmezse Fay Sistemleri altında expander olarak gömülür.

**Koşullu karar ağacı:**

```
F-31 pill bar (9 öğe) UX testi → GEÇTİ → "🌐 Tektonik Simülasyon" bağımsız panel
                                → KALDI  → "Fay Sistemleri" altında "🌐 Plaka Animasyonu" altsekme
```

**Panel içi düzen (wireframe özeti):**

```
┌─────────────────────────────────────────────────┐
│  🌐 Tektonik Plaka Hareketi Simülasyonu         │
│  ─────────────────────────────────────────────  │
│  [▶ Oynat]  [Zaman: ████░░░ 100 yıl]           │
│                                                  │
│  ┌──────────────── Mapbox ────────────────────┐ │
│  │  Plaka poligonları (renkli, hareket eder)  │ │
│  │  ★ Erzincan (sabit pin)                    │ │
│  │  → hız vektör okları (ince, yarı şeffaf)   │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  📍 Erzincan Kümülatif Kayması: ~2.2 m          │
│  ℹ️ [Bilimsel Not — expander]                   │
└─────────────────────────────────────────────────┘
```

---

## ⚠️ Riskler & Bağımlılıklar

| Risk | Seviye | Açıklama | Azaltma |
|---|---|---|---|
| **Bilimsel abartı riski** | 🔴 Yüksek | Lineer ekstrapolasyon 1.000 yılda deprem döngülerini, fay kilitleme-kayma geçişlerini, mantle akışını göz ardı eder; kullanıcı bunu "tahmin" sanabilir | Disclaimer zorunlu; Ajan 8 metni onaylamalı |
| **Performans riski** | 🟡 Orta | 52 plaka × 7 frame = 364 poligon kaydırma işlemi; pre-compute olmadan render yavaş | Ajan 5: frame'ler `@st.cache_data` ile önceden hesaplanacak |
| **Veri lisans riski** | 🟢 Düşük | Reilinger 2006 JGR makalesi açık erişim değil; tablo verileri akademik kullanım için CC değil | AGU'nun lisans politikası kontrol edilmeli; alternatif: UNAVCO public dataset |
| **F-31 bağımlılığı** | 🟡 Orta | Pill bar UX testi geçmezse yerleşim kararı değişir | Koşullu karar ağacı tanımlandı (yukarıda) |
| **F-35 çakışması** | 🟡 Orta | F-35 (mikroplaka GPS) aynı hız verilerini kullanır; iki ayrı dosya tutarsızlığa yol açar | Ajan 4: ortak `plate_velocities.geojson` + ayrı görselleştirme |
| **Koordinat projeksiyon hatası** | 🟢 Düşük | Düzlemsel Δlon/Δlat hesabı küresel geoid'e göre milyonlarca yılda sapma gösterir; 1.000 yılda ihmal edilebilir (<0.5 m) | Ajan 4 belgelesin; 1 My+ için uyarı eklensin |

---

## ✅ Başarı Kriterleri

Kullanıcı aşağıdakileri görünce "tamamdır" diyecek:

1. **Animasyon çalışıyor:** Slider 0 → 1.000 yıla ilerlediğinde harita üzerindeki plaka poligonları gözle görülür biçimde kayıyor (1.000 yılda Anadolu için ~22 m = Türkiye ölçeğinde fark edilir).
2. **Erzincan pin'i sabit ve bilgilendirici:** Her zaman adımında "Bu sürede Erzincan ~X m / ~X km kaydı" kutusu doğru hesaplıyor.
3. **Bilimsel disclaimer görünür:** "Bu simülasyon lineer hız ekstrapolasyonudur; gerçek jeolojik süreçleri temsil etmez" metni panel üstünde ya da expander içinde sabit duruyor.
4. **Hız ve akıcılık:** Frame geçişi < 400 ms, ilk yükleme < 2.5 s (Streamlit profiling).
5. **Yıl parametresi anlamlı:** 10 / 50 / 100 / 250 / 500 / 1.000 yıl basamakları kullanıcının "insan ömrü → tarih → jeoloji" algısına uygun sıralı — slider etiketleri okunaklı.
6. **Fay sınırları statik ve üst katmanda:** MTA fay segmentleri hareket etmez, plaka poligonlarının üstünde (z-order) görünür kalır; görsel çelişki yok.

---

*Misyon brifingi hazırlayan:* **Ajan 9 — Tasarım Master**  
*Tarih:* 2026-05-26  
*Sonraki adım:* Ajan 8'in Q-SCI-8.1–8.6 yanıtlarını bekle → Ajan 4 çalışmaya başlar → Ajan 5 frame mimarisini kurar → Ajan 9 UI entegrasyonunu onaylar.
