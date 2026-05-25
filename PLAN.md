# DepremRadarı — Yarıda Kalan İş Planı (v1.9 sonrası)

**Tarih:** 25 Mayıs 2026
**Bağlam:** Gemini Antigravity'de v1.8 sonrası performans turu tamamlandı (sidebar lazy loading + bbox optimizasyonları + B-grid vektörleştirme — tümü `git diff` üzerinde duruyor, henüz commit yok). Sonra iki yeni panel için alt-ajan kurulumu yapıldı ama kota dolduğu için kodlama başlamadı. Bu plan o iki paneli devralıyor.

---

## 🎯 Hedef

[earthquake.py](earthquake.py) içinde **iki yeni özellik** geliştirmek:

1. 🔭 **Gökbilimci paneli** — Astronomik korelasyon ekranı
2. 🚨 **Erken Uyarı Simülatörü** — P/S dalga geri sayım ekranı

---

## 📋 Adım 0 — Mevcut Değişiklikleri Commit (v1.9)

Çalışmaya temiz tabandan başlamak için önce mevcut staged + unstaged performans değişikliklerini tek bir commit'e topla:

```
v1.9: Sidebar lazy loading + bbox optimizasyonları + B-grid NumPy vektörleştirme

- Üst tab'lar sidebar radio menüsüne taşındı (sadece aktif menü render edilir)
- η/RTL/AMR/b-grid sub-tab'ları selectbox lazy loading
- Fay segmentlerine bbox precompute (min/max lat/lon)
- nearest_fault_vertex_distance() bbox ön elemesi (~5-10× hızlanma)
- calc_b_grid_cache() haversine NumPy vektörleştirme (~100× hızlanma)
```

Profilleme dosyaları (`earthquake_profiled.py`, `profile_st.py`) staged ama bunlar geçici — commit'e dahil etmeyebiliriz, scratch'e taşıyabiliriz. **Karar gerekli.**

---

## 🔭 Adım 1 — Gökbilimci Paneli

### Mevcut Altyapı (kullanılacak)
- [earthquake.py:9](earthquake.py#L9) `import ephem`
- [earthquake.py:2172](earthquake.py#L2172) `compute_environmental_features()` — her deprem için hesaplıyor:
  - `ay_cekim` (Ay çekim proxy, 1/d²)
  - `gunes_uzaklik` (AU)
  - `gezegen_cekim` (Jüpiter + Venüs çekim proxy)
  - `mevsim` (DoY sin dalgası)
  - `sicaklik` (iklim modeli)
  - `haftalik_aktivite`
- Şu an sadece **korelasyon matrisinde** ([earthquake.py:2384](earthquake.py#L2384)) kullanılıyor, kendi paneli yok.

### Yapılacak
Sidebar menüye yeni öğe: **"🔭 Astronomik Analiz"**

Panel içeriği:
1. **Mevcut anki gök durumu kartları** — şu an Erzincan üzerinde:
   - Ay fazı (yeni/dolunay), Ay-Dünya mesafesi (km), Ay çekim göstergesi
   - Güneş-Dünya mesafesi (AU), Güneş yükseklik açısı
   - Jüpiter, Venüs konum/uzaklık
2. **Ay fazı vs deprem büyüklüğü grafiği** — Seçili dönemdeki tüm depremler scatter plot:
   - X: Ay fazı (0=yeni ay, 0.5=dolunay)
   - Y: Magnitude
   - Renk: derinlik
3. **Ay çekim zaman serisi** — `ay_cekim` zaman içinde + üzerine deprem olayları nokta olarak
4. **Periyodogram / FFT** — Deprem sıklığı vs Ay periyodu (29.5 gün), gerçek korelasyon var mı?
5. **Gezegenlerin hizalanma anları** — Önemli konjonksiyon tarihleri ve o günlerin deprem aktivitesi

### Bilimsel Uyarı Kutusu
"Bu panel keşfedicidir; Ay/Güneş çekiminin deprem tetikleyici olduğu **kanıtlanmamıştır**. Literatürde zayıf korelasyon raporları vardır ancak nedensellik gösterilmemiştir." — bilimsel namus için zorunlu.

### Performans Notu
`compute_environmental_features` her deprem için ayrı ayrı `ephem` çağrısı yapıyor — pahalı. Panel için `@st.cache_data` ile sarmalanmış toplu hesaplama yapılmalı. Aktif Çalıştır butonu ile (mevcut İstatistik panel deseni gibi) tetiklenmeli, otomatik çalışmamalı.

---

## 🚨 Adım 2 — Erken Uyarı Simülatörü

### Mevcut Altyapı (kullanılacak)
- [earthquake.py:1907](earthquake.py#L1907) Bilgi Havuzu'nda P/S dalga eğitsel animasyonu var (statik).
- [earthquake.py:2166-2167](earthquake.py#L2166-L2167) Erzincan senaryosunda P/S/Rayleigh halkaları ve hız değerleri zaten tanımlı:
  - P-dalga: ~6 km/s
  - S-dalga: ~3.5 km/s
  - Rayleigh: ~2.5 km/s

### Yapılacak
Sidebar menüye yeni öğe: **"🚨 Erken Uyarı"** (veya Bilgi Havuzu altında alt-bölüm)

Panel içeriği:
1. **Olay seçici** — Son 24 saatteki en büyük 10 deprem dropdown'ı (M ≥ 3.5)
2. **Kullanıcı konumu girişi** — Önceden tanımlı şehirler dropdown (Erzincan, İstanbul, Ankara, İzmir, Erzurum, Trabzon, Diyarbakır, ...) veya manuel lat/lon
3. **Geri sayım hesaplaması**:
   - Merkez üs → kullanıcı konumu mesafesi (haversine, mevcut [earthquake_core.py](earthquake_core.py))
   - P-dalga varış süresi = mesafe / vp
   - S-dalga varış süresi = mesafe / vs
   - **"S − P" uyarı penceresi** = kullanıcının kaçmak/sığınmak için sahip olduğu saniye sayısı
4. **Görsel zaman ekseni** — 0'dan başlayıp P/S varışlarına kadar uzanan timeline (Plotly) + animasyonlu ilerleyen ibre (st.empty + time.sleep loop, fragment ile)
5. **Beklenen şiddet (MMI tahmini)** — Mesafeye göre azalan basit GMPE: `I = 1.5*M − 1.5*log10(R) − 3.5` benzeri formül + renk kodlu uyarı (yeşil/sarı/turuncu/kırmızı)
6. **Eğitimsel kart** — "Bu süre içinde ne yapmalı?" (lift kapatma, çök-kapan-tutun pozisyonu, kapı/pencere)

### Bilimsel Uyarı Kutusu
"Bu **simülasyondur**, gerçek bir EEW (Earthquake Early Warning) sistemi değildir. Türkiye'de operasyonel EEW olarak AFAD-EQE çalışmaktadır. Bu panel sadece *kavram demonstrasyonudur*."

### Önemli Edge Case
- Yüzeysel deprem (depth < 10 km): P/S hızları sığ kabuk için daha düşük olabilir → not düş
- Çok uzak deprem (>500 km): yüzey dalgaları baskın olur, S-P formülü zayıflar → uyarı kutusu
- Sıfıra çok yakın (epicenter < 5 km): formül anlamsız → "Zaten merkez üstesin" mesajı

---

## 📝 Adım 3 — CHANGELOG.md Güncellemesi

Mevcut [CHANGELOG.md](CHANGELOG.md) v1.6'da kalmış. Şunları ekle:
- v1.7 (mevcut commit)
- v1.8 (mevcut commit)
- v1.9 (Adım 0 commit)
- v1.10 (Gökbilimci paneli)
- v1.11 (Erken Uyarı Simülatörü)

---

## ✅ Adım 4 — Test

Streamlit uygulamasını yerelde başlat, iki yeni menüye gir, render edildiğini ve hata vermediğini doğrula:

```bash
streamlit run earthquake.py --server.port 8560
```

Manuel kontrol noktaları:
- [ ] Sidebar'da "🔭 Astronomik Analiz" görünüyor mu?
- [ ] Sidebar'da "🚨 Erken Uyarı" görünüyor mu?
- [ ] Eski menüler hâlâ çalışıyor mu? (regression yok)
- [ ] Astronomik panel Çalıştır butonu cevap veriyor mu?
- [ ] Erken Uyarı dropdown'da gerçek deprem geliyor mu?
- [ ] Geri sayım timeline'ı render oluyor mu?

---

## ❓ Onay Gerekli Sorular

1. **Adım 0 (v1.9 commit):** Profilleme dosyaları (`earthquake_profiled.py`, `profile_st.py`) commit'e dahil mi, scratch'e mi taşınsın?
2. **Adım 1 (Gökbilimci):** 5 alt-bileşenden hepsi mi yoksa MVP olarak sadece 1-2-3 mü? (4 = FFT ve 5 = konjonksiyon hesaplama daha karmaşık)
3. **Adım 2 (Erken Uyarı):** Sidebar'da ayrı menü mü, yoksa Bilgi Havuzu altında bir sub-tab mı?
4. **Sıralama:** Önce Gökbilimci mi, önce Erken Uyarı mı, paralel mi (iki panel birbirinden bağımsız)?
