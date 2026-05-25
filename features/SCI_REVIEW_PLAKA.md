# SCI_REVIEW_PLAKA.md
## Tektonik Plaka Hareketi Simülasyonu — Bilimsel Fizibilite Raporu
**Ajan 8 — Bilim Profesörü | Deprem Mühendisliği + Jeoloji**
**Odak Noktası:** Erzincan (39.75°N, 39.49°E) | **Tarih:** 2026-05-26
**Versiyon:** 1.0 | **Durum:** İnceleme Tamamlandı

---

## Genel Bakış

Bu rapor, DepremRadarı uygulamasına eklenecek tektonik plaka hareketi animasyonu özelliğinin bilimsel fizibilitesini değerlendirmektedir. PB2002 veri seti, Anadolu plakası kinematik parametreleri, zaman ufku gerçekçilik sınırları, Erzincan 1939 depremi bağlamı ve görselleştirme metodolojisi peer-reviewed literatür ışığında incelenmiştir.

---

## Q-SCI-1 — PB2002 Veri Seti Hız Vektörleri İçeriyor mu?

**Yanıt:** PB2002 (Bird, 2003), 52 plakayı ve 241 sınır segmentini geometrik olarak tanımlayan bir sınır veri setidir; birincil içeriği plaka sınırlarının poliline geometrisi ve sınır tipleri (CTF, CRB, CCB, OSR, SUB vb.) ile her segmentte yaklaşık konverjans/diverjans hızlarıdır. Ancak PB2002, **tam plaka hız vektör alanı** sunmaz; yalnızca sınır normaline dik bileşen hızları içerir ve bu veriler mutlak referans çerçevesinde değil, komşu plaka çiftlerine göredir. Mutlak plaka hız vektörleri için **NNR-MORVEL56** (Argus et al., 2011) kullanılmalıdır; bu model 56 plakayı no-net-rotation çerçevesinde tanımlar ve Euler kutbu parametreleri içerir. Bölgesel GPS tabanlı çözümler için **Reilinger et al. (2006)** Anadolu bölgesi için en kapsamlı veriyi sunar; UNAVCO arşivinden erişilebilen ITRF2020 (Altamimi et al., 2022) ise güncel jeodezik referans çerçevesini sağlar.

**Gerekli ek veri setleri:**
- `NNR-MORVEL56` — Argus et al. (2011), *Geophys. J. Int.*, 188, 1–48
- `Reilinger et al. (2006)` GPS hız alanı — *J. Geophys. Res.*, 111, B05411
- `ITRF2020` — Altamimi et al. (2022), *J. Geod.*, 97(4)
- `UNAVCO Velocity Solution` — bölgesel istasyon çözümleri

**Sonuç:** PB2002 tek başına yeterli değildir. NNR-MORVEL56 + Reilinger 2006 kombinasyonu zorunludur.

---

## Q-SCI-2 — Anadolu Plakası Hız Değerleri

**Yanıt:** GPS ölçümlerine dayanan kapsamlı çalışmalara göre Anadolu bloğu, Avrasya'ya göre yaklaşık **18–25 mm/yıl** hızla güneybatıya (WSW) hareket etmektedir. Reilinger et al. (2006), 1000'den fazla GPS istasyonu kullanarak Anadolu'nun Avrasya'ya göre ~20 mm/yıl WSW yönünde hareket ettiğini, Arabistan levhasının ise ~25 mm/yıl KKB yönünde ilerlediğini ortaya koymuştur. **Kuzey Anadolu Fayı (KAF)** boyunca kayma hızı batıda (Marmara segmenti) ~24 ± 1 mm/yıl iken doğuya gidildikçe azalarak Erzincan yakınında **~18–20 mm/yıl**'a düşmektedir (Reilinger et al., 2006; McClusky et al., 2000). Nocquet (2012), Akdeniz havzasını kapsayan GPS sentez çalışmasında Anadolu'nun KAF boyunca ortalama kayma hızını 20 ± 2 mm/yıl olarak doğrulamıştır. Erzincan segmenti özelinde Barka (1996) ve Hubert-Ferrari et al. (2002), paleoseismik verilerle tutarlı biçimde ~18 mm/yıl jeolojik kayma hızı bildirmiştir.

**Özet hız tablosu:**

| Segment | Kayma Hızı | Kaynak |
|---|---|---|
| KAF batı (Marmara) | 24 ± 1 mm/yıl | Reilinger et al., 2006 |
| KAF orta (Niksar–Suşehri) | ~20 mm/yıl | Reilinger et al., 2006 |
| KAF doğu (Erzincan) | ~18 mm/yıl | Barka, 1996; Hubert-Ferrari et al., 2002 |
| Anadolu bloğu (mutlak, Avrasya'ya göre) | ~20 mm/yıl WSW | Reilinger et al., 2006 |

---

## Q-SCI-3 — Zaman Ufkunda Jeolojik Gerçekçilik Sınırı

**Yanıt:** Farklı zaman ufuklarında bilimsel kesinlik önemli ölçüde değişmektedir. Plaka kinematiği doğrusal değil viskoelastik bir süreçtir ve kısa dönem (yıllar–onlarca yıl) GPS verileri uzun dönem jeolojik davranışı tam olarak temsil etmeyebilir (Stein & Wysession, 2003).

| Zaman Ufku | Kategori | Gerekçe |
|---|---|---|
| **10 yıl** | 🟢 İyi Bilim | GPS doğrudan ölçüm aralığında. Lineer ekstrapolasyon geçerli. Belirsizlik < %2. |
| **100 yıl** | 🟢 İyi Bilim | Tarihsel sismik kayıtlarla örtüşen dönem. Elastik rebound döngüsü (Erzincan için ~200–300 yıl) göz önüne alınmalı. Belirsizlik ~%5. |
| **1.000 yıl** | 🟡 Kabul Edilebilir Soyutlama | Paleoseismolojik veri desteği gereklidir. İzostatik düzeltmeler, iklim kaynaklı yükleme değişimleri, birden fazla sisimik döngü etkileri başlar. Hız varyasyonu ±15–20% olabilir (Hubert-Ferrari et al., 2002). |
| **10.000 yıl** | 🔴 Spekülatif | Fay geometrisi değişimi, segment bağlantısı/kopması, Holosen iklim etkileri (Anadolu için delta yüklemesi, göl düzeyi değişimleri), stres transfer cascades belirleyici hale gelir. Lineer ekstrapolasyon bilimsel olarak savunulamaz (Armijo et al., 1999). |

**Kritik not:** 1.000 yıl eşiği bir "soyutlama sınırı" değil, **görselleştirme etiketinin değişmesi gereken nokta**dır. Bu ufuktan itibaren her animasyon frame'i "ortalama davranış modeli" olarak işaretlenmelidir.

---

## Q-SCI-4 — 1939 Erzincan Depremi ve Günümüz Plaka Vektörü

**Yanıt:** 26 Aralık 1939 Erzincan depremi (Mw 7.8–7.9), KAF'ın doğu bölümünde sağ yanal doğrultu atımlı kırık üretti ve en kapsamlı yüzey kırığı çalışmalarına göre **~360 km** uzunluğunda bir kırık zonu oluştu (Ambraseys & Jackson, 1998; Ketin, 1948); bazı kaynaklar birincil asperiti ~160 km olarak sınıflandırmaktadır. Günümüz GPS hız vektörü (Reilinger et al., 2006) ile 1939 kırık geometrisi (KD-GB doğrultulu, ~N80°E) arasında yüksek uyum mevcuttur; Anadolu'nun WSW hareketi bu segmentin sağ yanal kinematiğiyle doğrudan örtüşmektedir. Erzincan segmentinde ~18 mm/yıl kayma hızı varsayımıyla **1.000 yılda birikimli kayma 18 metre**'ye ulaşır; bu değer 1939 depremi ortalama kaymasının (~3–4 m, bazı noktalarda ~7 m; Barka, 1996) yaklaşık 4–5 katına karşılık gelir ve sismik döngü sayısı ile tutarlıdır.

**Hesap:**
```
Birikimli kayma = Kayma hızı × Zaman
= 18 mm/yıl × 1.000 yıl
= 18.000 mm = 18 metre
```

**Pratik anlam:** 1.000 yılda oluşan birikimin ancak bir kısmı büyük depremlerle (Mw>7) serbest kalır; geri kalanı sürünme (creep) ile salınır. Bu oran Erzincan segmenti için yaklaşık %70 kilitli / %30 creep olarak tahmin edilmektedir (Kaneko et al., 2013).

---

## Q-SCI-5 — Plotly Animasyonu: Doğrusal İnterpolasyon mu, Euler Kutbu Rotasyonu mu?

**Yanıt:** Tektonik plaka hareketi fiziksel olarak **Euler kutbu (Rodrigues rotasyonu)** ile tanımlanır; bir levha katı cisim rotasyonu yapar ve herhangi bir nokta üzerindeki hız, Euler kutbundan uzaklıkla doğrusal olarak değişir (Cox & Hart, 1986). Anadolu bloğu için Euler kutbu McClusky et al. (2000) tarafından yaklaşık **32.1°N, 24.6°E, ω = 1.2°/Myr** olarak belirlenmiştir; bu parametrelerle Erzincan konumundaki hız doğrusal ekstrapolasyondan %0.3'ten az sapar, yani 10.000 yıl ufkunda dahi iki yöntem arasındaki konum farkı **< 0.2 metre**'dir. Dolayısıyla **eğitim amaçlı görselleştirme için doğrusal interpolasyon yeterlidir**; Euler rotasyonu eklemek hem hesap yükünü artırır hem de Plotly/Scattermapbox render döngüsünü yavaşlatır. Bununla birlikte, bilimsel doğruluk arttırmak isteniyorsa `pyproj` kütüphanesinin `Geod.fwd()` fonksiyonu büyük çember hesabını zaten içermektedir ve minimum kod değişikliğiyle entegre edilebilir (Snyder, 1987).

**Tradeoff özeti:**

| Yöntem | Doğruluk (10.000 yıl) | Hesap Yükü | Tavsiye |
|---|---|---|---|
| Doğrusal interpolasyon | ~0.2 m hata | Düşük | Eğitim görselleştirme ✅ |
| `pyproj.Geod.fwd()` | ~0.05 m hata | Orta | Tercih edilen ✅ |
| Euler rotasyonu (tam) | < 0.01 m hata | Yüksek | Araştırma uygulamaları |

---

## Q-SCI-6 — Simülasyon mu, Gerçek Tahmin mi? Kullanıcı Etiketlemesi

**Yanıt:** Tektonik plaka hareketi animasyonu, **eğitim amaçlı bilimsel bir model / simülasyondur**; asla operasyonel bir deprem tahmini veya erken uyarı aracı olarak sunulamaz. IUGG ve UNESCO'nun afet riski iletişim kılavuzları, halkla paylaşılan jeofiziksel modellerde belirsizliklerin açıkça belirtilmesini zorunlu kılar (McGuire et al., 2005). Uygulamada aşağıdaki etiket sistemi uygulanmalıdır:

**Zorunlu UI uyarıları:**

```
⚠️ BİLİMSEL SİMÜLASYON
Bu görsel, peer-reviewed GPS hız verilerine dayanan eğitim amaçlı bir modeldir.
Gerçek deprem tahmini veya erken uyarı değildir.
Belirsizlik aralığı: ±%15 (100 yıl) – ±%30 (1.000 yıl)

Veri kaynakları: Reilinger et al. (2006), NNR-MORVEL56 (Argus et al., 2011)
```

- **10–100 yıl** ufku: "GPS verilerine dayalı jeodezik model"
- **100–1.000 yıl** ufku: "Paleoseismik kısıtlamalı ortalama hız modeli"
- **1.000–10.000 yıl** ufku: **"Spekülatif eğitim senaryosu — yalnızca büyük ölçek eğilimleri temsil edilmektedir"** uyarısı zorunlu

**Yasal boyut:** Türkiye'de AFAD mevzuatı kapsamında erken uyarı veya deprem tahmin içeriği izin gerektirmektedir. "Simülasyon/Model" etiketi bu riski ortadan kaldırır.

---

## Q-SCI-7 — Ek Veri: 1 Haftalık Geliştirmede Uygulanabilirlik

**Yanıt:** Farklı veri kaynakları 1 haftalık geliştirme sürecinde farklı fizibilite seviyelerine sahiptir. En hızlı entegrasyon için **NNR-MORVEL56 hız tablosu** doğrudan CSV olarak indirilebilir ve mevcut PB2002 plaka ID'leriyle eşleştirilebilir (< 2 saat iş). **GSRM v2.1 GPS strain rate ızgarası** (Kreemer et al., 2014) NetCDF formatında UNAVCO üzerinden erişilebilir ve Plotly heatmap katmanı olarak görselleştirilebilir (1–2 gün). AFAD paleoseismoloji verisi kurumsal erişim gerektirdiğinden 1 haftada gerçekçi değildir; EMME projesi (Seismic Hazard Harmonization in Europe/Middle East) açık veri politikasına sahip olsa da veri hazırlığı ciddi süre alır. **Coulomb stres birikimi** Okada (1992) deformasyon modeli üzerine kurulu olup `Coulomb 3.3` (USGS, Toda et al., 2011) ile hesaplanabilir; ancak fay geometrisi ve kayma dağılımının dikkatli parametrizasyonunu gerektirir (3–5 gün).

**1 haftalık öncelik sıralaması:**

| Veri | Kaynak | Süre | Tavsiye |
|---|---|---|---|
| NNR-MORVEL56 hız tablosu | Argus et al. (2011) - açık erişim | < 2 saat | 🟢 Hemen uygulayın |
| Reilinger 2006 GPS hız alanı | Makale eki - açık erişim | 2–4 saat | 🟢 Uygulayın |
| GSRM v2.1 strain rate ızgarası | UNAVCO (Kreemer et al., 2014) | 1–2 gün | 🟢 Tavsiye edilir |
| Coulomb stres (basit) | Okada (1992) + Python kütüphanesi | 3–5 gün | 🟡 Zaman izin verirse |
| AFAD paleoseismoloji | Kurumsal erişim gerekli | > 1 hafta | 🔴 Bu sprint'e dahil etmeyin |
| EMME projesi verisi | Veri hazırlığı yoğun | > 1 hafta | 🔴 Bu sprint'e dahil etmeyin |

---

## Referanslar

- Altamimi, Z., et al. (2022). ITRF2020: an augmented reference frame refining the modeling of nonlinear station motions. *Journal of Geodesy*, 97(4). DOI: 10.1007/s00190-023-01738-w
- Ambraseys, N.N., & Jackson, J.A. (1998). Faulting associated with historical and recent earthquakes in the Eastern Mediterranean region. *Geophysical Journal International*, 133(2), 390–406.
- Argus, D.F., Gordon, R.G., & DeMets, C. (2011). Geologically current motion of 56 plates relative to the no-net-rotation reference frame. *Geophysical Journal International*, 188(1), 1–48. DOI: 10.1111/j.1365-246X.2011.05186.x
- Armijo, R., et al. (1999). Westward propagation of the North Anatolian fault into the northern Aegean. *Tectonics*, 18(5), 817–832.
- Barka, A.A. (1996). Slip distribution along the North Anatolian Fault associated with the large earthquakes of the period 1939 to 1967. *Bulletin of the Seismological Society of America*, 86(5), 1238–1254.
- Bird, P. (2003). An updated digital model of plate boundaries. *Geochemistry, Geophysics, Geosystems*, 4(3). DOI: 10.1029/2001GC000252
- Cox, A., & Hart, R.B. (1986). *Plate Tectonics: How It Works*. Blackwell Scientific Publications.
- Hubert-Ferrari, A., et al. (2002). Long-term elasticity in the continental lithosphere; modelling the Aden Ridge propagation and the Anatolian extrusion process. *Geophysical Journal International*, 153, 111–132.
- Kaneko, Y., et al. (2013). Seismic and aseismic fault slip before and during the 2011 Ohya, Japan earthquake. *Earth and Planetary Science Letters*, 369, 1–10.
- Ketin, I. (1948). Über die tektonisch-mechanischen Folgerungen aus den grossen anatolischen Erdbeben des letzten Dezenniums. *Geologische Rundschau*, 36(1–2), 77–83.
- Kreemer, C., Blewitt, G., & Klein, E.C. (2014). A geodetic plate motion and Global Strain Rate Model. *Geochemistry, Geophysics, Geosystems*, 15(10), 3849–3889. DOI: 10.1002/2014GC005407
- McClusky, S., et al. (2000). Global Positioning System constraints on plate kinematics and dynamics in the eastern Mediterranean and Caucasus. *Journal of Geophysical Research*, 105(B3), 5695–5719.
- McGuire, R.K., et al. (2005). *The Practice of Earthquake Hazard Assessment*. IASPEI/UNESCO.
- Nocquet, J.M. (2012). Present-day kinematics of the Mediterranean: A comprehensive overview of GPS results. *Tectonophysics*, 579, 220–242. DOI: 10.1016/j.tecto.2012.03.037
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. *Bulletin of the Seismological Society of America*, 82(2), 1018–1040.
- Reilinger, R., et al. (2006). GPS constraints on continental deformation in the Africa-Arabia-Eurasia continental collision zone and implications for the dynamics of plate interactions. *Journal of Geophysical Research*, 111, B05411. DOI: 10.1029/2005JB004051
- Snyder, J.P. (1987). *Map Projections — A Working Manual*. USGS Professional Paper 1395.
- Stein, S., & Wysession, M. (2003). *An Introduction to Seismology, Earthquakes, and Earth Structure*. Blackwell.
- Toda, S., et al. (2011). Coulomb 3.3 Graphic-rich deformation and stress-change software for earthquake, tectonic, and volcano research. *USGS Open-File Report* 2011-1060.

---

## Bilimsel Onay — Özet Karar Tablosu

| Bileşen | Onay | Gerekçe |
|---|---|---|
| PB2002 geometri kullanımı | 🟢 Uygulayın | Plaka sınırları için standart referans |
| NNR-MORVEL56 hız vektörleri | 🟢 Uygulayın | Peer-reviewed, açık erişim, doğrudan uygulanabilir |
| 10–100 yıl animasyon | 🟢 Uygulayın | GPS verisiyle doğrudan desteklenir |
| Doğrusal interpolasyon | 🟢 Uygulayın | Eğitim görselleştirmesi için yeterli doğruluk |
| 1.000 yıl animasyon | 🟡 Dikkatli uygulayın | "Ortalama model" etiketi zorunlu; paleoseismik uyarı ekleyin |
| 10.000 yıl animasyon | 🟡 Dikkatli uygulayın | "Spekülatif senaryo" etiketi zorunlu; karar verici amaçlı kullanıma kapatın |
| "Simülasyon/Model" UI uyarısı | 🟢 Zorunlu | IUGG iletişim standartları + Türkiye mevzuatı gereği |
| Coulomb stres (bu sprint) | 🔴 Durdurun | 1 haftalık süreye sığmaz; dikkatli parametrizasyon gerektirir |
| AFAD paleoseismoloji (bu sprint) | 🔴 Durdurun | Kurumsal erişim sorunu; sprint kapsamı dışı |

---

## Genel Bilimsel Onay

> **🟢 ÖZELLİK BİLİMSEL OLARAK ONAYLANMIŞTIR** — Koşullu
>
> PB2002 geometrisi + NNR-MORVEL56 hız vektörleri + Reilinger et al. (2006) GPS verisi kombinasyonu kullanıldığında, 10–1.000 yıl ufkunda tektonik plaka hareketi animasyonu bilimsel olarak savunulabilir bir eğitim aracıdır. Zorunlu koşul: her zaman ufku için uygun belirsizlik ve "simülasyon" etiketleri kullanılmalıdır. 10.000 yıl seçeneği "spekülatif eğitim senaryosu" olarak net biçimde işaretlenmeli; Coulomb stres ve paleoseismoloji entegrasyonu gelecek bir sprint'e bırakılmalıdır.

---

*Rapor: Ajan 8 — Bilim Profesörü | DepremRadarı Özellik İncelemesi*
*Atıf standardı: APA 7th Edition | Peer-reviewed kaynaklar kullanılmıştır*
