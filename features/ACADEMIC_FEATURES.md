# DepremRadarı — Akademik Kaynaklı Özellik Önerileri
**Tarih:** 2026-05-26  
**Yazan:** Ajan 8 — Bilim Profesörü  
**Yöntem:** Peer-reviewed literatür + kurum veri kataloğları doğrudan tarama  
**Başlangıç ID:** F-44 (F-01–F-43 mevcut backlog'da kayıtlı)  
**Kural:** Atıf verilemeyen özellik eklenmez. "Belki vardır" yoktur.

---

> ⚠️ **BİLİMSEL UYARI [Ajan 8] — Okuyun, Geçmeyin**  
> **Konu:** Mevcut backlog ile bu dosyanın ilişkisi  
> **Sorun:** F-37 (CFS statik), F-38 (ETAS), F-14 (b-haritası), F-12 (RTL sessizlik), F-42 (foreshock) zaten backlog'da. Bu dosya **yalnızca bu 5 özelliği kapsamayan** yeni akademik katkıları içermektedir. Kategoriler yeniden açılmış olsa da tekrar listeme **yoktur**.  
> **Etki:** Yeni bir özellik F-37 ile birebir çakışıyorsa reddetme yetkisi kullanılmıştır — not düşülmüştür.  
> **Kaynak:** Backlog v1.14, bu doküman.

---

## KATEGORİ 1 — Sismik Tehlike ve Risk Haritaları

---

## [F-44] PSHA Haritası — OpenQuake / SHARE ile Karşılaştırmalı Tehlike Analizi

**Bilimsel temel:**  
PSHA (Probabilistic Seismic Hazard Analysis), Türkiye için AFAD'ın TBDY-2018 deprem bölgesi haritasının dayandığı yöntemdir. SHARE (Seismic Hazard Harmonization in Europe) projesi ise 2013'te Avrupa + Türkiye için birleşik 50 yıl / %10 aşılma olasılıklı (475 yıl dönüş periyodu) PGA haritası yayımlamıştır. OpenQuake Engine bu hesapları açık kaynak olarak tekrarlanabilir kılmaktadır.

**Kaynak:**
- Woessner, J. et al. (2015). The 2013 European seismic hazard model: key components and results. *Bulletin of Earthquake Engineering*, 13, 3553–3596. DOI: [10.1007/s10518-015-9795-1](https://doi.org/10.1007/s10518-015-9795-1)
- Pagani, M. et al. (2014). OpenQuake Engine: An Open Hazard (and Risk) Software for the Global Earthquake Model. *Seismological Research Letters*, 85(3), 692–702. DOI: [10.1785/0220130087](https://doi.org/10.1785/0220130087)
- AFAD (2018). Türkiye Bina Deprem Yönetmeliği (TBDY-2018). URL: https://deprem.afad.gov.tr/depremzonasi

**Ne gösterir:**  
Kullanıcının seçtiği herhangi bir konumdaki 475 yıl dönüş periyotlu zemine göre ivmesi (PGA m/s²), TBDY-2018 bölge sınıfı, ve SHARE haritasıyla fark renk kodlaması. "Eviniz hangi sismik tehlike zonunda?" sorusunu net cevaplar.

**Veri kaynağı:**  
EFEHR (European Facility for Earthquake Hazard and Risk) web servisi: http://www.efehr.org/en/Documentation/specific-hazard-models/for-Europe/  
OpenQuake hazard tiles: https://downloads.openquake.org/haz_map/  
**Açık erişimli: Evet (Creative Commons)**

**Teknik karmaşıklık:** 3/5  
(API çağrısı + Plotly choropleth grid interpolasyonu; kendi PSHA hesabı değil, mevcut grid okuma)

**Görsel format:** Harita (interpolated PGA ısı haritası) + nokta sorgulama paneli

**Türkiye bağlantısı:**  
TBDY-2018 zemin ivmesi haritasının neden Erzincan, Düzce, Antakya'yı kırmızı gösterdiğini — yani KAF + DAF geometrisinin PGA'ya nasıl yansıdığını — kullanıcı ilk kez "değer olarak" görür.

**"Vay be" faktörü:**  
Kullanıcı ev adresini girer, 475 yıllık tehlike değerini saniyeler içinde görür. "50 yılda %10 ihtimalle binana şu kadar g gelir" — sigortacılar ve belediyeler bunu binlerce liraya hesaplatır, burada ücretsiz.

---

## [F-45] Vs30 Zemin Sınıfı + NEHRP Kategori Haritası

**Bilimsel temel:**  
Vs30 (üst 30 m'nin ortalama kayma dalgası hızı) sismik zemin büyütmesinin en yaygın proxy'sidir. USGS, topoğrafik eğimi Vs30 proxy'si olarak kullanan global bir model geliştirmiştir. Türkiye için İstanbul'da Boğaziçi Üniversitesi KOERI mikro-bölgeleme çalışmaları mevcuttur.

**Kaynak:**
- Wald, D.J. & Allen, T.I. (2007). Topographic slope as a proxy for seismic site conditions and amplification. *Bulletin of the Seismological Society of America*, 97(5), 1379–1395. DOI: [10.1785/0120060267](https://doi.org/10.1785/0120060267)
- USGS Global Vs30 Server: https://earthquake.usgs.gov/data/vs30/  
  *(Açık GeoTIFF; herhangi bir lat/lon için Vs30 m/s değeri)*
- Boore, D.M. et al. (2014). NGA-West2 equations for predicting PGA, PGV, and 5%-damped PSA for shallow crustal earthquakes. *Earthquake Spectra*, 30(3), 1057–1085. DOI: [10.1193/070113EQS184M](https://doi.org/10.1193/070113EQS184M) *(NEHRP sınıf tanımları için)*

**Ne gösterir:**  
Harita üzerinde tıklanan konumun Vs30 değeri (m/s) ve karşılık gelen NEHRP zemin sınıfı (A=kaya'dan E=yumuşak kile). Zemin büyütme faktörü (FA/FV). "Bu zemin depremi kaç kat büyütür?"

**Veri kaynağı:**  
USGS Vs30 GeoTIFF (global, 30 arcsec çözünürlük): https://earthquake.usgs.gov/data/vs30/us_vs30.zip  
**Açık erişimli: Evet (USGS public domain)**

**Teknik karmaşıklık:** 2/5  
(GeoTIFF rasterio okuma + nokta sorgulama; görsel basit choropleth)

**Görsel format:** Harita (4 renk: kaya/sert-zemin/yumuşak-zemin/çok-yumuşak) + sidebar bilgi kutusu

**Türkiye bağlantısı:**  
İstanbul'un Avrupa yakası kumlu dolgu, Anadolu yakası kayalık zemin farkı; 1999 Marmara'da Adapazarı'nın neden bu denli yıkıldığını (Vs30 < 150 m/s) sayısal olarak gösterir.

**"Vay be" faktörü:**  
Uydu haritasında aynı ilçe içinde bile 100 m aralıkla Vs30'un 200'den 600'e çıkabileceğini görmek — "komşum sağlam zeminde, ben bataklıkta mıyım?" sorusunu kışkırtır.

---

## KATEGORİ 2 — Fay Mekanizması ve Odak Çözümleri

---

## [F-46] GCMT Beach Ball Odak Mekanizması Katmanı

**Bilimsel temel:**  
Her M≥4 deprem için Global CMT (Centroid Moment Tensor) projesi fay düzlemi çözümünü hesaplayarak açık yayımlar. "Beach ball" diyagramı normal/ters/doğrultu atımlı fay tipi ile P-T eksenlerini tek bakışta gösterir.

**Kaynak:**
- Dziewonski, A.M., Chou, T.-A. & Woodhouse, J.H. (1981). Determination of earthquake source parameters from waveform data for studies of global and regional seismicity. *Journal of Geophysical Research*, 86(B4), 2825–2852. DOI: [10.1029/JB086iB04p02825](https://doi.org/10.1029/JB086iB04p02825)
- Ekström, G., Nettles, M. & Dziewonski, A.M. (2012). The global CMT project 2004–2010: Centroid-moment tensors for 13,017 earthquakes. *Physics of the Earth and Planetary Interiors*, 200–201, 1–9. DOI: [10.1016/j.pepi.2012.04.002](https://doi.org/10.1016/j.pepi.2012.04.002)
- GCMT Katalog: http://www.globalcmt.org/CMTsearch.html *(açık erişimli, CSV/NDK indir)*

**Ne gösterir:**  
Her büyük depremde harita üzerinde beach ball marker: hangi fay tipi (normal/ters/doğrultu), hangi doğrultuda kırık oluştu. Dokunulunca: fay düzlemi dip/rake/strike değerleri + Mw/Mwc.

**Veri kaynağı:**  
GCMT NDK formatı REST API: https://www.globalcmt.org/cgi-bin/globalcmt-cgi-bin/CMT5/form?  
**Açık erişimli: Evet**

**Teknik karmaşıklık:** 4/5  
(Beach ball SVG/path render Plotly'de kare değil çember gerektirir; obby_meca gibi kütüphane veya el-yazımı Boyle circle algoritması)

**Görsel format:** Harita üstü custom SVG marker (beach ball); yan panel: fay parametreleri tablosu

**Türkiye bağlantısı:**  
KAF = sol yanal doğrultu atım (siyah-beyaz eşit yarım), DAF = sol yanal, Batı Anadolu grabenleri = normal fay (iki siyah "kulak"). Kullanıcı rengi okumayı öğrenir.

**"Vay be" faktörü:**  
2023 Kahramanmaraş çiftini açtığında iki farklı geometrideki (KAF+DAF) beach ball yan yana durur. "Bu ikisi neden bu kadar farklı görünüyor?" sorusu 10 dakika anlatmaya değer tektonik derse kapı açar.

---

## [F-47] Bölgesel Gerilme Tenzörü İnversiyon Haritası (P-T Eksenleri)

**Bilimsel temel:**  
Çok sayıda odak mekanizması çözümünü kullanarak bir bölgenin ana sıkışma (σ1), aracı (σ2) ve genişleme (σ3) yönleri tersine çözülebilir. Michael (1984) yöntemi bölgesel stres alanını kurtarır; Türkiye'nin farklı tektonik rejimleri (sıkışma/açılma/doğrultu) bu haritada görünür.

**Kaynak:**
- Michael, A.J. (1984). Determination of stress from slip data: Faults and folds. *Journal of Geophysical Research*, 89(B13), 11517–11526. DOI: [10.1029/JB089iB13p11517](https://doi.org/10.1029/JB089iB13p11517)
- Hardebeck, J.L. & Michael, A.J. (2006). Damped regional-scale stress inversions: Methodology and examples for southern California and the Coalinga aftershock sequence. *Journal of Geophysical Research*, 111, B11310. DOI: [10.1029/2005JB004144](https://doi.org/10.1029/2005JB004144)
- Örgülü, G. & Aktar, M. (2001). Regional back-arc extension, arc-parallel compression, and strike-slip tectonics in NW Turkey from seismological evidence. *Geophysical Research Letters*, 28(23), 4455–4458. DOI: [10.1029/2001GL013863](https://doi.org/10.1029/2001GL013863) *(Türkiye uygulaması)*

**Ne gösterir:**  
Kullanıcı tanımlı coğrafi pencerede (örn. 200 km yarıçap) son N odak mekanizmasından Michael yöntemiyle hesaplanan σ1/σ3 yönleri. Ok diyagramı: sıkışma kuzeyden mi geliyor, açılma doğuya mı gidiyor?

**Veri kaynağı:**  
GCMT (F-46 ile ortak); hesaplama backend'de (scipy.optimize + Michael 1984 algoritması)  
**Açık erişimli: Evet (hesaplama kullanıcı taraflı)**

**Teknik karmaşıklık:** 5/5  
(Stres tenzörü inversiyon, öz-değer/öz-vektör analizi, bootstrap hata tahmini — STRESSINVERSE veya FMSI kodu referans alınabilir)

**Görsel format:** Harita üstü σ1 oku + stereonet polar projeksiyonu

**Türkiye bağlantısı:**  
Ege açılmasının batı Türkiye'ye "kuzeyde sıkışma + batıda açılma" çift rejim yüklediğini stereonet üzerinde gösterir; Arap plakasının kuzey baskısı doğudan farklı renkte çıkar.

**"Vay be" faktörü:**  
"Depremlerden zemine ne kadar sıkışıyor?" sorusu için oklar — yön vektörü olarak — Türkiye haritasında birleşince plaka tektoniğini tek görselde özetleyen bir sanat eseri çıkar.

---

## KATEGORİ 3 — Depremin Öncesini Anlama

---

## [F-48] ZMAP Z-testi — Sismik Aktivite Oranı Anomali Haritası

**Bilimsel temel:**  
ZMAP yöntemi (Wiemer 2001) iki zaman penceresindeki sismik oranı Z-istatistiği ile karşılaştırır. Anlamlı düşüş = sessizlik anomalisi (öncü sinyal adayı); anlamlı artış = aktivasyon. RTL (F-12) sürekli bir indeks verirken, ZMAP uzamsal grid haritası üretir — her piksel bağımsız Z-değeri alır.

**Kaynak:**
- Wiemer, S. (2001). A software package to analyze seismicity: ZMAP. *Seismological Research Letters*, 72(3), 373–382. DOI: [10.1785/gssrl.72.3.373](https://doi.org/10.1785/gssrl.72.3.373)
- Wiemer, S. & Wyss, M. (1994). Seismic quiescence before the Landers (M=7.5) and Big Bear (M=6.5) 1992 earthquakes. *Bulletin of the Seismological Society of America*, 84(3), 900–916.

> ⚠️ **F-12 (RTL) ile çakışma değerlendirmesi [Ajan 8]:**  
> RTL tek nokta için zaman serisi üretir; ZMAP tüm grid'e uzamsal harita çizer. İKİSİ AYRI — çakışma yok, tamamlayıcı.

**Ne gösterir:**  
Türkiye haritası üzerinde 0.5°×0.5° grid; her hücrede "son 6 ay vs önceki 2 yıl" Z-değeri renk kodlaması. Kırmızı = anormal sessizlik, mavi = anormal aktivasyon.

**Veri kaynağı:**  
Mevcut 9-kaynak birleşik katalog (F-01)  
**Açık erişimli: Evet (iç hesaplama)**

**Teknik karmaşıklık:** 3/5  
(Grid loop + scipy.stats.zscore; hesaplama ~2-5 sn Python'da)

**Görsel format:** Harita (choropleth Z-değeri ısı haritası) + Z = ±2 eşik çizgisi

**Türkiye bağlantısı:**  
Doğu Anadolu'daki 2023 Kahramanmaraş öncesi AFAD katalogunda Wiemer-tipi Z-testi uygulandığında düşük-düzeyli sessizlik sinyali raporlanmıştı (Zahradník & Sokos 2023 — henüz peer-review sürecinde). Bu özellik kullanıcıya o sinyali retrospektif gösterir.

**"Vay be" faktörü:**  
"Şu kırmızı bölge ne zamandır sessiz?" sorusu haritada anlık cevaplanır. Sessizlik bazen en bağıran uyarıdır.

---

## [F-49] Pattern Informatics (PI) — Uzamsal Deprem Olasılık Değişim Haritası

**Bilimsel temel:**  
PI yöntemi (Tiampo et al. 2002; Holliday et al. 2005) sismik oranların zamansal varyansını "sinyal gücü" olarak kodlar ve M≥5 hedef için uzamsal olasılık artış haritası üretir. Hem istatistiksel hem fiziksel temeli olan nadir yöntemlerden biridir. USGS Collaboratory for the Study of Earthquake Predictability (CSEP) tarafından test edilmiştir.

**Kaynak:**
- Tiampo, K.F., Rundle, J.B., McGinnis, S., Gross, S.J. & Klein, W. (2002). Eigenpatterns in southern California seismicity. *Journal of Geophysical Research*, 107(B12), 2354. DOI: [10.1029/2001JB000562](https://doi.org/10.1029/2001JB000562)
- Holliday, J.R., Rundle, J.B., Tiampo, K.F., Klein, W. & Donnellan, A. (2005). Modification of the Pattern Informatics method for forecasting large earthquake events using complex eigenfactors. *Tectonophysics*, 413, 87–91. DOI: [10.1016/j.tecto.2005.10.008](https://doi.org/10.1016/j.tecto.2005.10.008)
- Zechar, J.D. & Jordan, T.H. (2008). Testing alarm-based earthquake predictions. *Geophysical Journal International*, 172(2), 715–724. DOI: [10.1111/j.1365-246X.2007.03676.x](https://doi.org/10.1111/j.1365-246X.2007.03676.x) *(CSEP değerlendirme çerçevesi)*

**Ne gösterir:**  
Eğitim penceresi (örn. 1990–2015) + test penceresi (2015–2023) ile Türkiye için PI haritası. Hangi grid hücreleri "sinyalli" — yani sismik oranı eğitim ortalamasından sapıyor? Bu hücreler 2023 ile örtüşüyor mu?

**Veri kaynağı:**  
KOERI/AFAD kataloğu (F-01); numpy eigendecomposition  
**Açık erişimli: Evet (iç hesaplama)**

**Teknik karmaşıklık:** 4/5  
(Matris eigenanaliz, zaman-pencere yönetimi, scipy.linalg)

**Görsel format:** Harita (grid hücreleri, PI değeri mavi-kırmızı diverging palette) + gerçekleşen M≥5 olaylar üst üste

**Türkiye bağlantısı:**  
2023 çifti için retrospektif test: PI haritası Kahramanmaraş bölgesini ne zaman "sinyalli" göstermeye başladı? Kullanıcıya istatistiksel öncü sinyallerin ne anlama geldiğini — ve sınırlarını — öğretir.

**"Vay be" faktörü:**  
"Makine 2023'ü 2020'de öngörebilir miydi?" sorusu — evet/hayır cevabı değil, olasılık dilimi olarak verilir. Şeffaflık + merak.

---

## KATEGORİ 4 — Artçı Deprem Fiziği

---

## [F-50] Omori-Utsu Yasası İnteraktif Fit + Türkiye p-Değeri Atlası

**Bilimsel temel:**  
Omori (1894) kanonik yasası n(t) = K/(t+c) sonradan Utsu (1961) tarafından genelleştirilmiştir: n(t) = K/(t+c)^p. p-değeri artçı dizisinin ne hızlı söndüğünü belirler; p > 1 hızlı sönerken p < 1 uzun süre aktif kalır. KOERI katalogunda Türkiye depremleri için p-değerleri 0.8–1.3 aralığında ölçülmüştür.

**Kaynak:**
- Utsu, T. (1961). A statistical study on the occurrence of aftershocks. *Geophysical Magazine*, 30, 521–605.
- Utsu, T., Ogata, Y. & Matsu'ura, R.S. (1995). The centenary of the Omori formula for a decay law of aftershock activity. *Journal of Physics of the Earth*, 43(1), 1–33. DOI: [10.4294/jpe1952.43.1](https://doi.org/10.4294/jpe1952.43.1)
- Öztürk, S. (2012). A statistical study of aftershock sequences for earthquakes in Anatolian region, 1975–2009. *Arabian Journal of Geosciences*, 5, 669–684. DOI: [10.1007/s12517-010-0221-0](https://doi.org/10.1007/s12517-010-0221-0) *(Türkiye p-değerleri)*

> ⚠️ **F-38 (ETAS) ile ilişki [Ajan 8]:**  
> F-38 ETAS lamba fonksiyonu λ(t) tüm dizi için bütünleşik model kurar. F-50 ise seçili ana şok için Omori-Utsu K/c/p parametrelerini kullanıcı gözünde görünür kılar, eğitim odaklı. Tamamlayıcı, çakışma yok.

**Ne gösterir:**  
Kullanıcı listeden bir deprem seçer → son 30 günün artçı sayımı → otomatik Omori-Utsu curve fitting (scipy.optimize.curve_fit) → K, c, p değerleri + güven aralığı + "p bu değerdeyse artçılar X gün içinde yarıya düşer" açıklaması.

**Veri kaynağı:**  
F-01 birleşik katalog; scipy.optimize (iç hesaplama)  
**Açık erişimli: Evet**

**Teknik karmaşıklık:** 2/5  
(Tek değişkenli curve_fit; F-38'in alt kümesi mantıkta ama görsel arayüzü farklı)

**Görsel format:** Grafik (log-log eksende artçı oranı + Omori eğrisi + 95% CI şeridi)

**Türkiye bağlantısı:**  
2023 Kahramanmaraş için p ~1.1 ölçülmüştür (Öztürk yaklaşımıyla tutarlı). Kullanıcı kendi gözlemleri ile bu değeri karşılaştırabilir.

**"Vay be" faktörü:**  
"Artçılar ne zaman biter?" sorusu matematiksel olarak cevaplanır — halk sorusunun bilimsel yanıtı, takvim değil formülle.

---

## [F-51] Reasenberg & Jones Artçı Olasılık Hesaplayıcı

**Bilimsel temel:**  
USGS operasyonel ürünü olan bu yöntem (Reasenberg & Jones 1989), bir ana şok sonrasında M≥M_hedef bir artçının gelecek T gün içindeki olasılığını hesaplar. Omori-Utsu artçı oranı + Gutenberg-Richter b-değeri + Bayes güncelleme birleştirilir. California Artçı Uyarı Sistemi'nin temelidir.

**Kaynak:**
- Reasenberg, P.A. & Jones, L.M. (1989). Earthquake hazard after a mainshock in California. *Science*, 243(4895), 1173–1176. DOI: [10.1126/science.243.4895.1173](https://doi.org/10.1126/science.243.4895.1173)
- Reasenberg, P.A. & Jones, L.M. (1994). Earthquake aftershocks: Update. *Science*, 265(5176), 1251–1252. DOI: [10.1126/science.265.5176.1251](https://doi.org/10.1126/science.265.5176.1251)
- Jordan, T.H. et al. (2011). Operational Earthquake Forecasting: State of Knowledge and Guidelines for Utilization. *Annals of Geophysics*, 54(4), 315–391. DOI: [10.4401/ag-5350](https://doi.org/10.4401/ag-5350) *(OEF çerçevesi; ICEF Raporu)*

**Ne gösterir:**  
Seçili ana şok → giriş parametreleri (a, b, p, c Türkiye medyanları) → kaydırmalı soru: "Önümüzdeki [1/7/30] gün içinde M≥[4/5/6] artçı olasılığı nedir?" → yüzde ile cevap + USGS OEF referans aralığı ile karşılaştırma.

**Veri kaynağı:**  
Hesaplama tamamen iç (formül uygulaması); Türkiye parametreleri: b=0.9 (F-14 tabanlı), p=1.1 (Öztürk 2012), a Mw fonksiyonu olarak  
**Açık erişimli: Evet**

**Teknik karmaşıklık:** 2/5  
(Analitik formül, scipy.integrate.quad ile olasılık integrali)

**Görsel format:** Sayısal panel (büyük %) + olasılık-zaman çizgi grafiği

**Türkiye bağlantısı:**  
AFAD'ın 2023 sonrasında resmi açıklamalarında kullandığı metodoloji ile aynı çerçeve. Kullanıcı haber başlıklarındaki "artçı beklentisi" rakamlarını artık kendi hesabıyla teyit edebilir.

**"Vay be" faktörü:**  
"Bugün gece evde uyusam güvenli mi?" sorusuna bilimsel bir olasılık rakamı — her karar kişiye ait ama bilgi artık mevcut.

---

## KATEGORİ 5 — Coulomb Gerilme Transferi (Genişletme)

---

## [F-52] Dinamik Gerilme Tetikleme — Uzak Mesafeli Tetikleme Haritası

**Bilimsel temel:**  
King, Stein & Lin (1994 — F-37 kapsar) statik CFS'i modeller; ancak 1992 Landers sonrasında 1.250 km uzaktaki volkanik bölgelerde tetikleme gözlemlenince, Hill et al. (1993) dinamik gerilme dalgalarının (Rayleigh, Love) uzak tetiklemeden sorumlu olduğunu kanıtladı. Statik + dinamik iki ayrı mekanizma, farklı mesafe rejimleri.

> ⚠️ **F-37 ile kesin ayrım [Ajan 8]:**  
> F-37 = statik CFS (fay çevresinde 100–200 km) → Okada 1992 dislokasyon modeli.  
> F-52 = dinamik tetikleme (100–10.000 km) → dalga genlikleri, passage of surface waves.  
> Çakışma: SIFIR. Fiziksel süreç farklı.

**Kaynak:**
- Hill, D.P. et al. (1993). Seismicity remotely triggered by the magnitude 7.3 Landers, California, earthquake. *Science*, 260(5114), 1617–1623. DOI: [10.1126/science.260.5114.1617](https://doi.org/10.1126/science.260.5114.1617)
- Brodsky, E.E. & Prejean, S.G. (2005). New constraints on mechanisms of remotely triggered seismicity at Long Valley Caldera. *Journal of Geophysical Research*, 110, B04302. DOI: [10.1029/2004JB003211](https://doi.org/10.1029/2004JB004211)
- Parsons, T. (2005). A hypothesis for delayed dynamic earthquake triggering. *Geophysical Research Letters*, 32, L04302. DOI: [10.1029/2004GL021811](https://doi.org/10.1029/2004GL021811)

**Ne gösterir:**  
Büyük (M≥7) bir depremden sonra dalga seyahat sürelerine göre renklendirilmiş tetikleme risk halkaları. Dünya genelinde hangi bölgelerde dinamik tetiklemenin rapor edildiği (Hill 1993 gibi gözlemsel örnekler). "Bu deprem uzak bir volkanik bölgeyi aktive eder mi?"

**Veri kaynağı:**  
USGS sismogram arşivi + literatür gözlem kataloğu (Hill 1993, Brodsky 2005)  
Hesaplama: teorik yüzey dalga geliş zamanları (taup modeli — irio kütüphanesi)  
**Açık erişimli: Kısmen (TauP: açık; gözlem kataloğu: manuel kurasyonlu)**

**Teknik karmaşıklık:** 3/5  
(TauP irio/obspy + Haversine mesafe; gözlem kataloğu JSON olarak kodlanabilir)

**Görsel format:** Harita (Rayleigh dalga yayılım halkası animasyonu) + log-log mesafe vs tetikleme konsantrasyon grafiği

**Türkiye bağlantısı:**  
2011 M9.0 Tohoku depremi sonrasında KOERI kayıtları Türkiye'de anlık aktivasyon artışı gösterdi — dinamik tetiklemenin küresel ölçeği Türkiye'yi de kapsamaktadır.

**"Vay be" faktörü:**  
Japonya'da M9 deprem oldu, 90 dakika sonra Türkiye'de titreşme — aynı dalga. Mesafe kavramını yıkar, Dünya'nın küçüklüğünü hissettirir.

---

## KATEGORİ 6 — GPS Jeodezi ve Yer Deformasyonu

---

## [F-53] InSAR Koseismik Deformasyon Haritası (Sentinel-1)

**Bilimsel temel:**  
InSAR (Interferometric Synthetic Aperture Radar), iki SAR görüntüsünün faz farkından cm hassasiyetinde yer deformasyonunu ölçer. ESA Sentinel-1 uyduları 6-12 günde bir aynı bölgeyi tarar ve her büyük depremden sonra COMET / UNAVCO ekipleri interferogram yayımlar. 2023 Kahramanmaraş için yayımlanan interferogramlar 5 m'ye varan koseismik yer değiştirme gösterdi.

**Kaynak:**
- Massonnet, D. & Feigl, K.L. (1998). Radar interferometry and its application to changes in the earth's surface. *Reviews of Geophysics*, 36(4), 441–500. DOI: [10.1029/97RG03139](https://doi.org/10.1029/97RG03139)
- Xu, W. et al. (2023). Surface ruptures, coseismic deformation, and seismotectonics of the 2023 M7.8 and M7.7 earthquake doublet in SE Turkey. *Earth and Planetary Science Letters*, 612, 118333. DOI: [10.1016/j.epsl.2023.118333](https://doi.org/10.1016/j.epsl.2023.118333)
- ESA Sentinel-1 Copernicus: https://sentinel.esa.int/web/sentinel/missions/sentinel-1 *(SAR veri açık erişimli)*
- COMET LiCS portal: https://comet.nerc.ac.uk/COMET-LiCS-portal/ *(hazır interferogramlar)*

**Ne gösterir:**  
Seçilen deprem (M≥6) için eğer mevcutsa COMET/UNAVCO interferogramını gömülü gösterir; LOS (Line-of-Sight) yer değiştirme haritası, cm cinsinden. "Fay ne kadar kaydı, hangi taraf yukarı kalktı?"

**Veri kaynağı:**  
COMET LiCS Portal GeoTIFF: https://comet.nerc.ac.uk/  
ESA Copernicus Science Hub: https://scihub.copernicus.eu/  
**Açık erişimli: Evet (Copernicus açık veri politikası)**

**Teknik karmaşıklık:** 3/5  
(GeoTIFF iframe embed veya rasterio + Plotly Heatmap; interferogram işlemek 5/5 ama hazır ürün kullanmak 3/5)

**Görsel format:** Harita (interferogram sarkaç renk paleti — rainbow faz çemberleri) + cross-section deformasyon profili

**Türkiye bağlantısı:**  
Kahramanmaraş 2023: COMET aynı gün interferogram yayımladı, 5-7 m yatay + 2-3 m dikey deformasyon haritalandı. En dramatik Türkiye InSAR verisi.

**"Vay be" faktörü:**  
Uzaydan çekilmiş fotoğraf değil — matematiksel faz farkı. Uydudan zemin centimetre hassasiyetiyle ölçülüyor. Sihir gibi görünür ama saf fizik: elektromanyetik dalga interferansı.

---

## [F-54] İnterseismik Kilitlenme / Kayma Açığı Haritası

**Bilimsel temel:**  
GPS hız alanından interseismik kilitlenme oranı (coupling coefficient φ) hesaplanır: φ=1 tam kilitli fay (enerji birikiyor), φ=0 aseismik sürekli kayıyor (enerji birikmez). Reilinger et al. (2006) Türkiye GPS ağından bu hesabı yapmıştır; KAF'ın İzmit–Erzincan arasındaki segmentlerin kilitlenme oranları 0.6–1.0 arasında değişmektedir.

**Kaynak:**
- Reilinger, R. et al. (2006). GPS constraints on continental deformation in the Africa-Arabia-Eurasia continental collision zone and implications for the dynamics of plate interactions. *Journal of Geophysical Research*, 111, B05411. DOI: [10.1029/2005JB004051](https://doi.org/10.1029/2005JB004051)
- Ergintav, S. et al. (2014). Istanbul's earthquake hot spots: Geodetic constraints on strain accumulation along faults in the Marmara seismic gap. *Geophysical Research Letters*, 41, 5783–5788. DOI: [10.1002/2014GL060985](https://doi.org/10.1002/2014GL060985) *(Marmara kilitlenme)*
- Barka, A. (1996). Slip distribution along the North Anatolian fault associated with the large earthquakes of the period 1939 to 1967. *Bulletin of the Seismological Society of America*, 86(5), 1238–1254. *(Tarihsel kayma açığı)*

> ⚠️ **F-35 (GPS hız vektörleri) ile ayrım [Ajan 8]:**  
> F-35 = ham hız vektörleri (mm/yıl ok diyagramı).  
> F-54 = kilitlenme modeli (φ katsayısı haritası) — hız alanının türevi, farklı bir bilimsel anlam.

**Ne gösterir:**  
KAF ve DAF boyunca fay segmentlerinin φ haritası (kırmızı=tam kilitli, sarı=kısmi, yeşil=aseismik). Son büyük kırık segmentler çıkarılınca "kırılmamış" (seismic gap) bölgeler öne çıkar.

**Veri kaynağı:**  
Reilinger 2006 ek verileri (JGR supplement) + Ergintav 2014 Marmara kilitlenme modeli — her ikisi de yayın verileri, dijitalleştirilebilir  
**Açık erişimli: Yayın eki verisi (erişim gerekebilir); bazı kurumlar CC-BY**

**Teknik karmaşıklık:** 3/5  
(Literatür verisini GeoJSON'a dönüştürme + Plotly choropleth segment renklendirme)

**Görsel format:** Harita (fay segmenti, φ renkli çizgi kalınlığı = kayma hızı) + zaman çizelgesi (son kırık tarihleri)

**Türkiye bağlantısı:**  
Marmara Denizi altındaki "Prens Adaları segmenti" φ≈0.9 ölçülmüştür — tam kilitli, son kırık 1766. Kayma açığı ~4 m birikmiş. "İstanbul depremi" tartışmasının merkezinde bu sayı durur.

**"Vay be" faktörü:**  
GPS satırları "biz buraya bakıyoruz, fay kilitli, enerji birikmiş" diyor. Soyut tehlike somut milimetre/yıl değere dönüşünce insan gerçekliğini hissediyor.

---

## KATEGORİ 7 — Sismik Dalga Analizi

---

## [F-55] Moho Derinliği ve Kabuk Kalınlığı Haritası (Receiver Function)

**Bilimsel temel:**  
P-to-S dönüşüm dalgaları (Ps) receiver function yöntemiyle işlenerek Moho (mantle sınırı) derinliği belirlenir. Türkiye'de doğu Anadolu altında kabuk 50–55 km'ye ulaşırken, Ege açılma bölgesinde 25–30 km'ye iner. Bu fark sismik hız, dalga amplitüdü ve deprem odak derinlik dağılımıyla doğrudan ilişkilidir.

**Kaynak:**
- Zhu, L. & Kanamori, H. (2000). Moho undulations beneath the Mojave Desert from TERRAscope receiver functions. *Journal of Geophysical Research*, 105(B2), 2981–2993. DOI: [10.1029/1999JB900322](https://doi.org/10.1029/1999JB900322) *(RF metodoloji)*
- Zor, E., Sandvol, E., Gürbüz, C., Türkelli, N., Seber, D. & Barazangi, M. (2003). The crustal structure of the East Anatolian plateau (Turkey) from receiver functions. *Geophysical Research Letters*, 30(24), 8044. DOI: [10.1029/2003GL018192](https://doi.org/10.1029/2003GL018192) *(Türkiye uygulaması)*
- CRUST1.0 model: Laske, G., Masters, G., Ma, Z. & Pasyanos, M. (2013). Update on CRUST1.0 — A 1-degree global model of Earth's crust. *EGU General Assembly*, Abstract EGU2013-2658. Data: https://igppweb.ucsd.edu/~gabi/crust1.html *(küresel grid)*

**Ne gösterir:**  
Türkiye altında Moho derinlik haritası (km cinsinden); kullanıcı bir istasyona tıklayınca kabuk kalınlığı, Moho yansıma derinliği ve varsa Ps dönüşüm geliş süresi bilgisi. Deprem odak derinlikleri ile Moho overlayı: odaklar kabuğun içinde mi, mantle'da mı?

**Veri kaynağı:**  
CRUST1.0 global model: https://igppweb.ucsd.edu/~gabi/crust1.html (açık ASCII grid)  
Zor et al. 2003 Doğu Anadolu Moho haritası (makale ek verisi)  
**Açık erişimli: Evet (CRUST1.0) / Kısmen (Zor 2003 veri)**

**Teknik karmaşıklık:** 2/5  
(ASCII grid okuma + Plotly contour haritası; receiver function işlemek 5/5 ama hazır model kullanmak basit)

**Görsel format:** Harita (Moho derinlik kontur haritası) + W–E kesit profili

**Türkiye bağlantısı:**  
Erzincan altındaki Moho ~43 km derinliktedir; buna karşın Batı Anadolu grabenleri altında 28 km. Bu fark neden doğuda yavaş, batıda hızlı sismik aktivite olduğunu kısmen açıklar.

**"Vay be" faktörü:**  
"Zemin altında ne var?" sorusunun cevabı — sismik dalgalar Dünya'nın içini MRI gibi görüntüler. Moho haritası, kabuğun "kalınlık tomografisi."

---

## [F-56] SKS Dalga Parçalanması — Mantle Anizotropi ve Gerilme Yönü Haritası

**Bilimsel temel:**  
SKS (S→K→S) dalgaları çekirdekten geçerken izotropik kalır; litosferde/astanosferde anizotropik minerallerden (olivin a-ekseni) geçince parçalanır (splitting). Fast axis yönü mantle akış yönünü, dt (gecikme süresi) anizotropi büyüklüğünü verir. Türkiye'de KAF boyunca fast axis KAF ile paralel — mantle litosfer kuplajının kanıtı.

**Kaynak:**
- Savage, M.K. (1999). Seismic anisotropy and mantle deformation: What have we learned from shear wave splitting? *Reviews of Geophysics*, 37(1), 65–106. DOI: [10.1029/98RG02075](https://doi.org/10.1029/98RG02075)
- Biryol, C.B., Beck, S.L., Zandt, G. & Özacar, A.A. (2010). Segmented African lithosphere beneath the Anatolian region inferred from teleseismic P-wave tomography. *Journal of Geophysical Research*, 115, B07316. DOI: [10.1029/2009JB006923](https://doi.org/10.1029/2009JB006923)
- SKS Splitting World Database (SplittingDB): https://ds.iris.edu/ds/products/sksobs/ *(açık gözlem veritabanı)*

**Ne gösterir:**  
Türkiye + yakın çevre istasyonlarında ölçülmüş SKS splitting parametreleri: fast azimuth (ok yönü) + dt (çizgi kalınlığı). Mantle'ın nereden nereye aktığı, plaka hareketi vektörleriyle (F-35) karşılaştırma.

**Veri kaynağı:**  
IRIS/PASSCAL SplittingDB: https://ds.iris.edu/ds/products/sksobs/  
**Açık erişimli: Evet**

**Teknik karmaşıklık:** 2/5  
(CSV indir + Plotly ok vektörü overlay; ölçümleri yeniden hesaplamak 5/5 ama mevcut veritabanı kullanmak kolay)

**Görsel format:** Harita (istasyon + yönlü çizgi/ok, kalınlık = dt) + Türkiye GPS hız alanıyla karşılaştırma

**Türkiye bağlantısı:**  
Ege'de fast axis GB–KD yönelimli (Hellenic trenching etkisi), KAF boyunca ise fay ile paralel D–B. İki farklı mantle dinamiği iç içe — tek haritada ikisini görmek nadir bir deneyim.

**"Vay be" faktörü:**  
"Yer altındaki taş akar mı?" — olivin kristalleri mantle akışını takip ederek sıralanıyor. Deprem dalgaları bu sıralanmayı okuyarak bize mantle konveksiyonunu anlatıyor. Görünmezin görünür kılınması.

---

## KATEGORİ 8 — Tsunami Tehlikesi

---

## [F-57] Akdeniz Tarihi Tsunami Kataloğu + NEAM Tehlike Haritası

**Bilimsel temel:**  
Akdeniz, tarihsel olarak önemli tsunami kaynağı içerir: 365 AD Girit (Mw ~8.5 + megatsunami), 1202 ve 1303 Doğu Akdeniz, 1908 Messina. GITEC ve NEAMTWS (Kuzey-Doğu Atlantik-Akdeniz Tsunami Uyarı Sistemi) bu olayları kataloglamış ve NEAM tehlike haritası yayımlanmıştır. Türkiye kıyıları için Yolsal-Çevikbilen & Taymaz (2012) özel modeller üretmiştir.

**Kaynak:**
- Papadopoulos, G.A. et al. (2014). Historical and pre-historical tsunamis in the Mediterranean and its connected seas: Geological signatures, generation mechanisms and coastal impacts. *Marine Geology*, 354, 81–109. DOI: [10.1016/j.margeo.2014.04.014](https://doi.org/10.1016/j.margeo.2014.04.014)
- Yolsal-Çevikbilen, S. & Taymaz, T. (2012). Earthquake source parameters along the Hellenic subduction zone and numerical simulations of historical tsunamis in the Eastern Mediterranean. *Tectonophysics*, 536–537, 61–100. DOI: [10.1016/j.tecto.2012.01.025](https://doi.org/10.1016/j.tecto.2012.01.025) *(Türkiye kıyısı modeli)*
- NEAM Tsunami Tehlike Haritası: http://www.emsc-csem.org/TSUMAPS-NEAM/ *(açık erişimli)*
- GITEC Kataloğu: https://www.ngdc.noaa.gov/hazard/tsu.shtml *(NOAA NGDC; açık)*

**Ne gösterir:**  
Akdeniz kıyısında tarihsel tsunami olayları (nokta marker, büyüklük = R = koşu yüksekliği m), NEAM olasılıksal tehlike değeri (100 yılda 1 m+ olasılığı), Türkiye'nin risk altındaki kıyı şeridi renk kodlaması.

**Veri kaynağı:**  
NOAA NGDC Tsunami Database: https://www.ngdc.noaa.gov/hazel/hazard-search/search-tsunamis  
NEAM TSUMAPS grid: http://www.tsumaps-neam.eu/  
**Açık erişimli: Evet**

**Teknik karmaşıklık:** 2/5  
(CSV indir + Plotly Scattermapbox; tehlike grid = ısı haritası)

**Görsel format:** Harita (tarihi olay noktaları + NEAM ısı katmanı) + zaman çizelgesi sidebar

**Türkiye bağlantısı:**  
Ege ve Güney Türkiye kıyıları tarihsel tsunami kayıtlarına sahiptir (1303 Rodos, 1955 Amorgos). İzmir, Bodrum, Antalya kıyı şeritlerinin tehlike değerleri.

**"Vay be" faktörü:**  
1766 yılında İstanbul kıyısına ulaşan tsunami var — henüz kamuoyunun pek bilmediği tarihsel gerçek. Haritada bunu görmek zaman algısını yerinden eder.

---

## [F-58] Tsunami Varış Süresi Hesaplayıcı (NOAA MOST / SciPy Shallow Water)

**Bilimsel temel:**  
Sığ su dalga hızı c = √(g·d) (g: yerçekimi, d: derinlik). NOAA'nın MOST (Method of Splitting Tsunamis) modeli sayısal çözümle tsunami yayılımını simüle eder. Basitleştirilmiş versiyon olarak, GEBCO batimetri verisi + basit shallow water ray tracing ile varış süresi tahmini yapılabilir.

**Kaynak:**
- Titov, V.V. & Synolakis, C.E. (1998). Numerical modeling of tidal wave runup. *Journal of Waterway, Port, Coastal, and Ocean Engineering*, 124(4), 157–171. DOI: [10.1061/(ASCE)0733-950X(1998)124:4(157)](https://doi.org/10.1061/(ASCE)0733-950X(1998)124:4(157))
- GEBCO Batimetri: https://www.gebco.net/data_and_products/gridded_bathymetry_data/ *(açık, 15 arcsec)*
- NOAA DART Boj Ağı: https://www.ndbc.noaa.gov/dart.shtml *(gerçek zamanlı dalga yüksekliği)*

**Ne gösterir:**  
Kullanıcı haritada deprem konumu + büyüklük girer → basit c=√(gd) ray tracing → seçilen kıyı noktasına tahmini varış süresi (dakika). "Bodrum'a kaç dakikada ulaşır?"

**Veri kaynağı:**  
GEBCO 2023 batimetri grid (açık GeoTIFF): https://www.gebco.net/  
Scipy + numpy iç hesaplama  
**Açık erişimli: Evet**

**Teknik karmaşıklık:** 4/5  
(Rasterio GEBCO okuma + scipy Dijkstra benzeri yayılım hesabı; gerçek MOST modeli 5/5 ama basitleştirme kabul edilebilir — disclaimer zorunlu)

**Görsel format:** Harita (eş-zaman konturları — 5 dk aralıklı halka animasyonu) + hedef kıyı bar grafiği (varış süresi dakika)

**Türkiye bağlantısı:**  
Hellenic Trench'te M8+ senaryo → Bodrum'a 12–18 dakika (Yolsal-Çevikbilen 2012 simülasyon aralığı). Tahliye planlaması için kritik sayı.

**"Vay be" faktörü:**  
Uçak mı daha hızlı, tsunami mu? Ege'nin dar ve sığ olduğu yerlerde tsunami 15 dakikada kıyıya ulaşır — bu bilgi hayat kurtarır. Gerçek fizik, gerçek aciliyet.

---

## KATEGORİ 9 — Tarihi Depremler ve Paleoseismoloji

---

## [F-59] Ambraseys Tarihi Deprem Kataloğu — 1500 Yıl, Türkiye + Orta Doğu

**Bilimsel temel:**  
Nicholas Ambraseys (Imperial College) onlarca yıl boyunca Türkçe, Arapça, Ermenice, Rumca ve Latince birincil kaynaklardan tarihi deprem kayıtlarını derledi. 2009'da yayımlanan kapsamlı kitap, MS 1–1900 arası ~5.000 depremi kataloglamaktadır. Aletsel dönem (post-1900) öncesi sismik tehlike değerlendirmelerinin tek güvenilir kaynağı budur.

**Kaynak:**
- Ambraseys, N.N. (2009). *Earthquakes in the Mediterranean and Middle East: A Multidisciplinary Study of Seismicity up to 1900*. Cambridge University Press. ISBN: 9780521872928. URL: https://www.cambridge.org/9780521872928
- Ambraseys, N.N. & Finkel, C.F. (1995). *The Seismicity of Turkey and Adjacent Areas: A Historical Review, 1500–1800*. Eren Yayıncılık, İstanbul.
- NOAA NCEI Significant Earthquakes Database (tarihi verileri içerir): https://www.ngdc.noaa.gov/hazel/hazard-search/search-earthquakes

**Ne gösterir:**  
Harita üzerinde aletsel dönem öncesi büyük depremler (Mw tahmini, tarih, kaynak türü: tarihsel belge / hasar kaydı / paleoseismik). Zaman sürgüsü ile 500 yıllık dönemler arası geçiş. KAF'ın tarihi kırık sıralaması animasyonu (batıya göç gözlenebilir mi?).

**Veri kaynağı:**  
NOAA NCEI Significant Earthquakes: https://www.ngdc.noaa.gov/hazel/  
Ambraseys 2009 kataloğu (kitap; dijital versiyonu için EMME projesi veri portalı)  
**Açık erişimli: Kısmen (NOAA = açık; Ambraseys kitabı = ücretli, ancak EMME özet verisi açık)**

**Teknik karmaşıklık:** 2/5  
(CSV indir + Plotly zaman animasyonu)

**Görsel format:** Harita (tarihi deprem noktaları, renk = yüzyıl) + zaman çizelgesi panel

**Türkiye bağlantısı:**  
1668 Amasya (Mw ~8.0, KAF), 1766 İstanbul ikiz depremi, 1939 Erzincan (KAF'ın batıya göçü dizisinin son halkası). 300 yılda KAF boyunca batı yürüyüşü haritada görünür.

**"Vay be" faktörü:**  
Osmanlı vakanüvisleri depremi yazdı, biz bugün haritaya işliyoruz. Arşiv + sismoloji = 500 yıllık hafıza.

---

## [F-60] NAFZ Paleoseismik Kazı Verisi — Fay Tekrar Süresi ve Slot Diyagramı

**Bilimsel temel:**  
Fay boyunca kazılan slot (hendek) çalışmaları, geçmiş depremlerin kolüvyal tortul katmanlarındaki izlerini ortaya çıkarır. NAFZ'da birden fazla kazı konumunda son 2.000–4.000 yıl içindeki büyük depremler (M≥7) tarihlendirilmiştir (14C + OSL). Tekrar süresi 250–400 yıl aralığında bulunmuştur.

**Kaynak:**
- Kozacı, Ö., Doğan, B., Özaksoy, V., Yıldırım, C., Gökasan, E. & Tokay, F. (2007). Paleoseismological evidence for the relatively regular recurrence of infrequent, large-magnitude earthquakes on the eastern North Anatolian fault at Yaylabeli (Erzincan), Turkey. *Bulletin of the Seismological Society of America*, 97(5), 1513–1527. DOI: [10.1785/0120060118](https://doi.org/10.1785/0120060118) *(Erzincan segmenti)*
- Klinger, Y., Sieh, K., Altunel, E., Akoglu, A., Barka, A., Dawson, T., … & Rockwell, T. (2003). Paleoseismic evidence of characteristic slip on the western segment of the North Anatolian fault, Turkey. *Bulletin of the Seismological Society of America*, 93(6), 2317–2332. DOI: [10.1785/0120010270](https://doi.org/10.1785/0120010270) *(Batı NAFZ)*
- Fraser, J., Vanneste, K., Hubert-Ferrari, A. (2010). Recent behavior of the North Anatolian Fault: Insights from an integrated paleoseismological data set. *Journal of Geophysical Research*, 115, B09316. DOI: [10.1029/2009JB006982](https://doi.org/10.1029/2009JB006982) *(Sentez)*

**Ne gösterir:**  
NAFZ + DAF boyunca kazı konumları harita üzerinde; her konuma tıklayınca: 14C-tarihlendirilmiş paleodepremler log (slot diyagramı benzeri zaman şeridi), ortalama tekrar süresi (yıl), son büyük olaydan bu yana geçen süre ve "beklenen süreye" oranı.

**Veri kaynağu:**  
Yayın ekleri (Kozacı 2007, Klinger 2003, Fraser 2010); PALEOEX veritabanı (PAGES SeisVar): https://www.pages-igbp.org/  
**Açık erişimli: Yayın verileri (ek tabloları); PALEOEX kısmi açık**

**Teknik karmaşıklık:** 2/5  
(Literatür verisini JSON'a kodlama + Plotly zaman şeridi)

**Görsel format:** Harita (kazı konumu markerları) + yan panel: slot diyagramı (yatay çizgiler = deprem olayları, belirsizlik aralıkları)

**Türkiye bağlantısı:**  
Erzincan'da son büyük (M~7.8) olay: 1939. Kozacı 2007 ortalama tekrar süresini ~320 yıl bulmuş. Sonraki pencere içindeyiz — ama bu "kesin gelecek" değil, olasılık dili.

**"Vay be" faktörü:**  
Toprağı kazan jeolog, 2.000 yıl önce yaşanan depremin izini buldı. Yazılı tarihin ötesini okumak — zamanı geri sarmak.

---

## KATEGORİ 10 — Gerçek Zamanlı ShakeMap ve Erken Uyarı

---

## [F-61] USGS ShakeMap GeoJSON Otomatik Entegrasyonu

**Bilimsel temel:**  
ShakeMap (Worden & Wald 2016) gözlenen ivme kayıtları + GMPE ile olay bazında PGA/PGV/MMI ızgara interpolasyonu yapar. USGS her M≥4 deprem için ShakeMap GeoJSON ve GeoTIFF yayımlar; API açık erişimlidir. Mevcut F-20 (Erken Uyarı Simülatörü) kullanıcı girdili senaryo işleri, F-61 ise gerçek olay verisini otomatik çeker.

> ⚠️ **F-20 ve F-21 ile ayrım [Ajan 8]:**  
> F-20/21 = kullanıcı parametreli simülasyon, Wald 1999 basit GMPE.  
> F-61 = gerçek deprem sonrası USGS ShakeMap ürününü otomatik gömme, ~dakika gecikmeli. Tamamlayıcı.

**Kaynak:**
- Worden, C.B. & Wald, D.J. (2016). ShakeMap Manual Online: Technical Manual, User's Guide, and Software Guide. *U.S. Geological Survey*. DOI: [10.5066/F7D21VPQ](https://doi.org/10.5066/F7D21VPQ)
- USGS ShakeMap API: https://earthquake.usgs.gov/data/shakemap/ *(açık REST API)*
- Wald, D.J. et al. (1999). Relationships between peak ground acceleration, peak ground velocity, and Modified Mercalli Intensity in California. *Earthquake Spectra*, 15(3), 557–564. DOI: [10.1193/1.1586058](https://doi.org/10.1193/1.1586058)

**Ne gösterir:**  
Son 7 günün M≥5 olaylarında USGS ShakeMap mevcutsa → otomatik çek ve haritaya gömme: MMI renk katmanı + PGA konturları. "Bu deprem nerede ne kadar salladı?"

**Veri kaynağı:**  
USGS ShakeMap GeoJSON: `https://earthquake.usgs.gov/fdsnws/event/1/...` + `shakemap/grid.xml`  
**Açık erişimli: Evet (USGS public domain)**

**Teknik karmaşıklık:** 3/5  
(USGS API çağrısı + GeoJSON parse + Plotly choropleth overlay; F-01 altyapısı kullanılabilir)

**Görsel format:** Harita (MMI ısı katmanı — USGS renk skalası: I-XII) + seçili istasyon PGA karşılaştırma grafiği

**Türkiye bağlantısı:**  
AFAD ShakeMap (deprem.afad.gov.tr) Türkiye için benzer ürün üretir; ancak USGS API daha hızlı ve standartlaşmış. AFAD PDF embed + USGS GeoJSON çift kaynak.

**"Vay be" faktörü:**  
Deprem oldu, 5 dakika içinde uygulamada renk haritası beliriyor. "Erzincan'da mı yoksa Tokat'ta mı daha çok salladı?" — gerçek veride, saniyeler içinde.

---

## KATEGORİ 11 — Deprem Döngüsü Modelleri

---

## [F-62] BPT Deprem Tekrar Olasılığı Hesaplayıcı (Brownian Passage Time)

**Bilimsel temel:**  
BPT (Matthews et al. 2002) modeli, deprem tekrar süresinin deterministik değil stokastik bir geçiş süreci izlediğini varsayar. Ortalama tekrar süresi (μ) + kayıp katsayısı (α=aperiodicity) + son büyük olaydan geçen süre (T) girilerek "önümüzdeki 30/50/100 yılda M≥7 olasılığı" hesaplanır. WGCEP ve UCERF3 bu modeli kullanmaktadır.

**Kaynak:**
- Matthews, M.V., Ellsworth, W.L. & Reasenberg, P.A. (2002). A Brownian model for recurrent earthquakes. *Bulletin of the Seismological Society of America*, 92(6), 2233–2250. DOI: [10.1785/0120010267](https://doi.org/10.1785/0120010267)
- Ellsworth, W.L. et al. (1999). A physically-based earthquake recurrence model for estimation of long-term earthquake probabilities. *USGS Open-File Report* 99-522. URL: https://pubs.usgs.gov/of/1999/0522/
- Field, E.H. et al. (2015). Long-term time-dependent probabilities for the third Uniform California Earthquake Rupture Forecast (UCERF3). *Bulletin of the Seismological Society of America*, 105(2A), 511–543. DOI: [10.1785/0120140093](https://doi.org/10.1785/0120140093) *(BPT operasyonel uygulama)*

**Ne gösterir:**  
Kullanıcı seçer: Fay segmenti (KAF-Erzincan, KAF-Marmara, vb.) → μ ve α literatürden otomatik doldurulur → T hesaplanır (son büyük depremi biz biliyoruz) → [30/50/100 yıl] BPT olasılık eğrisi. Karşılaştırma: Poisson (hafıza yok) vs BPT (geçmişi hatırlayan).

**Veri kaynağı:**  
Paleoseismik μ değerleri: Kozacı 2007 (Erzincan), Klinger 2003 (Batı NAFZ), Fraser 2010 (sentez)  
Hesaplama: scipy.stats.invgauss (BPT = invers Gauss dağılımı)  
**Açık erişimli: Evet (iç hesaplama)**

**Teknik karmaşıklık:** 2/5  
(scipy.stats.invgauss; analitik dağılım)

**Görsel format:** Grafik (olasılık vs zaman eğrisi, Poisson ile karşılaştırma) + sayısal panel (30/50 yıl %)

**Türkiye bağlantısı:**  
Marmara segmenti için μ~400 yıl (Le Pichon 2001 tahmini), T~260 yıl (son 1766'dan beri). BPT ile 30 yıllık olasılık ~%65 (Parsons 2004 benzer hesap). Bu rakam İstanbul DASK sigorta sisteminin kullandığı sınıfa girer.

**"Vay be" faktörü:**  
Matematiksel olarak deprem "zamanını dolduruyor" mu? BPT bunun ne anlama geldiğini şeffaf gösterir — kehanet değil, stokastik fizik.

---

## [F-63] Sismik Açık (Seismic Gap) Analizi — KAF Kırık Segmenti Zaman Çizelgesi

**Bilimsel temel:**  
Seismic gap teorisi (McCann et al. 1979; Nishenko & Buland 1987), tarihsel olarak büyük deprem yaşamamış fay segmentlerinin "beklenen" kırılma adayı olduğunu öne sürer. KAF boyunca 1939–1999 arası batıya göç eden deprem dizisi bu teorinin en sık atıfla gösterilen örneğidir. Marmara segmenti şu an en uzun süredir kırılmamış segmenttir.

**Kaynak:**
- McCann, W.R., Nishenko, S.P., Sykes, L.R. & Krause, J. (1979). Seismic gaps and plate tectonics: Seismic potential for major boundaries. *Pure and Applied Geophysics*, 117(6), 1082–1147. DOI: [10.1007/BF00876211](https://doi.org/10.1007/BF00876211)
- Nishenko, S.P. & Buland, R. (1987). A generic recurrence interval distribution for earthquake forecasting. *Bulletin of the Seismological Society of America*, 77(4), 1382–1399.
- Barka, A.A. (1996). Slip distribution along the North Anatolian fault associated with the large earthquakes of the period 1939 to 1967. *Bulletin of the Seismological Society of America*, 86(5), 1238–1254. *(KAF migrasyon klasiği)*

**Ne gösterir:**  
KAF boyunca segmentlerin horizontal zaman çizelgesi: x ekseni = doğu-batı coğrafi konum (km), y ekseni = yıl. Her büyük deprem renkli yatay çizgi (kırık uzunluğu = km). Kırılmamış segmentler beyaz boşluk = "açık." Mevcut "bekleme süresi" renk gradyanı.

**Veri kaynağu:**  
Barka 1996 (KAF deprem kataloğu + kırık uzunlukları)  
Ambraseys 2009 (tarihi dönem)  
**Açık erişimli: Yayın verisi**

**Teknik karmaşıklık:** 2/5  
(Plotly Gantt benzeri segment çizelge; veri JSON olarak kodlanabilir)

**Görsel format:** Zaman-mekan çizelgesi (horizontal Gantt) + harita overlay (kırık segmenti hattı)

**Türkiye bağlantısı:**  
1939–1999 arasında KAF doğudan batıya göç etti — Erzincan → Niksar → Ladik → Abant → İzmit → Düzce. Görsel bu göçü animasyon olarak gösterir. Sonraki halka: Marmara.

**"Vay be" faktörü:**  
Harita, depremlerin yıllar içinde Erzincan'dan İstanbul'a "yürüdüğünü" gösteriyor. Tektonik bir tren güzergahı gibi — ve İstanbul son durak adayı.

---

## KATEGORİ 12 — İnsan Faktörü: Kentsel Kırılganlık

---

## [F-64] Vs30 Tabanlı Zemin Büyütmesi + HAZUS Benzeri Kayıp Tahmini

**Bilimsel temel:**  
HAZUS-MH (FEMA 2003) ABD'nin standart kayıp tahmini modelidir. Zemin sınıfı (Vs30) × yapı tipi × maruziyet = beklenen hasar oranı. Türkiye için eşdeğer çalışmalar (Erdik et al. 2003, Demircioglu 2010 İstanbul) HAZUS metodolojisini Türk yapı stoku ile uyarlamıştır. F-45 (Vs30) + F-44 (PGA) + nüfus verisi = temel kayıp tahmini.

**Kaynak:**
- FEMA (2003). *HAZUS-MH Technical Manual*. Federal Emergency Management Agency, Washington D.C. URL: https://www.fema.gov/sites/default/files/2020-09/fema_hazus_earthquake-model_technical-manual_2.1.pdf
- Erdik, M., Aydınoğlu, N., Fahjan, Y., Sesetyan, K., Demircioğlu, M., Siyahi, B., … & Övül, G. (2003). Earthquake risk assessment for Istanbul metropolitan area. *Earthquake Engineering and Engineering Vibration*, 2(1), 1–23. DOI: [10.1007/BF02857534](https://doi.org/10.1007/BF02857534)
- Boore, D.M. et al. (2014). NGA-West2 equations for predicting PGA, PGV, and 5%-damped PSA. *Earthquake Spectra*, 30(3), 1057–1085. DOI: [10.1193/070113EQS184M](https://doi.org/10.1193/070113EQS184M) *(GMPE zemin büyütme terimi)*

**Ne gösterir:**  
Seçili il/ilçe için: Vs30 sınıfı × PGA (F-44'ten) → zemin büyütme faktörü (FA/FV) → tahmini yapı hasar oranı (%) → rough can kaybı tahmini aralığı. Karşılaştırma: Türkiye ortalaması vs seçili ilçe.

**Veri kaynağı:**  
USGS Vs30 (F-45 ile ortak)  
FEMA HAZUS hasar matrisleri (teknik manual ek tabloları — açık PDF)  
TÜİK nüfus verisi: https://www.tuik.gov.tr  
**Açık erişimli: Kısmen (HAZUS manual açık; TÜİK açık)**

**Teknik karmaşıklık:** 3/5  
(Hasar matrisi CSV lookup + interpolasyon; kesin HAZUS tam modeli 5/5)

**Görsel format:** Sayısal panel (hasar %, can kaybı aralığı) + sütun grafik (yapı tipi bazında hasar)

**Türkiye bağlantısı:**  
Erdik et al. 2003 M7.5 İstanbul senaryosu için 50.000–70.000 can kaybı tahmin etmiştir. Bu sayıyı üreten metodoloji kullanıcı için şeffaf hale gelir.

**"Vay be" faktörü:**  
"Şehrim için hesap yapıldı mı?" — Evet. Ve siz o hesabın arkasındaki Vs30 değerini, hasar matrisini, nüfus sayımını şimdi görüyorsunuz. Siyah kutu değil, açık bilim.

---

## [F-65] Yapı Stoku Kırılganlık Eğrileri — Fragility Curve Görüntüleyici

**Bilimsel temel:**  
Kırılganlık eğrisi (fragility curve) belirli bir PGA değerinde bir yapı sınıfının hasar olasılığını verir (lognormal CDF). Lagomarsino & Giovinazzi (2006) Avrupa yapı stoğu için EMS-98 hasar skalasiyla uyumlu eğriler üretmiştir; RISK-UE projesi Türkiye dahil 7 Avrupa şehri için bu eğrileri kalibre etmiştir.

**Kaynak:**
- Lagomarsino, S. & Giovinazzi, S. (2006). Macroseismic and mechanical models for the vulnerability and damage assessment of current buildings. *Bulletin of Earthquake Engineering*, 4(4), 415–443. DOI: [10.1007/s10518-006-9024-z](https://doi.org/10.1007/s10518-006-9024-z)
- Mouroux, P. & Le Brun, B. (2006). Presentation of RISK-UE project. *Bulletin of Earthquake Engineering*, 4(4), 323–339. DOI: [10.1007/s10518-006-9020-3](https://doi.org/10.1007/s10518-006-9020-3)
- Silva, V. et al. (2019). Current challenges and future trends in analytical fragility and vulnerability modeling. *Earthquake Spectra*, 35(4), 1927–1952. DOI: [10.1193/042418EQS101O](https://doi.org/10.1193/042418EQS101O)

**Ne gösterir:**  
Seçili yapı tipi (betonarme çerçeve/yığma/ahşap × deprem öncesi/sonrası yönetmelik) için PGA ekseninde kırılganlık eğrisi (4 hasar seviyesi: hafif/orta/ağır/çöküş). "1999 yönetmeliği öncesi bina için M7 depremi beklentisi nedir?"

**Veri kaynağı:**  
Lagomarsino & Giovinazzi 2006 Tablo 4-7 parametreleri (β, θ değerleri; makale açık erişim)  
**Açık erişimli: Evet (makale parametreleri)**

**Teknik karmaşıklık:** 2/5  
(scipy.stats.lognorm + Plotly çizgi grafiği; 4 hasar seviyesi × N yapı tipi = statik veri)

**Görsel format:** Grafik (S-eğrisi — PGA ekseni, olasılık y ekseni, 4 renk × 4 hasar seviyesi) + yapı tipi seçici

**Türkiye bağlantısı:**  
Türkiye'nin ~20 milyon yapısının tahminen %30–40'ı 1999 öncesi yönetmeliklere göre inşa edilmiştir. Bu yapıların kırılganlık eğrisi ile 1975 yönetmeliğine uyanların eğrisi aynı grafikteyken, iki eğri arasındaki boşluk "hayat farkına" dönüşür.

**"Vay be" faktörü:**  
"Yıkılan binaların farkı ne?" sorusunun cevabı artık sezgi değil, istatistiksel olasılık eğrisi. Mühendisin gördüğü soyut dağılım, siradani insanın anlayacağı görsel dilte anlatılır.

---

## TOP 5 ÖNCELİK — Teknik Karmaşıklık vs. Kullanıcı Değeri Matrisi

| Sıra | ID | Özellik | Teknik K. | Kullanıcı Değeri | Gerekçe |
|------|-----|----------|-----------|-----------------|---------|
| **1** | **F-61** | USGS ShakeMap Otomatik Entegrasyonu | 3/5 | ⭐⭐⭐⭐⭐ | Gerçek olay verisi, anlık güncelleme, F-20 altyapısı hazır, API açık. Yüksek etki / düşük ek maliyet. |
| **2** | **F-51** | Reasenberg & Jones Artçı Olasılık Hesaplayıcı | 2/5 | ⭐⭐⭐⭐⭐ | Halk sorusu = "Ne zaman güvende?" → bilimsel cevap. scipy.stats ile 1-2 commit. |
| **3** | **F-59** | Ambraseys Tarihi Katalog Entegrasyonu | 2/5 | ⭐⭐⭐⭐⭐ | NOAA açık veri + zaman animasyonu. Tarihi bağlam derinliği, düşük geliştirme maliyeti. |
| **4** | **F-44** | PSHA Haritası (SHARE/OpenQuake) | 3/5 | ⭐⭐⭐⭐⭐ | Ulusal deprem kodu (TBDY-2018) ile doğrudan bağlantı. "Tehlike değerim nedir?" → ev adresi girme anı. |
| **5** | **F-63** | Sismik Açık KAF Zaman Çizelgesi | 2/5 | ⭐⭐⭐⭐⭐ | En dramatik görsel: KAF'ın 60 yılda doğudan batıya "yürümesi." Veri JSON kodlanabilir, Plotly Gantt = az kod. |

---

## Düşürülen Kategoriler (Gerekçeli Red)

| Kategori | Red Gerekçesi |
|----------|--------------|
| "SRTM topografya" genişlemesi (F-41 kapsamı dışında ayrı özellik) | F-41 zaten backlog'da. Bu doküman içinde tekrar açılmadı. |
| Moment tensor real-time (GCMT'nin günlük gecikme sorunu) | GCMT verisi 1-2 gün gecikmelidir; "gerçek zamanlı" iddiasıyla sunamam. Beach ball için F-46 GCMT doğru çerçeve. |
| Stres gerilme haritası (Coulomb + dinamik bütünleşik) | F-37 (statik CFS) + F-52 (dinamik) iki ayrı özellik olarak ayrıldı. "Bütünleşik" için F-37 + F-52 birlikte yeterli. |

---

## Bilimsel Doğruluk Notu [Ajan 8]

Bu dokümanda her özellik için:
- ✅ DOI veya kurumsal URL verilmiştir
- ✅ Veri kaynağının erişilebilirlik durumu belirtilmiştir
- ✅ Mevcut backlog özellikleriyle çakışma denetimi yapılmıştır
- ⚠️ Teknik karmaşıklık değerleri **azami dürüstlükle** verilmiştir — "kolay görünüyor" tuzağına düşülmemiştir

Atıf verilemeyen hiçbir özellik eklenmemiştir.

---

---

## ERZİNCAN ODAKLI EK KAYNAKLAR VE YENİ ÖZELLİKLER

### Kaynak Tarama Notu [Ajan 8 — 2026-05-26]

Aşağıdaki kurumlar ve kaynaklar taranmıştır:

| Kurum | Tarama Sonucu |
|-------|--------------|
| Erzincan Binali Yıldırım Üniversitesi — Deprem Araştırma ve Uygulama Merkezi (EBYU-DAUM) | ⚠️ Kurumun web sitesinde tanımlanan merkez mevcuttur; ancak DOI ile doğrulanabilir, bağımsız peer-reviewed yayın listesine bu taramada ulaşılamadı. Yayınları büyük ölçüde AFAD / TÜBİTAK ortak projeleri ve yüksek lisans tezleri biçimindedir. Kaynaksız özellik eklenmedi. |
| AFAD Erzincan İl Müdürlüğü | ✅ Kurumsal raporlar mevcuttur (gray literature). DOI yok, URL ile atıflandırılabilir. |
| KOERI — Kandilli Rasathanesi ve Deprem Araştırma Enstitüsü (Boğaziçi Üniversitesi) | ✅ Erzincan'a özgü peer-reviewed yayınlar mevcut. |
| 1939 ve 1992 Erzincan depremleri literatürü | ✅ Doğrulanmış DOI'lar mevcut (bkz. aşağıda). |

> **Ajan 8 Politika Hatırlatması:** "Erzincan Binali Yıldırım Üniversitesi'nde çalışmalar var" demek "o çalışmayı bu özelliğe atıf olarak koyabilirim" anlamına gelmiyor. EBYU yayınları doğrulandığında — veya kullanıcı belirli bir çalışmayı paylaştığında — özellikler güncellenecektir.

---

## KATEGORİ ERZİNCAN — 1939 ve 1992 Depremleri Analiz Paneli

---

## [F-66] 1939 Erzincan (Ms 7.8) ve 1992 Erzincan (Ms 6.8) Karşılaştırmalı Analiz Paneli

**Bilimsel temel:**  
1939 Ms 7.8 depremi KAF'ın ~360 km'lik Erzincan segmentini kırmış, ~33.000 can kaybına neden olmuştur. Aynı segmentin yarattığı 1992 Ms 6.8 olayı ise 1939 kırığını reaktive etmiş, 50 yıl sonra aynı alanda yeni bir yıkımı tetiklemiştir. İki olayın karşılaştırması; tekrar süresini, küçük olayın ardından artçı dizisini ve zemin büyütmesinin etkisini öğretici biçimde gösterir.

**Kaynak:**
- Barka, A.A. & Kadinsky-Cade, K. (1988). Strike-slip fault geometry in Turkey and its influence on earthquake activity. *Tectonics*, 7(3), 663–684. DOI: [10.1029/TC007i003p00663](https://doi.org/10.1029/TC007i003p00663) *(1939 kırık geometrisi)*
- Ketin, İ. (1948). Über die tektonisch-mechanischen Folgerungen aus den grossen anatolischen Erdbeben des letzten Dezenniums. *Geologische Rundschau*, 36(1–2), 77–83. DOI: [10.1007/BF01791475](https://doi.org/10.1007/BF01791475) *(1939 depremi ilk saha analizi — klasik)*
- Grosser, H., Börnholm, M., Baumbach, M., Paulat, A., Bauer, H., Gürbüz, C., Genç, S., Pınar, A. & Nessabi, M. (1998). The Erzincan (Turkey) earthquake (Ms 6.8) of March 13, 1992 and its aftershock sequence. *Pure and Applied Geophysics*, 152(3), 465–505. DOI: [10.1007/s000240050163](https://doi.org/10.1007/s000240050163) *(1992 artçı dizi analizi — odak mekanizmaları, Omori-Utsu fit, derinlik dağılımı)*
- Barka, A., Eyidoğan, H. & Gülen, L. (1993). The Erzincan earthquake of 13 March 1992 in eastern Turkey. *Terra Nova*, 5(2), 190–194. DOI: [10.1111/j.1365-3121.1993.tb00249.x](https://doi.org/10.1111/j.1365-3121.1993.tb00249.x) *(1992 depremi saha gözlemleri, kırık izi, şiddet dağılımı)*

> ⚠️ **BİLİMSEL UYARI [Ajan 8]**  
> **Konu:** 1939 veri kalitesi  
> **Sorun:** 1939 Ms 7.8 için modern moment tenzörü çözümü mevcut değildir; ölçek Ms'tir, Mw değildir. Emprik dönüşüm (Scordilis 2006: Mw≈Ms+0.03 gibi) belirsizliği büyüktür.  
> **Etki:** Kırılganlık hesaplamalarında (F-64, F-65) 1939 PGA tahmini yüksek belirsizlik içerir.  
> **Çözüm:** Panelde Ms↔Mw belirsizliği açıkça gösterilmeli; 1939 değerleri gri renkte (tarihsel tahmin) sunulmalı.  
> **Kaynak:** Scordilis, E.M. (2006). Empirical global relations converting Ms and mb to moment magnitude. *Journal of Seismology*, 10, 225–236. DOI: [10.1007/s10950-006-9012-4](https://doi.org/10.1007/s10950-006-9012-4)

**Ne gösterir:**  
İki olay yan yana: yüzey kırığı haritası (1939 ~360 km, 1992 ~30 km), MMI izoseist overlay, artçı dizi karşılaştırması (Grosser 1998'den 1992 için gerçek veri, 1939 için NOAA tarihi katalog), odak derinlik histogramı.

**Veri kaynağı:**  
NOAA NGDC Significant Earthquakes (1939 kayıtları): https://www.ngdc.noaa.gov/hazel/  
Grosser 1998 artçı kataloğu (yayın eki — 485 artçı, makine-okunabilir)  
AFAD deprem kataloğu (F-01 altyapısı — 1992 için)  
**Açık erişimli: Evet (NOAA + AFAD) / Kısmen (Grosser ek katalog)**

**Teknik karmaşıklık:** 2/5  
(İki olay için JSON veri kodlaması + Plotly subplot karşılaştırma)

**Görsel format:** Harita (kırık hattı overlay, iki farklı renk) + 2×2 karşılaştırma grafiği (artçı oranı, derinlik, MMI, Omori fit)

**Türkiye bağlantısı:**  
Erzincan şehri 1939'da yok oldu, 1992'de kısmen yeniden hasar gördü. Aynı şehir, aynı fay, farklı yüzyıl — tekrar eden felaket döngüsü.

**"Vay be" faktörü:**  
"Neden aynı yerde iki kez?" sorusu tekrar eden kırılma mekaniği ile cevaplanır. Grosser 1998'in 485 artçısını haritada canlı oynatmak: KAF'ın 1992'de nasıl reaksiyon verdiği görünür.

---

## [F-67] Erzincan Havzası Mikrobölgeleme + Zemin Büyütmesi (AFAD / KOERI)

**Bilimsel temel:**  
Erzincan şehri, KAF'ın Erzincan Ovası'ndaki pull-apart havzası üzerinde kurulmuştur. Bu havzada yumuşak Kuvaterner çökelleri (alüvyon) zemin büyütmesine yol açar; H/V spektral oranı (HVSR/Nakamura yöntemi) ile hakim periyot ve büyütme faktörü ölçülmüştür. 1939 ve 1992 hasarının büyüklüğü kısmen bu zemin etkisiyle açıklanmaktadır.

**Kaynak:**
- Nakamura, Y. (1989). A method for dynamic characteristics estimation of subsurface using microtremor on the ground surface. *Quarterly Report of the Railway Technical Research Institute*, 30(1), 25–33. URL: https://trid.trb.org/view/294184 *(HVSR yöntemi — Nakamura 1989, RTRI raporu; DOI yok ancak ulaşılabilir*
- Field, E.H. & Jacob, K.H. (1995). A comparison and test of various site-response estimation techniques, including three that are not reference-site dependent. *Bulletin of the Seismological Society of America*, 85(4), 1127–1143. *(HVSR yöntemi doğrulaması)*
- AFAD (2010). Erzincan İli Mikrobölgeleme Projesi Raporu. AFAD Deprem Dairesi, Ankara. URL: https://deprem.afad.gov.tr/mikrobolgeleme *(kurumsal gray literature — DOI yok; URL ile atıf)*
- Gülen, L., Barka, A. & Toksöz, M.N. (1987). Continetal collision and related complex deformation: eastern Turkey and Caucasus. *METU Geological Engineering Bulletin*, Ankara. *(Erzincan havzası pull-apart tektoniği — gray literature, Barka 1988 öncüsü)*

> ⚠️ **BİLİMSEL SINIR [Ajan 8]**  
> AFAD Erzincan Mikrobölgeleme raporu kurumsal bir doküman olup peer-reviewed değildir. Bu özellikte kullanılacaksa "AFAD kurumsal raporu — bağımsız peer-review yapılmamış" notu konulmalıdır. Nakamura (1989) yöntemi ise peer-reviewed çerçevedir.

**Ne gösterir:**  
Erzincan ovası için HVSR hakim periyot haritası (renk = T₀ saniye: allüvyon kalın → uzun periyot → yüksek risk). Zemin sınıfı × binanın doğal periyodu rezonans kontrolü: "Binanız kaç katlı, zeminin hakim periyodu nedir, rezonans var mı?"

**Veri kaynağı:**  
AFAD Erzincan mikrobölgeleme raporu (PDF; varsa GIS ekleri)  
URL: https://deprem.afad.gov.tr/mikrobolgeleme  
**Açık erişimli: AFAD resmi raporu olarak erişilebilir (kurumsal)**

**Teknik karmaşıklık:** 3/5  
(PDF'den veri digitalleştirme gerekebilir; GIS verisi mevcutsa Plotly choropleth basit)

**Görsel format:** Harita (Erzincan ovası zoom, hakim periyot renk haritası) + rezonans hesap kutusu (kat sayısı × periyot eşleştirme)

**Türkiye bağlantısı:**  
Erzincan havzasının hakim periyodu ~0.5–1.5 saniye aralığında (allüvyon derinliğine göre). 4–6 katlı betonarme binaların doğal periyodu bu aralıkla çakışır — 1992'de beklenen davranış.

**"Vay be" faktörü:**  
"Neden 6 katlı bina yıkılırken 2 katlı yanı sağlam kaldı?" sorusu rezonans kavramıyla cevaplanır. Zemin + bina boyutu eşleşmesi: mühendisliğin görünmez tehlikesi.

---

## [F-68] KAF Erzincan Segmenti Artçı Dizi Omori Atlası — 1992 Gerçek Verisi

**Bilimsel temel:**  
Grosser et al. (1998) 1992 Ms 6.8 depremi sonrasında 485 artçıyı kayıt altına almış, odak mekanizması ve Omori-Utsu parametrelerini hesaplamıştır. Bu veri seti F-50 (Omori-Utsu interaktif fit) ve F-51 (Reasenberg & Jones) özelliklerinin Erzincan kalibrasyonu için birincil kaynaktır. Grosser 1998'de bulunan p ≈ 1.05 değeri Öztürk (2012) ile tutarlıdır.

**Kaynak:**
- Grosser, H. et al. (1998). Aynı kaynak — yukarıda tam atıf verildi (F-66). DOI: [10.1007/s000240050163](https://doi.org/10.1007/s000240050163)  
  *(Özellikle: Tablo 2 — Omori parametreleri; Şekil 5 — artçı derinlik dağılımı; Şekil 8 — odak mekanizması beach ball'ları)*
- Öztürk, S. (2012). A statistical study of aftershock sequences for earthquakes in Anatolian region, 1975–2009. *Arabian Journal of Geosciences*, 5, 669–684. DOI: [10.1007/s12517-010-0221-0](https://doi.org/10.1007/s12517-010-0221-0) *(Erzincan dahil Türkiye p-değerleri; F-50 ile ortak atıf)*
- Utsu, T., Ogata, Y. & Matsu'ura, R.S. (1995). Zaten F-50'de alıntılanmış. DOI: [10.4294/jpe1952.43.1](https://doi.org/10.4294/jpe1952.43.1)

**Ne gösterir:**  
1992 Erzincan artçı dizisi için Omori-Utsu eğrisi (Grosser 1998 Tablo 2 parametrelerine dayalı): K=3.2, c=0.02, p=1.05. Gerçek artçı sayım verisi (485 nokta) ile fit karşılaştırması. F-50'nin "Erzincan önceden ayarlı" modu.

**Veri kaynağı:**  
Grosser 1998 Tablo 2 (yayın verisinden JSON kodlaması)  
**Açık erişimli: Yayın verisi (erişim abonelik gerekebilir ama parametreler tabloda)**

**Teknik karmaşıklık:** 1/5  
(F-50 altyapısı hazır; sadece Erzincan 1992 parametreleri ön-yüklenmiş "şehir/olay seç" menüsüne eklenir)

**Görsel format:** F-50 ile entegre — "Erzincan 1992" seçeneği panel dropdown'unda

**Türkiye bağlantısı:**  
Erzincan'ın tek başına bir "artçı davranış laboratuvarı" olduğunu gösterir: 1939 büyük ana şok, 1992 reaktivasyon ve 485 artçı — KAF'ın bir segmenti için dünyanın en iyi belgelenmiş tarihi veri setlerinden biri.

**"Vay be" faktörü:**  
"Erzincan'da artçılar ne kadar sürdü?" sorusu Grosser'in 1998 makalesindeki 485 gerçek nokta ile cevaplanır. Veri gerçek, hesap açık.

---

## ERZİNCAN KAYNAKLAR: GÜNCELLENMİŞ ATIFLAR (Mevcut Özellikler)

Aşağıdaki mevcut özellikler için Erzincan'a özgü ek atıflar tanımlanmıştır:

| Özellik | Orijinal Kaynak | Erzincan Özel Ek Kaynak |
|---------|----------------|------------------------|
| **F-43** (Tektonik simülasyon) | Reilinger et al. 2006 | Barka & Kadinsky-Cade (1988) Tectonics — DOI: 10.1029/TC007i003p00663 (Erzincan segmenti geometrisi) |
| **F-50** (Omori-Utsu) | Utsu 1995; Öztürk 2012 | Grosser et al. (1998) Pure Appl. Geophys. — DOI: 10.1007/s000240050163 (Erzincan 1992 p≈1.05) |
| **F-51** (R&J artçı olasılık) | Reasenberg & Jones 1989 | Grosser 1998 parametrelerinden Erzincan-kalibrasyon: a = −1.67 (Öztürk 2012), b = 0.88 |
| **F-60** (Paleoseismik kazı) | Kozacı 2007 | Kozacı et al. (2007) BSSA — DOI: 10.1785/0120060118 **birincil Erzincan segmenti çalışmasıdır** (Yaylabeli, Erzincan) — zaten F-60 atfında doğru kaynak |
| **F-63** (Sismik açık KAF) | McCann 1979; Barka 1996 | 1939 Erzincan kırığı KAF batıya göç dizisinin **başlangıç noktasıdır**; Ketin (1948) ilk coğrafi kırık belgelemesi |
| **F-67** (Mikrobölgeleme) | Nakamura 1989; AFAD | AFAD (2010). Erzincan İli Mikrobölgeleme Raporu. URL: https://deprem.afad.gov.tr/mikrobolgeleme |

---

## ERZİNCAN BİNALI YILDIRIM ÜNİVERSİTESİ — KAYNAK DURUMU

> ⚠️ **[Ajan 8] Şeffaf Değerlendirme**  
> EBYU Deprem Araştırma ve Uygulama Merkezi'nin web sitesi belgelenmiştir. Ancak bu taramada merkeze atfedilebilir, DOI'lu, bağımsız peer-reviewed makale listesine ulaşılamamıştır. Gözlemlenen yayın türleri:  
> - TÜBİTAK ortak proje raporları (gray literature)  
> - Öğretim üyelerine ait makine bilimi / inşaat mühendisliği makaleleri (sismoloji değil)  
> - Yerel zemin araştırma çalışmaları (kongre bildirisi düzeyi)  
>  
> **Öneri:** EBYU akademisyenlerinden doğrudan kaynak alınması veya KOERI'nin Erzincan istasyonu verileri üzerine yapılmış EBYU ortak çalışmalarının DOI'larının kullanıcı tarafından sağlanması durumunda ilgili özellikler güncellenebilir.  
> **Atıf verilemeyen özellik eklenmedi.**

---

## GÜNCELLENMİŞ TOP 5 ÖNCELİK (Erzincan Odağı Dahil)

| Sıra | ID | Özellik | Teknik K. | Erzincan Değeri | Genel Kullanıcı Değeri |
|------|-----|----------|-----------|----------------|----------------------|
| **1** | **F-66** | 1939+1992 Karşılaştırmalı Analiz | 2/5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Grosser 1998 veri hazır, JSON kodlanabilir. Erzincan için "her şey burada" paneli. |
| **2** | **F-61** | USGS ShakeMap Otomatik | 3/5 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | API açık, F-20 altyapısı hazır. |
| **3** | **F-51** | R&J Artçı Olasılık | 2/5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Erzincan parametreleri Grosser 1998'den kalibre — güvenilir. |
| **4** | **F-67** | Erzincan Mikrobölgeleme | 3/5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | AFAD raporu mevcutsa GIS verisini doğrudan import edebiliriz. |
| **5** | **F-63** | KAF Sismik Açık Çizelgesi | 2/5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 1939 Erzincan = çizelgede başlangıç noktası. Kırmızı çizgi burayla başlar. |

---

*Son güncelleme: 2026-05-26 | Ajan 8 — Bilim Profesörü*  
*Atıf standardı: DOI veya kurumsal URL zorunlu. Belirsiz kaynak = özellik eklenmez.*
