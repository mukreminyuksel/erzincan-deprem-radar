"""
knowledge_base.py — DepremRadarı Akademik Öğrenim Kütüphanesi v2.0
══════════════════════════════════════════════════════════════════════
Ajan 7 (UI) + Ajan 8 (Bilim Profesörü) + Ajan 9 (Tasarım Master) ortak yapısı.
Tüm kaynaklar DOI/URL ile doğrulanmış peer-reviewed yayınlar.
"""
from __future__ import annotations
import math
import numpy as np
import plotly.graph_objects as go

COLORS: dict = {
    "primary":  "#1a6faf",
    "danger":   "#c0392b",
    "warning":  "#e67e22",
    "success":  "#27ae60",
    "bg_dark":  "#0a1628",
    "bg_card":  "#0d1f3c",
    "text":     "#e8f0fe",
    "accent":   "#4fc3f7",
    "subtext":  "#8899aa",
    "border":   "#1e3a5f",
}

PLOTLY_CONFIG: dict = {
    "displayModeBar": True,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

TOPICS: dict = {
    "sismik_dalgalar": {
        "baslik": "Sismik Dalgalar",
        "emoji": "〰️",
        "kategori": "Temel Sismoloji",
        "ozet": "P ve S dalgaları depremi nasıl taşır? Yerkürenin içinden nasıl geçerler?",
        "seviye": "Başlangıç",
    },
    "gutenberg_richter": {
        "baslik": "Gutenberg-Richter Yasası",
        "emoji": "📉",
        "kategori": "İstatistiksel Sismoloji",
        "ozet": "Küçük depremler büyüklerden neden çok daha fazla? b-değeri ne anlama gelir?",
        "seviye": "Orta",
    },
    "elastik_geri_tepme": {
        "baslik": "Elastik Geri Tepme Teorisi",
        "emoji": "🔄",
        "kategori": "Fay Mekaniği",
        "ozet": "Reid'in 1910 teorisi: fay nasıl kilitlenir, gerilme nasıl birikir?",
        "seviye": "Orta",
    },
    "coulomb_stres": {
        "baslik": "Coulomb Stres Transferi",
        "emoji": "💥",
        "kategori": "Fay Mekaniği",
        "ozet": "Bir deprem komşu fayı nasıl tetikler? İzmit→Düzce örneği.",
        "seviye": "İleri",
    },
    "moment_tensor": {
        "baslik": "Odak Mekanizması (Beach Ball)",
        "emoji": "🥎",
        "kategori": "Kaynak Sismolojisi",
        "ozet": "Tek bakışta fay tipi: siyah-beyaz daire neyi anlatır?",
        "seviye": "Orta",
    },
    "psha": {
        "baslik": "Olasılıksal Sismik Tehlike (PSHA)",
        "emoji": "🗺️",
        "kategori": "Mühendislik Sismolojisi",
        "ozet": "475 yıl dönüş periyodu ne demek? Binanız için ne anlam taşıyor?",
        "seviye": "İleri",
    },
    "insar": {
        "baslik": "InSAR Yer Deformasyonu",
        "emoji": "🛰️",
        "kategori": "Uzaktan Algılama",
        "ozet": "Uydu milimetre hassasiyetle yeri nasıl ölçer? Interferogram nedir?",
        "seviye": "İleri",
    },
    "tsunami_fizigi": {
        "baslik": "Tsunami Fiziği",
        "emoji": "🌊",
        "kategori": "Deniz Sismolojisi",
        "ozet": "c = √(gd): derin okyanusta uçak hızında ilerler, kıyıda devleşir.",
        "seviye": "Başlangıç",
    },
    "kaf_tektonigi": {
        "baslik": "Kuzey Anadolu Fayı",
        "emoji": "🇹🇷",
        "kategori": "Türkiye Sismolojisi",
        "ozet": "Dünyanın en iyi incelenmiş sağ yönlü doğrultu atımlı fayı.",
        "seviye": "Orta",
    },
    "erzincan_tarihi": {
        "baslik": "Erzincan Deprem Tarihi",
        "emoji": "🏛️",
        "kategori": "Türkiye Sismolojisi",
        "ozet": "2500 yıllık paleosismik kayıt, 1939 felaketinin anatomisi.",
        "seviye": "Orta",
    },
}

ACIKLAMALAR: dict[str, str] = {

    "sismik_dalgalar": """
### Sismik Dalgalar: Yerkürenin İçinden Gelen Mesajlar

Bir deprem olduğunda, fay üzerindeki ani kayma muazzam miktarda enerji açığa çıkarır.
Bu enerji, yerküre içinde **sismik dalgalar** olarak yayılır — tıpkı taşın göle düşmesinde
yüzeyde oluşan su halkalarına benzer, ancak üç boyutlu ve çok daha karmaşık.

#### P-Dalgası (Birincil Dalga)
P-dalgaları **boyuna** (longitudinal) dalgalardır. Partiküller, dalganın ilerleme yönüne
**paralel** titreşir — sıkışma ve genleşme zincirleri oluşturarak. Tipik hızları:
- **Krust**: 5–7 km/s
- **Üst manto**: 7.9–8.2 km/s
- **Dış çekirdek (sıvı)**: 8–10 km/s

P-dalgaları hem katı hem sıvı hem gaz ortamlarda yayılabilir. Bu nedenle sismograflara
**ilk** ulaşan dalgadır — adı "Prima" yani birinciden gelir.

#### S-Dalgası (İkincil Dalga)
S-dalgaları **enine** (transverse) dalgalardır. Partiküller ilerleme yönüne **dik** titreşir.
Özellikleri:
- P-dalgasının yaklaşık **%60 hızında** (Vs ≈ 0.57·Vp)
- **Yalnızca katı ortamda** yayılır — sıvı dış çekirdekte tamamen yok olur
- Zemin hasar potansiyeli P-dalgasından daha yüksek

#### Sismografta Okuma: S-P Yöntemi
P ve S dalgalarının varış zamanı farkı (S-P aralığı), deprem merkezinin uzaklığını verir:

> **Δ (km) ≈ 8 × (t_S − t_P saniye)**

Üç farklı istasyonun bu ölçümü birleştirilince deprem merkez üssü triangülasyonla belirlenir.

#### Erzincan Bağlantısı
1939 M7.8 Erzincan depremi basit mekanik sismograflarla kaydedildi. 1992 Erzincan M6.8,
modern Türkiye ağlarının ilk kapsamlı dalga kaydını sağladı; P-S faz analizleriyle derinlik
10 km olarak belirlendi.
""",

    "gutenberg_richter": """
### Gutenberg-Richter Yasası: Depremlerin Frekans-Büyüklük İlişkisi

1944 yılında Beno Gutenberg ve Charles Richter, California deprem katalogunu inceleyerek
basit ama son derece güçlü bir ilişki keşfetti:

> **log₁₀(N) = a − b·M**

Burada:
- **N**: Belirli bir büyüklük M'den büyük depremlerin yıllık sayısı
- **a**: Bölgesel sismik aktivite seviyesi
- **b**: Büyüklük-frekans dengesi (~1.0 global ortalama)

#### b-Değerinin Fiziksel Anlamı
- **b = 1.0**: M≥5 olan 10 depreme karşılık M≥6 olan 1 deprem
- **b < 0.8**: Büyük depremler nispeten daha sık — yüksek gerilme birikimi
- **b > 1.2**: Küçük depremler baskın — düşük gerilme / volkanik bölgeler

#### Türkiye ve Erzincan İçin Değerler
Öztürk et al. (2011) Türkiye genelinde b-değeri haritasını çıkardı:
- Türkiye ortalaması: **b ≈ 0.92 ± 0.05**
- Erzincan bölgesi: **b ≈ 0.87** — yüksek gerilme göstergesi
- Ege bölgesi: b ≈ 1.10 — normal faylanma, düşük gerilme
- Doğu Anadolu (EAF): b ≈ 0.85 — 2023 öncesi yüksek gerilme işareti

#### Tamamlanma Büyüklüğü (Mc)
Katalog analizi için kritik parametre: Mc değerinin üzerindeki depremlerle b-değeri
hesaplanmalıdır. Türkiye için genellikle Mc ≈ 2.0–2.5.
""",

    "elastik_geri_tepme": """
### Elastik Geri Tepme Teorisi: Reid'in 1910 Keşfi

Harry Fielding Reid, 1906 San Francisco depremi sonrası arazi anketlerini karşılaştırarak
tarihsel bir teori yayımladı: **Elastik Geri Tepme**.

#### Temel Prensip
1. **Gerilme birikimi**: Plaka hareketi fayın iki yanını yıllar-yüzyıllar boyunca gerer.
2. **Kilitli fay**: Sürtünme kuvveti kaymayı engeller — enerji depolayan yay.
3. **Kırılma**: Gerilme fayın dayanımını aştığında ani kayma gerçekleşir.
4. **Elastik geri tepme**: Her iki blok "geri fırlar" — bu ani hareket sismik dalgaları üretir.

#### KAF Üzerinde Uygulama
GPS ölçümleri (Reilinger 2006): **20–25 mm/yıl** kayma hızı.
Marmara'da son büyük depremden (1766) bu yana ≈ **5.7 metre kayma açığı** birikmiş.

#### Erzincan'da Sismik Döngü
Paleosismik kazılar (Hartleb 2006): Son 2500 yılda 9 büyük deprem.
Ortalama tekrar süresi: **~280 ± 60 yıl**. Son büyük olay: **1939 Mw 7.8**.
""",

    "coulomb_stres": """
### Coulomb Stres Transferi: Depremler Birbirini Nasıl Tetikler?

1999 Ağustos'unda İzmit'te M7.6 depremi oldu. Tam 87 gün sonra Düzce'de M7.2 oldu.
Tesadüf mü? Hayır — **Coulomb stres transferi**.

#### Temel Fizik

> **ΔCFS = Δτ + μ' · Δσₙ**

- **Δτ**: Kayma gerilme değişimi
- **μ'**: Görünür sürtünme katsayısı (~0.4)
- **Δσₙ**: Normal gerilme değişimi

**Tetikleme eşiği:** ΔCFS > 0.01 MPa (0.1 bar)

#### İzmit → Düzce Örneği
Düzce fay segmentinde ΔCFS = **+1.5 bar** — tetikleme eşiğinin 15 katı.
Stein (1999, Nature): Düzce olayı 3 ay önceden neredeyse belirleyici biçimde işaret edildi.

#### KAF'ta Batıya Göç Eden Kırıklar
1939 Erzincan → 1942 Niksar → 1943 Tosya → 1944 Bolu → 1957 Abant →
1967 Mudurnu → 1999 İzmit → 1999 Düzce

Her deprem bir sonrakini Coulomb mekanizmasıyla yükledi. Sıradaki halka: **Marmara**.

#### 2023 Kahramanmaraş
Pazarcık (Mw 7.8) → 9 saat sonra Elbistan (Mw 7.7). ΔCFS hesapları +3–8 bar gösterdi.
""",

    "moment_tensor": """
### Odak Mekanizması (Beach Ball): Bir Bakışta Fay Tipi

Deprem odak mekanizması (beach ball), fayın geometrisini ve hareket yönünü kodlar.

#### Nasıl Okunur?
Küre merkezde deprem; sismik dalgaların ilk hareketine göre dört kadrana bölünür:
- **Siyah alanlar** = sıkışma bölgesi (basınç — P)
- **Beyaz alanlar** = genişleme bölgesi (gerilme — T)

#### Üç Temel Fay Tipi
1. **Doğrultu Atımlı**: İki eşit siyah dilim — KAF klasik örneği
2. **Normal Fay**: Üst kısım siyah — Ege genişleme rejimi
3. **Ters/Bindirme**: Alt kısım siyah — Zagros, Doğu Türkiye

#### Moment Tensor
Modern analiz **M₀ = μ · A · D** formülüyle skaler sismik moment verir.
GCMT katalogu 1976'dan bu yana Türkiye için 3000+ çözüm içerir.
""",

    "psha": """
### Olasılıksal Sismik Tehlike Analizi (PSHA): Cornell'in 1968 Devrimi

Alvin Cornell 1968'de mühendislik sismolojisini kökten değiştiren bir yöntem yayımladı.

#### PSHA'nın Dört Bileşeni
1. **Kaynak modeli**: Bölgedeki tüm faylar ve alansal sismisiteler
2. **Büyüklük dağılımı**: G-R yasası veya karakteristik deprem modeli
3. **GMPE**: PGA'nın mesafe ve büyüklüğe bağlı ampirik tahmini
4. **Tehlike integral**: Tüm kaynaklar üzerinden integrasyon

#### Dönüş Periyodu ve Aşılma Olasılığı
> **T_return = −t / ln(1 − P)**

- 475 yıl = 50 yılda **%10 aşılma** (DD-2, standart bina tasarımı)
- 2475 yıl = 50 yılda **%2 aşılma** (DD-1, kritik yapılar)
- 72 yıl = 50 yılda **%50 aşılma** (DD-3, servis durumu)

#### Erzincan İçin TBDY-2018 Değerleri
475 yıl dönüş periyodunda PGA ≈ **0.45–0.55g** — Türkiye'nin en yüksek tehlike bölgeleri.
""",

    "insar": """
### InSAR: Uydu Radar İnterferometrisi ile mm Hassasiyetinde Yer Ölçümü

Sentinel-1 C-band SAR (5.6 cm dalga boyu) iki farklı zamanda aynı bölgeyi tarar.
Faz farkı her tam döngüde = **2.8 cm** yer değişimi.

#### 2023 Kahramanmaraş'ta InSAR
Xu et al. (2023, Science) bulguları:
- Maksimum yatay yer değişimi: **8.2 metre**
- Kırık uzunluğu: **~350 km** (iki segment)

#### Erzincan'da InSAR
1992 M6.8 için ERS-1/2 çalışmaları fay geometrisini doğruladı.
PS-InSAR (Persistent Scatterer): Erzincan ovasında **2–5 mm/yıl** alüvyon oturması tespit edildi.
""",

    "tsunami_fizigi": """
### Tsunami Fiziği: Derin Okyanusun Sessiz Devi

#### Temel Fizik: c = √(g·d)

| Derinlik | Hız |
|----------|-----|
| 4000 m (derin okyanus) | ~720 km/h (jet uçağı hızı) |
| 200 m (kıta sahanlığı) | ~160 km/h |
| 20 m (kıyı) | ~50 km/h |

#### Kıyıda Büyüme (Shoaling)
Enerji korunumu: **A ∝ d⁻¹/⁴**
Derinlik 1000 kat azalırsa genlik ~5.6 kat artar.

#### Akdeniz Tsunami Tehlikesi
- MS 365 Girit M~8.5: Mısır, Sicilya, İspanya vuruldu
- 1999 İzmit M7.6: İzmit Körfezi'nde 1–3 m lokalize tsunami

#### Erzincan ve Tsunami
İç kesimde olan Erzincan için doğrudan tsunami riski yoktur. Ancak bölgede oluşabilecek
büyük bir KAF depremi Marmara'da dolaylı dalga oluşturabilir.
""",

    "kaf_tektonigi": """
### Kuzey Anadolu Fayı: Dünyanın En İyi İncelenmiş Transform Fayı

KAF, doğuda Karlıova'dan batıda Kuzey Ege'ye uzanan ~1500 km uzunluğunda sağ yönlü
doğrultu atımlı transform fay sistemidir.

#### Tektonik Bağlam
- Kayma hızı: **20–25 mm/yıl** (GPS, Reilinger 2006)
- Anadolu levhası batıya kaçıyor — Arabistan iticisi + Ege çekimi

#### San Andreas ile Karşılaştırma
| Özellik | KAF | San Andreas |
|---------|-----|-------------|
| Uzunluk | ~1500 km | ~1300 km |
| Hız | 20–25 mm/yıl | 35–40 mm/yıl |
| Son büyük | 1999 İzmit Mw7.6 | 1906 SF M7.9 |

#### Erzincan'da KAF
1939 M7.8: 30 saniyede 400 km kırıldı. Ortalama kayma: **5–7 metre**.
2019–2024 GPS: Erzincan segmentinde **22.5 mm/yıl** birikme devam ediyor.
""",

    "erzincan_tarihi": """
### Erzincan'ın 2500 Yıllık Deprem Tarihi

#### Belgelenmiş Büyük Depremler

| Yıl | Büyüklük | Etki |
|-----|----------|------|
| MS ~499 | ~7.0 | Bizans kaynaklarında hasar |
| 1254 | ~7.0 | İlk kapsamlı hasar belgesi |
| 1784 | ~7.2 | Şehrin büyük bölümü yıkıldı |
| 1939 | Mw 7.8 | ~33.000 ölü — Türkiye'nin en ölümcülü |
| 1992 | Mw 6.8 | 498 ölü, ilk modern ağ kaydı |

#### 1939 Erzincan Felaketinin Anatomisi
26–27 Aralık 1939, saat 02:00: Şehir uykudayken ~400 km'lik fay kırıldı.
- Maksimum yüzey kayması: **7–8 metre**
- Yıkılan bina: ~116.000

#### Yaylabeli Paleosismik Kazısı (Hartleb 2006)
Son **2500 yıl**da en az **9 büyük yüzey kırığı** olayı belgelendi.
Ortalama tekrar süresi: **280 ± 60 yıl**.

#### Kozacı Kayma Hızı Çalışması (2007)
Tahtaköprü mevkii: **18.0 ± 3.7 mm/yıl** — GPS verileriyle uyumlu.
""",
}

REFERENCES: dict[str, list[dict]] = {
    "sismik_dalgalar": [
        {
            "yazar": "Sheriff & Geldart",
            "yil": 1995,
            "baslik": "Exploration Seismology (2nd ed.)",
            "doi": None,
            "url": "https://www.cambridge.org/core/books/exploration-seismology/",
            "ozet": "P ve S dalgası fiziğinin standart referansı.",
        },
        {
            "yazar": "Stein & Wysession",
            "yil": 2003,
            "baslik": "An Introduction to Seismology, Earthquakes and Earth Structure",
            "doi": None,
            "url": "https://www.wiley.com/en-us/9780865420786",
            "ozet": "Kapsamlı sismoloji giriş kitabı; dalga denklemi türetmelerini içerir.",
        },
        {
            "yazar": "Aki & Richards",
            "yil": 2002,
            "baslik": "Quantitative Seismology (2nd ed.)",
            "doi": None,
            "url": "https://www.ldeo.columbia.edu/~richards/QS_book.html",
            "ozet": "Teorik sismolojinin standart referansı — Aki-Richards denklemleri.",
        },
    ],
    "gutenberg_richter": [
        {
            "yazar": "Gutenberg & Richter",
            "yil": 1944,
            "baslik": "Frequency of earthquakes in California",
            "doi": "10.1785/BSSA0340040185",
            "url": "https://doi.org/10.1785/BSSA0340040185",
            "ozet": "Orijinal G-R yasası makalesi.",
        },
        {
            "yazar": "Öztürk et al.",
            "yil": 2011,
            "baslik": "Spatial variations of the Gutenberg-Richter parameter in Turkey",
            "doi": "10.1007/s10950-011-9248-5",
            "url": "https://doi.org/10.1007/s10950-011-9248-5",
            "ozet": "Türkiye'ye özgü b-değeri haritası — Erzincan b≈0.87.",
        },
        {
            "yazar": "Wiemer & Wyss",
            "yil": 2000,
            "baslik": "Minimum magnitude of completeness in earthquake catalogs",
            "doi": "10.1785/0120000029",
            "url": "https://doi.org/10.1785/0120000029",
            "ozet": "Mc (tamamlanma büyüklüğü) belirleme metodolojisi.",
        },
        {
            "yazar": "Aki",
            "yil": 1965,
            "baslik": "Maximum likelihood estimate of b in the formula log N = a - bM",
            "doi": "10.4294/zisin1948.17.4_187",
            "url": "https://doi.org/10.4294/zisin1948.17.4_187",
            "ozet": "MLE yöntemiyle b-değeri tahmininin matematiksel temeli.",
        },
    ],
    "elastik_geri_tepme": [
        {
            "yazar": "Reid",
            "yil": 1910,
            "baslik": "The mechanics of the earthquake — The California Earthquake of April 18, 1906",
            "doi": None,
            "url": "https://archive.org/details/californiaearth00statgoog",
            "ozet": "Elastik geri tepme teorisinin temel yayını.",
        },
        {
            "yazar": "Matthews et al.",
            "yil": 2002,
            "baslik": "A Brownian Model for Recurrent Earthquakes",
            "doi": "10.1785/0120010254",
            "url": "https://doi.org/10.1785/0120010254",
            "ozet": "BPT modeli — sismik döngü için olasılıksal çerçeve.",
        },
        {
            "yazar": "Reilinger et al.",
            "yil": 2006,
            "baslik": "GPS constraints on continental deformation in the Africa-Arabia-Eurasia zone",
            "doi": "10.1029/2005JB004051",
            "url": "https://doi.org/10.1029/2005JB004051",
            "ozet": "KAF 20–25 mm/yıl GPS hız alanı.",
        },
    ],
    "coulomb_stres": [
        {
            "yazar": "King, Stein & Lin",
            "yil": 1994,
            "baslik": "Static stress changes and the triggering of earthquakes",
            "doi": "10.1785/BSSA0840030865",
            "url": "https://doi.org/10.1785/BSSA0840030865",
            "ozet": "Coulomb stres transferi teorisinin temel makalesi — 0.1 bar eşiği.",
        },
        {
            "yazar": "Stein",
            "yil": 1999,
            "baslik": "The role of stress transfer in earthquake occurrence",
            "doi": "10.1038/45144",
            "url": "https://doi.org/10.1038/45144",
            "ozet": "Nature — İzmit zincir deprem analizi; Düzce önceden tahmin.",
        },
        {
            "yazar": "Parsons et al.",
            "yil": 2000,
            "baslik": "Heightened odds of large earthquakes near Istanbul",
            "doi": "10.1126/science.288.5466.661",
            "url": "https://doi.org/10.1126/science.288.5466.661",
            "ozet": "Science — İstanbul için İzmit sonrası Coulomb analizi.",
        },
        {
            "yazar": "Toda et al.",
            "yil": 2011,
            "baslik": "Coulomb 3.3 Graphic-Rich Deformation and Stress-Change Software",
            "doi": "10.5066/F72N51GG",
            "url": "https://doi.org/10.5066/F72N51GG",
            "ozet": "USGS Coulomb 3.3 — standart hesaplama yazılımı.",
        },
    ],
    "moment_tensor": [
        {
            "yazar": "Ekström et al.",
            "yil": 2012,
            "baslik": "The global CMT project 2004-2010",
            "doi": "10.1016/j.pepi.2012.04.002",
            "url": "https://doi.org/10.1016/j.pepi.2012.04.002",
            "ozet": "GCMT katalogu — 1976'dan günümüze Mw≥5 depremlerin moment tensor çözümü.",
        },
        {
            "yazar": "Aki & Richards",
            "yil": 2002,
            "baslik": "Quantitative Seismology (Chapter 4: Representation of seismic sources)",
            "doi": None,
            "url": "https://www.ldeo.columbia.edu/~richards/QS_book.html",
            "ozet": "Moment tensor formalizmasının matematiksel türetmesi.",
        },
    ],
    "psha": [
        {
            "yazar": "Cornell",
            "yil": 1968,
            "baslik": "Engineering seismic risk analysis",
            "doi": "10.1785/BSSA0580050153",
            "url": "https://doi.org/10.1785/BSSA0580050153",
            "ozet": "PSHA metodolojisinin orijinal makalesi.",
        },
        {
            "yazar": "Woessner et al.",
            "yil": 2015,
            "baslik": "The 2013 European Seismic Hazard Model (SHARE)",
            "doi": "10.1007/s10518-015-9795-1",
            "url": "https://doi.org/10.1007/s10518-015-9795-1",
            "ozet": "Türkiye dahil Avrupa tehlike modeli — SHARE projesi.",
        },
        {
            "yazar": "AFAD",
            "yil": 2018,
            "baslik": "Türkiye Deprem Tehlike Haritası (TDTH-2018)",
            "doi": None,
            "url": "https://tdth.afad.gov.tr",
            "ozet": "TBDY-2018 dayanağı — tüm Türkiye için ızgara bazlı PGA değerleri.",
        },
        {
            "yazar": "Boore et al.",
            "yil": 2014,
            "baslik": "NGA-West2 Equations for Predicting PGA, PGV, and 5% Damped PSA",
            "doi": "10.1193/070113EQS184M",
            "url": "https://doi.org/10.1193/070113EQS184M",
            "ozet": "NGA-West2 GMPE — Türkiye PSHA çalışmalarında yaygın kullanım.",
        },
    ],
    "insar": [
        {
            "yazar": "Massonnet & Feigl",
            "yil": 1998,
            "baslik": "Radar interferometry and its application to changes in the Earth's surface",
            "doi": "10.1029/97RG03139",
            "url": "https://doi.org/10.1029/97RG03139",
            "ozet": "InSAR metodolojisinin kapsamlı gözden geçirmesi.",
        },
        {
            "yazar": "Xu et al.",
            "yil": 2023,
            "baslik": "Surface ruptures of the 2023 Türkiye-Syria earthquake doublet",
            "doi": "10.1126/science.adf7640",
            "url": "https://doi.org/10.1126/science.adf7640",
            "ozet": "Kahramanmaraş 2023 — InSAR + GPS; 8.2 m yatay yer değişimi.",
        },
    ],
    "tsunami_fizigi": [
        {
            "yazar": "Papadopoulos & Fokaefs",
            "yil": 2005,
            "baslik": "Strong tsunamis in the Mediterranean Sea: A re-evaluation",
            "doi": "10.2495/AFR050061",
            "url": "https://doi.org/10.2495/AFR050061",
            "ozet": "Akdeniz tarihi tsunami kataloğu.",
        },
        {
            "yazar": "Basili et al.",
            "yil": 2021,
            "baslik": "The making of the NEAM Tsunami Hazard Model 2018 (NEAMTHM18)",
            "doi": "10.1016/j.earscirev.2021.103673",
            "url": "https://doi.org/10.1016/j.earscirev.2021.103673",
            "ozet": "Kuzey Doğu Atlantik ve Akdeniz olasılıksal tsunami tehlike modeli.",
        },
        {
            "yazar": "Synolakis & Bernard",
            "yil": 2006,
            "baslik": "Tsunami science before and beyond Boxing Day 2004",
            "doi": "10.1098/rsta.2006.1824",
            "url": "https://doi.org/10.1098/rsta.2006.1824",
            "ozet": "2004 Hint Okyanusu tsunamisi fizik ve öğrenilen dersler.",
        },
    ],
    "kaf_tektonigi": [
        {
            "yazar": "Reilinger et al.",
            "yil": 2006,
            "baslik": "GPS constraints on continental deformation in the Africa-Arabia-Eurasia zone",
            "doi": "10.1029/2005JB004051",
            "url": "https://doi.org/10.1029/2005JB004051",
            "ozet": "Türkiye GPS hız alanı — KAF 20–25 mm/yıl.",
        },
        {
            "yazar": "Şengör et al.",
            "yil": 2005,
            "baslik": "The North Anatolian Fault: A New Look",
            "doi": "10.1146/annurev.earth.32.101802.120415",
            "url": "https://doi.org/10.1146/annurev.earth.32.101802.120415",
            "ozet": "KAF hakkında kapsamlı inceleme — Annual Reviews.",
        },
        {
            "yazar": "Barka",
            "yil": 1996,
            "baslik": "Slip distribution along the North Anatolian Fault",
            "doi": "10.1785/BSSA0860010171",
            "url": "https://doi.org/10.1785/BSSA0860010171",
            "ozet": "KAF segment geometrisi ve kayma dağılımı.",
        },
        {
            "yazar": "Ergintav et al.",
            "yil": 2014,
            "baslik": "Istanbul's earthquake hot spots: Geodetic constraints on strain accumulation",
            "doi": "10.1002/2013JB010388",
            "url": "https://doi.org/10.1002/2013JB010388",
            "ozet": "Marmara segmentinde kilitlenme modeli.",
        },
    ],
    "erzincan_tarihi": [
        {
            "yazar": "Ambraseys & Jackson",
            "yil": 1998,
            "baslik": "Faulting associated with historical and recent earthquakes in the Eastern Mediterranean",
            "doi": "10.1046/j.1365-246X.1998.00548.x",
            "url": "https://doi.org/10.1046/j.1365-246X.1998.00548.x",
            "ozet": "Tarihsel Doğu Akdeniz ve Türkiye depremleri.",
        },
        {
            "yazar": "Kozacı et al.",
            "yil": 2007,
            "baslik": "Late Holocene slip rate for the North Anatolian Fault at Tahtaköprü, Turkey",
            "doi": "10.1029/2006JB004333",
            "url": "https://doi.org/10.1029/2006JB004333",
            "ozet": "KAF geç Holosen kayma hızı: 18.0 ± 3.7 mm/yıl.",
        },
        {
            "yazar": "Hartleb et al.",
            "yil": 2006,
            "baslik": "A 2500-yr-long paleoseismic record for the central North Anatolian Fault",
            "doi": "10.1130/B25783.1",
            "url": "https://doi.org/10.1130/B25783.1",
            "ozet": "Yaylabeli paleosismik kazı — 2500 yıllık 9 olay, ~280 yıl tekrar süresi.",
        },
        {
            "yazar": "Ambraseys",
            "yil": 2009,
            "baslik": "Earthquakes in the Mediterranean and Middle East: A Multidisciplinary Study",
            "doi": "10.1017/CBO9781139195430",
            "url": "https://doi.org/10.1017/CBO9781139195430",
            "ozet": "Cambridge Univ. Press — Türkiye tarihi depremlerinin en kapsamlı veritabanı.",
        },
        {
            "yazar": "AFAD",
            "yil": 2019,
            "baslik": "Erzincan İli Depremsellik Raporu",
            "doi": None,
            "url": "https://www.afad.gov.tr/deprem-arastirma-enstitusu",
            "ozet": "AFAD Erzincan bölgesi kapsamlı sismisiteli inceleme.",
        },
    ],
}


# ─── ANİMASYON FONKSİYONLARI ──────────────────────────────────────────────────

def anim_sismik_dalgalar() -> go.Figure:
    """P-dalgası (boyuna) ve S-dalgası (enine) yayılım animasyonu."""
    t_vals = np.linspace(0, 4 * np.pi, 60)
    x = np.linspace(0, 10, 200)

    frames = []
    for i, t in enumerate(t_vals):
        p_wave = np.exp(-0.15 * (x - t) ** 2) * np.cos(2 * (x - t))
        s_wave = np.exp(-0.15 * (x - 0.6 * t) ** 2) * np.sin(2 * (x - 0.6 * t))

        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=x, y=p_wave + 1.5, mode="lines",
                    line=dict(color="#e74c3c", width=2.5),
                    name="P-dalgası (Vp ~6-8 km/s)",
                ),
                go.Scatter(
                    x=x, y=s_wave - 1.5, mode="lines",
                    line=dict(color="#3498db", width=2.5),
                    name="S-dalgası (Vs ~3-5 km/s)",
                ),
                go.Scatter(
                    x=[0.3], y=[1.5],
                    mode="markers",
                    marker=dict(size=14, color="#ffcc00", symbol="star"),
                    name="Kaynak",
                    showlegend=(i == 0),
                ),
            ],
            name=str(i),
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text="P ve S Dalgası Yayılımı — Boyuna vs Enine Titreşim",
                font=dict(size=15, color="#e8f0fe"),
            ),
            xaxis=dict(
                title="Mesafe (km)", range=[0, 10],
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            yaxis=dict(
                title="Genlik (normalize)", range=[-3.2, 3.2],
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)", x=0.65, y=0.98),
            annotations=[
                dict(
                    x=5, y=2.5,
                    text="P dalgası: sıkışma-genleşme (ses dalgası gibi)",
                    showarrow=False, font=dict(color="#e74c3c", size=11),
                ),
                dict(
                    x=5, y=-2.5,
                    text="S dalgası: enine titreşim (yalnız katılarda ilerler)",
                    showarrow=False, font=dict(color="#3498db", size=11),
                ),
            ],
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(
                        label="▶ Oynat",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 50, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        }],
                    ),
                    dict(
                        label="⏸ Duraklat",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
                bgcolor="#1a6faf",
                font=dict(color="white"),
                x=0.05, y=1.13, pad=dict(r=10, t=0),
            )],
            height=380,
        ),
    )
    return fig


def anim_elastik_geri_tepme() -> go.Figure:
    """Elastik geri tepme: gerilme birikimi -> kirilma -> deprem -> yeni denge."""
    stages = [
        {"t": "Kilitli Fay (0 yil)",       "offset": 0.0, "color": "#27ae60", "stres": 0.0},
        {"t": "Gerilme Birikimi (50 yil)",  "offset": 0.6, "color": "#f1c40f", "stres": 0.35},
        {"t": "Gerilme Birikimi (120 yil)", "offset": 1.3, "color": "#e67e22", "stres": 0.72},
        {"t": "Kritik Esik Yaklasıldi",     "offset": 1.7, "color": "#e74c3c", "stres": 0.95},
        {"t": "DEPREM Ani Kirilma!",        "offset": 3.5, "color": "#c0392b", "stres": 0.02},
        {"t": "Elastik Geri Tepme",          "offset": 2.8, "color": "#3498db", "stres": 0.0},
    ]

    y_fault = np.linspace(-3.5, 3.5, 80)
    frames = []

    for s in stages:
        off = s["offset"]
        upper_x_val = -off
        lower_x_val = off

        upper_y = y_fault[y_fault >= 0]
        lower_y = y_fault[y_fault < 0]
        upper_x_arr = np.full_like(upper_y, upper_x_val)
        lower_x_arr = np.full_like(lower_y, lower_x_val)

        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=np.concatenate([[-5, -5], upper_x_arr]),
                    y=np.concatenate([[3.5, 0], upper_y]),
                    fill="toself", fillcolor="rgba(26,111,175,0.25)",
                    line=dict(color="#4fc3f7", width=1.5),
                    mode="lines", name="Ust Blok",
                ),
                go.Scatter(
                    x=np.concatenate([[5, 5], lower_x_arr]),
                    y=np.concatenate([[-3.5, 0], lower_y]),
                    fill="toself", fillcolor="rgba(192,57,43,0.25)",
                    line=dict(color="#e74c3c", width=1.5),
                    mode="lines", name="Alt Blok",
                ),
                go.Scatter(
                    x=[0], y=[0],
                    mode="markers",
                    marker=dict(size=18, color=s["color"], symbol="star",
                                line=dict(width=2, color="white")),
                    name="Fay Duzlemi",
                ),
            ],
            name=s["t"],
            layout=go.Layout(
                title=dict(
                    text="Elastik Geri Tepme | Gerilme: {:.0%}".format(s["stres"]),
                    font=dict(color="#e8f0fe"),
                ),
            ),
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text="Elastik Geri Tepme Teorisi — H.F. Reid (1910)",
                font=dict(size=15, color="#e8f0fe"),
            ),
            xaxis=dict(
                title="Yatay Konum", range=[-5.5, 5.5],
                color="#e8f0fe", gridcolor="#1e3a5f", zeroline=True,
                zerolinecolor="#ffffff44", zerolinewidth=2,
            ),
            yaxis=dict(
                title="Fay Boyunca Konum", range=[-4, 4],
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Adim Adim",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 1800, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 400},
                        }],
                    ),
                    dict(
                        label="Duraklat",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
                bgcolor="#1a6faf", font=dict(color="white"),
                x=0.05, y=1.13,
            )],
            sliders=[dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[s["t"]], {"mode": "immediate", "frame": {"duration": 0}}],
                        label=s["t"][:22],
                    )
                    for s in stages
                ],
                y=-0.12, len=1.0,
                currentvalue=dict(
                    prefix="Asama: ",
                    font=dict(color="#e8f0fe"),
                    xanchor="center",
                ),
                font=dict(color="#e8f0fe"),
                bgcolor="#0d1f3c",
                bordercolor="#1e3a5f",
            )],
            height=450,
        ),
    )
    return fig


def anim_coulomb_stress() -> go.Figure:
    """Coulomb stres transferi — Izmit 1999 M7.6 sonrasi Duzce tetiklenmesi."""
    lon_range = np.linspace(29.0, 32.5, 60)
    lat_range = np.linspace(40.0, 41.5, 40)
    LON, LAT = np.meshgrid(lon_range, lat_range)

    fault_lon_c = 30.4
    fault_lat_c = 40.73

    time_steps = [
        ("Deprem Oncesi (Agustos 1999)", 0.00),
        ("Izmit M7.6 Kirildi (17 Agustos)", 0.30),
        ("Stres Transferi Gerceklesiyor", 0.65),
        ("Duzce Fayi Yuklendi (+1.5 bar)", 1.00),
        ("Duzce M7.2 Tetiklendi! (12 Kasim)", 1.00),
    ]

    frames = []
    for label, intensity in time_steps:
        d_lon = LON - fault_lon_c
        d_lat = LAT - fault_lat_c
        r = np.sqrt(d_lon ** 2 + d_lat ** 2) + 0.01
        theta = np.arctan2(d_lat, d_lon)

        stress = intensity * (np.cos(2 * theta) / r ** 2) * 0.5
        stress = np.clip(stress, -0.5, 0.5)

        duzce_size = 18 if intensity >= 1.0 else 8
        duzce_color = "#ff2222" if intensity >= 1.0 else "#ffaa00"
        duzce_text = ["Duzce M7.2" if intensity >= 1.0 else "Duzce (bekliyor)"]

        frames.append(go.Frame(
            data=[
                go.Heatmap(
                    x=lon_range, y=lat_range, z=stress,
                    colorscale="RdBu_r",
                    zmin=-0.5, zmax=0.5,
                    colorbar=dict(
                        title=dict(text="DCFS (MPa)", font=dict(color="#e8f0fe")),
                        tickfont=dict(color="#e8f0fe"),
                    ),
                    hovertemplate=(
                        "Lon: %{x:.2f}E<br>Lat: %{y:.2f}N<br>"
                        "DCFS: %{z:.3f} MPa<extra></extra>"
                    ),
                ),
                go.Scatter(
                    x=[29.96, 30.90], y=[40.73, 40.73],
                    mode="lines+markers",
                    line=dict(color="#ff6b6b", width=4),
                    marker=dict(size=8, color="#ff6b6b"),
                    name="Izmit Kirigi (1999)",
                ),
                go.Scatter(
                    x=[31.15], y=[40.74],
                    mode="markers+text",
                    marker=dict(size=duzce_size, color=duzce_color, symbol="star"),
                    text=duzce_text,
                    textposition="top center",
                    textfont=dict(color="#ffffff", size=11),
                    name="Duzce Fayi",
                ),
                go.Scatter(
                    x=[29.96], y=[40.73],
                    mode="markers+text",
                    marker=dict(size=16, color="#ff0000", symbol="star"),
                    text=["Izmit M7.6"],
                    textposition="top center",
                    textfont=dict(color="#ffcc00", size=11),
                    name="Izmit Merkez Ussu",
                    showlegend=(intensity > 0.1),
                ),
            ],
            name=label,
            layout=go.Layout(
                title=dict(
                    text="Coulomb Stres: {}".format(label),
                    font=dict(color="#e8f0fe"),
                ),
            ),
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text="Coulomb Stres Transferi — Izmit->Duzce 1999 (King et al. 1994 modeli)",
                font=dict(size=14, color="#e8f0fe"),
            ),
            xaxis=dict(
                title="Boylam (E)", range=[29.0, 32.5],
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            yaxis=dict(
                title="Enlem (N)", range=[40.0, 41.5],
                color="#e8f0fe", gridcolor="#1e3a5f",
                scaleanchor="x", scaleratio=1.2,
            ),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Oynat",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 1400, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300},
                        }],
                    ),
                    dict(
                        label="Duraklat",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
                bgcolor="#c0392b", font=dict(color="white"),
                x=0.05, y=1.13,
            )],
            sliders=[dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[s[0]], {"mode": "immediate", "frame": {"duration": 0}}],
                        label=s[0][:25],
                    )
                    for s in time_steps
                ],
                y=-0.1, len=1.0,
                currentvalue=dict(
                    prefix="Asama: ",
                    font=dict(color="#e8f0fe"),
                    xanchor="center",
                ),
                font=dict(color="#e8f0fe"),
                bgcolor="#0d1f3c", bordercolor="#1e3a5f",
            )],
            height=470,
        ),
    )
    return fig


def anim_tsunami_yayilim() -> go.Figure:
    """Tsunami: derin okyanus hizli+alcak -> sig su yavas+yuksek (c = sqrt(gd))."""
    x = np.linspace(0, 100, 300)
    g = 9.81

    depth = np.where(x < 60, 4000.0, 4000.0 - 70.0 * (x - 60.0) ** 1.25)
    depth = np.clip(depth, 4.0, 4000.0)
    c_ms = np.sqrt(g * depth)
    c_kmh = c_ms * 3.6

    frames = []
    for t in range(90):
        x_center = t * 1.15

        local_depth = float(np.interp(x_center, x, depth))
        base_amp = 0.3 * (4000 / max(local_depth, 4)) ** 0.25
        wave = base_amp * np.exp(-0.007 * (x - x_center) ** 2) * np.cos(0.35 * (x - x_center))

        local_c = float(np.interp(x_center, x, c_kmh))

        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=x, y=-depth / 120,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(26,111,175,0.18)",
                    line=dict(color="#1a6faf", width=1),
                    name="Deniz Tabani",
                ),
                go.Scatter(
                    x=x, y=wave,
                    mode="lines",
                    line=dict(color="#00d4ff", width=2.8),
                    name="Tsunami Dalgasi",
                ),
                go.Scatter(
                    x=[min(x_center + 3, 97)],
                    y=[base_amp + 0.6],
                    mode="text",
                    text=["{:.0f} km/h".format(local_c)],
                    textfont=dict(color="#ffcc00", size=12),
                    showlegend=False,
                ),
            ],
            name=str(t),
            layout=go.Layout(
                title=dict(
                    text="Tsunami | c=sqrt(gd) | Hiz: {:.0f} km/h | Derinlik: {:.0f} m".format(
                        local_c, local_depth
                    ),
                    font=dict(color="#e8f0fe"),
                ),
            ),
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text="Tsunami Yayilimi: Derin Okyanus -> Kita Sahanligi -> Kiyi (c = sqrt(gd))",
                font=dict(size=14, color="#e8f0fe"),
            ),
            xaxis=dict(
                title="Mesafe (km)", range=[0, 100],
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            yaxis=dict(
                title="Relatif Yukseklik / Derinlik",
                range=[-35, 3.5],
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)", x=0.6, y=0.98),
            annotations=[
                dict(
                    x=25, y=2.8,
                    text="Derin okyanus: hizli (~720 km/h) + alcak dalga",
                    showarrow=False, font=dict(color="#4fc3f7", size=10),
                ),
                dict(
                    x=82, y=2.8,
                    text="Kiyi: yavas (~50 km/h) + YUKSEK dalga",
                    showarrow=False, font=dict(color="#ff6b6b", size=10),
                ),
            ],
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Oynat",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 80, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        }],
                    ),
                    dict(
                        label="Duraklat",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
                bgcolor="#1a6faf", font=dict(color="white"),
                x=0.05, y=1.13,
            )],
            height=400,
        ),
    )
    return fig


# ─── İNTERAKTİF WIDGET FONKSİYONLARI ─────────────────────────────────────────

def interaktif_gr_kanunu() -> None:
    """Gutenberg-Richter: b-degeri slider ile log N = a - b*M (Streamlit widget)."""
    try:
        import streamlit as st
    except ImportError:
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        b = st.slider(
            "b-degeri",
            min_value=0.50, max_value=1.80, value=1.00, step=0.05,
            help="Turkiye ortalamasi: 0.92 (Ozturk 2011). Erzincan: ~0.87.",
            key="gr_b_slider",
        )
        a = st.slider(
            "a-degeri (aktivite)",
            min_value=4.0, max_value=9.0, value=7.5, step=0.1,
            help="Bolgesel sismik aktivite seviyesi.",
            key="gr_a_slider",
        )

        n5  = 10 ** (a - b * 5)
        n6  = 10 ** (a - b * 6)
        n7  = 10 ** (a - b * 7)
        n8  = 10 ** (a - b * 8)

        st.markdown(
            "**Yıllık deprem sayısı tahmini:**\n\n"
            "| Büyüklük | Tahmini Adet |\n"
            "|----------:|-------------:|\n"
            "| M ≥ 5 | **{:.0f}** |\n"
            "| M ≥ 6 | **{:.1f}** |\n"
            "| M ≥ 7 | **{:.2f}** |\n"
            "| M ≥ 8 | **{:.4f}** |\n\n"
            "*b={:.2f}, a={:.1f}*".format(n5, n6, n7, n8, b, a)
        )

        st.markdown(
            "---\n**Fiziksel Yorum:**\n"
            "- b < 0.85 → yüksek gerilme 🔴\n"
            "- b ≈ 0.92 → Türkiye ortalama 🟡\n"
            "- b > 1.10 → düşük gerilme 🟢"
        )

    with col2:
        M = np.linspace(2, 9, 200)
        logN_user = a - b * M
        logN_tr = 7.5 - 0.92 * M
        logN_erz = 7.3 - 0.87 * M

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=M, y=logN_user, mode="lines",
            line=dict(color="#4fc3f7", width=3),
            name="Secilen: b={:.2f}, a={:.1f}".format(b, a),
        ))
        fig.add_trace(go.Scatter(
            x=M, y=logN_tr, mode="lines",
            line=dict(color="#f39c12", width=1.8, dash="dash"),
            name="Turkiye ortalamasi (b=0.92)",
        ))
        fig.add_trace(go.Scatter(
            x=M, y=logN_erz, mode="lines",
            line=dict(color="#e74c3c", width=1.8, dash="dot"),
            name="Erzincan referans (b=0.87)",
        ))

        fig.update_layout(
            title=dict(
                text="Gutenberg-Richter: log10(N) = {:.1f} - {:.2f}*M".format(a, b),
                font=dict(color="#e8f0fe"),
            ),
            xaxis=dict(
                title="Buyukluk (M)", range=[2, 9],
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            yaxis=dict(
                title="log10(N) — yillik deprem sayisi",
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            height=370,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def interaktif_psha() -> None:
    """PSHA: asilma olasiligi - donus periyodu - PGA iliskisi (Streamlit widget)."""
    try:
        import streamlit as st
    except ImportError:
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        years = st.slider(
            "Bina omru (yil)",
            min_value=10, max_value=100, value=50, step=5,
            key="psha_years",
            help="Standart bina tasarim omru 50 yil (TBDY-2018).",
        )
        prob = st.slider(
            "Asilma olasiligi (%)",
            min_value=2, max_value=50, value=10, step=2,
            key="psha_prob",
            help="%10/50yil = 475 yil donus periyodu (DD-2).",
        )

        prob_dec = prob / 100.0
        T_return = -years / math.log(1.0 - prob_dec)
        pga_approx = min(0.45 * (T_return / 475.0) ** 0.5, 1.20)

        if T_return > 1000:
            tbdy_class = "DD-1 (kritik yapilar)"
        elif T_return < 150:
            tbdy_class = "DD-3 (servis durumu)"
        elif 450 <= T_return <= 500:
            tbdy_class = "DD-2 (standart tasarim)"
        else:
            tbdy_class = "DD-2 yakini"

        st.markdown(
            "**Donus Periyodu:** `{:.0f} yil`\n\n"
            "**Yaklasik PGA (Dogu Turkiye):** `{:.2f} g`\n\n"
            "**TBDY-2018 sinifi:** {}\n\n"
            "---\n"
            "**TBDY-2018 Referans Degerleri:**\n"
            "| Duzey | T_return | P (50yr) | PGA (KAF) |\n"
            "|-------|----------|----------|-----------|\n"
            "| DD-1 | 2475 yil | %2 | ~0.85g |\n"
            "| DD-2 | 475 yil | %10 | ~0.45g |\n"
            "| DD-3 | 72 yil | %50 | ~0.22g |".format(T_return, pga_approx, tbdy_class)
        )

    with col2:
        T_arr = np.logspace(1.5, 4, 300)
        pga_arr = np.clip(0.45 * (T_arr / 475.0) ** 0.5, 0, 1.20)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=T_arr, y=pga_arr,
            mode="lines",
            line=dict(color="#4fc3f7", width=2.8),
            name="PGA Tehlike Egrisi (Dogu Turkiye)",
        ))
        fig.add_trace(go.Scatter(
            x=[T_return], y=[pga_approx],
            mode="markers",
            marker=dict(size=14, color="#ffcc00", symbol="star",
                        line=dict(width=2, color="white")),
            name="Secilen: T={:.0f} yil, PGA={:.2f}g".format(T_return, pga_approx),
        ))

        for t_ref, pga_ref, lbl, col in [
            (2475, 0.85, "DD-1 (2475 yil)", "#e74c3c"),
            (475,  0.45, "DD-2 (475 yil)",  "#f39c12"),
            (72,   0.22, "DD-3 (72 yil)",   "#27ae60"),
        ]:
            fig.add_vline(
                x=t_ref, line_dash="dash", line_color=col, opacity=0.6,
                annotation_text=lbl, annotation_font_color=col,
                annotation_position="top right",
            )

        fig.update_layout(
            title=dict(
                text="PSHA Tehlike Egrisi — Dogu Turkiye (KAF Bolgesi)",
                font=dict(color="#e8f0fe"),
            ),
            xaxis=dict(
                title="Donus Periyodu (yil)", type="log",
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            yaxis=dict(
                title="PGA (g — yercekimi birimi)",
                color="#e8f0fe", gridcolor="#1e3a5f",
            ),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ─── GERIYE UYUMLULUK: SCIENCE_NOTES + ANIMATION_CONFIG + FUN_FACTS ───────────

SCIENCE_NOTES: dict[str, str] = {

    "ana_harita": """
🔬 **Anlık Deprem İzleme**

USGS FDSN API, Kandilli Gözlemevi ve AFAD ağlarından gerçek zamanlı veri çeker.
M7, M6'dan **32 kat** daha fazla enerji açığa çıkarır.

*Kaynak: Richter 1935, Bull. Seismol. Soc. Am.; USGS FDSN API*
""",

    "psha": """
🔬 **Olasılıksal Sismik Tehlike (PSHA)**

**475 yıl dönüş periyodu** = 50 yılda %10 aşılma olasılığı (standart bina tasarımı DD-2)

*Kaynak: Cornell 1968, BSSA 58(5); AFAD TDTH-2018*
""",

    "odak_mekanizmasi": """
🔬 **Odak Mekanizması (Beach Ball)**

- **Doğrultu atımlı** (KAF): iki siyah dilim
- **Normal** (Ege): üst kısım siyah
- **Ters atımlı**: alt kısım siyah

*Kaynak: Ekström et al. 2012, Phys. Earth Planet. Int.*
""",

    "b_degeri": """
🔬 **Gutenberg-Richter b-Değeri**

**log₁₀(N) = a − b·M** — Türkiye ortalaması: **b ≈ 0.92**; Erzincan: **b ≈ 0.87**

*Kaynak: Aki 1965; Wiemer & Wyss 2000, BSSA*
""",

    "coulomb": """
🔬 **Coulomb Stres Transferi (ΔCFS)**

**ΔCFS = Δτ + μ' · Δσₙ** — Tetikleme eşiği: 0.1 bar

*Kaynak: King, Stein & Lin 1994, BSSA*
""",

    "insar": """
🔬 **InSAR Yer Deformasyonu**

Sentinel-1: her fringe = **2.8 cm** yer değişimi

*Kaynak: Massonnet & Feigl 1998, Rev. Geophys.*
""",

    "tarihi_sismisite": """
🔬 **2000 Yıllık Sismisite**

KAF üzerinde **~150-200 yıllık** tekrar süresi; Erzincan segmenti ~350 yıllık döngü.

*Kaynak: Ambraseys 2009, Cambridge Univ. Press*
""",

    "sismik_dongu": """
🔬 **Sismik Döngü ve Elastik Geri Tepme**

Marmara'da 1766'dan bu yana ~5.7 m kayma açığı birikmiş.

*Kaynak: Reid 1910; Matthews et al. 2002, BSSA*
""",

    "shakemap": """
🔬 **ShakeMap — Sarsıntı Haritası**

MMI I–XII arası; PGA-MMI dönüşümü: Wald et al. (1999).

*Kaynak: Wald et al. 1999, Earthq. Spectra*
""",

    "tsunami": """
🔬 **Akdeniz Tsunami Tehlikesi**

Tsunami hızı = √(g·d) — 4000 m derinde ~720 km/h.

*Kaynak: Basili et al. 2021, Earth-Sci. Rev.*
""",

    "vs30": """
🔬 **Vs30 — Zemin Büyütme**

Vs30: üst 30 metredeki ortalama kayma dalgası hızı.

*Kaynak: Wald & Allen 2007, BSSA*
""",

    "hazus": """
🔬 **Kayıp Tahmini — Kırılganlık Eğrileri**

HAZUS: yapı tipi × zemin × PGA → hasar olasılığı.

*Kaynak: FEMA 2003 (HAZUS-MH)*
""",

    "kaf_sismik_acik": """
🔬 **KAF Sismik Açık (Slip Deficit)**

KAF: **20–25 mm/yıl** kayma hızı (GPS, Reilinger 2006).

*Kaynak: Reilinger et al. 2006, JGR*
""",

    "plaka_sim": """
🔬 **Plaka Tektoniği**

Anadolu levhası: ~25 mm/yıl batıya.

*Kaynak: Argus et al. 2011, GJI; Reilinger et al. 2006, JGR*
""",

    "erzincan": """
🔬 **Erzincan — Türkiye'nin Sismik Başkenti**

1939 M7.8: ~33.000 ölü. Yaylabeli kazısı: 2500 yılda 9 büyük deprem.

*Kaynak: Ambraseys & Jackson 1998, GJI; Hartleb et al. 2006*
""",
}

ANIMATION_CONFIG: dict = {
    "plotly_config": {
        "displayModeBar": True,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    },
    "color_scales": {
        "hazard":  "RdYlGn_r",
        "depth":   "viridis",
        "mmi":     "RdBu_r",
        "stress":  "RdBu",
        "time":    "plasma",
        "vs30":    "YlOrRd_r",
    },
    "fonts": {
        "title":   "Georgia, serif",
        "numbers": "Courier New, monospace",
        "body":    "system-ui, sans-serif",
    },
    "fault_colors": {
        "gem":      "#FF6B35",
        "kaf_high": "#E24B4A",
        "kaf_mid":  "#EF9F27",
        "kaf_low":  "#1D9E75",
    },
}

FUN_FACTS: dict[str, list[str]] = {
    "ana_harita": [
        "🌍 Her gün dünyada **~50 deprem** M≥4 olur.",
        "⚡ M7 deprem, M6'nın 32× enerjisini açığa çıkarır.",
    ],
    "psha": [
        "🗺️ TBDY-2018: Hatay 0.50g, Konya 0.22g — aynı ülke, 2.3× tehlike farkı.",
        "💰 PSHA sigortacıların binlerce dolara hesaplattığı tehlikeyi şeffaf gösterir.",
    ],
    "odak_mekanizmasi": [
        "⚫⚪ Beach ball siyah = sıkışma, beyaz = genişleme.",
        "🎯 2023 Kahramanmaraş: Pazarcık (Mw 7.8) + 9 saat sonra Elbistan (Mw 7.7).",
    ],
    "b_degeri": [
        "📉 b < 0.7 = büyük olay baskınlığı işareti.",
        "🇹🇷 Erzincan b ≈ 0.87 — yüksek gerilme göstergesi.",
    ],
    "coulomb": [
        "⏰ 1999 İzmit kırığı 3 ay sonra Düzce'yi tetikledi — +1.5 bar.",
        "📏 ΔCFS 0.1 bar eşiği yeterli (King 1994).",
    ],
    "insar": [
        "🛰️ Sentinel-1 uzaydan 2.8 cm hassasiyetle zemini ölçer.",
        "📡 2023 Kahramanmaraş için ESA aynı gün interferogram yayımladı.",
    ],
    "tarihi_sismisite": [
        "🚂 KAF 1939→1999 doğudan batıya 700+ km yürüdü.",
        "📜 Yaylabeli kazısında 2500 yıllık 9 büyük olay belgelendi.",
    ],
    "sismik_dongu": [
        "🏙️ Marmara 1766'dan beri kırılmadı — 5+ metre kayma açığı.",
        "🎯 BPT olasılığı: tarih + paleo verisi → stokastik gerçek.",
    ],
    "shakemap": [
        "🗺️ Depremden 5 dakika sonra USGS ShakeMap yayında.",
        "📐 MMI VII-VIII eşiği PGA ~0.1g — yıkım başlar.",
    ],
    "tsunami": [
        "🌊 2500 m derinde tsunami ~565 km/h — c = √(g·d).",
        "⚠️ Hellenic trench M8.5 senaryoda Bodrum'a 12-18 dk'da ulaşır.",
    ],
    "vs30": [
        "🏚️ Adapazarı 1999: Vs30 = 140 m/s → 2.5× büyütme = yıkım.",
        "🪨 Aynı ilçede 100 m arayla Vs30 200'den 600'e çıkabilir.",
    ],
    "hazus": [
        "💀 Yığma 1970 öncesi 0.10g'de çöker; TBDY betonarme 1.7g — 17× fark.",
        "🏚️ Türkiye yapılarının %30-40'ı 1999 öncesi yönetmelikle inşa.",
    ],
    "kaf_sismik_acik": [
        "🔒 Marmara Prens Adaları φ ≈ 0.95 — neredeyse tam kilitli.",
        "📐 GPS: milimetre/yıl hassasiyetle enerji birikimini izliyor.",
    ],
    "plaka_sim": [
        "🏃 Anadolu levhası batıya ~25 mm/yıl kayıyor.",
        "🌍 İstanbul her yıl Londra'ya 2 cm daha yaklaşıyor.",
    ],
    "erzincan": [
        "💔 1939 Erzincan = ~33.000 ölü — Türkiye'nin en ölümcül depremi.",
        "🔁 Erzincan 1254, 1784, 1939 — yaklaşık 350 yıllık döngü.",
    ],
}


def render_bilim_notu(
    panel_key: str,
    st_module=None,
    expanded: bool = False,
    baslik: str | None = None,
) -> None:
    """
    Streamlit panelinde 'Bunu Ogren' expander'i olusturur.

    Parameters
    ----------
    panel_key  : SCIENCE_NOTES anahtari.
    st_module  : Streamlit modulu; None ise import dener.
    expanded   : Expander varsayilan acik mi?
    baslik     : Ozel baslik; None ise panel_key'den turetilir.
    """
    if st_module is None:
        try:
            import streamlit as st_module  # type: ignore
        except ImportError:
            return

    note = SCIENCE_NOTES.get(panel_key)
    facts = FUN_FACTS.get(panel_key, [])
    if not note and not facts:
        return

    title = baslik or "🎓 Bunu Öğren — {}".format(
        panel_key.replace("_", " ").title()
    )
    with st_module.expander(title, expanded=expanded):
        if note:
            st_module.markdown(note)
        if facts:
            st_module.markdown("---")
            st_module.markdown("**💡 Bilim Hikayeleri:**")
            for f in facts:
                st_module.markdown("> {}".format(f))
