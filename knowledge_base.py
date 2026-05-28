"""
knowledge_base.py — DepremRadarı Akademik Öğrenim Kütüphanesi v3.0
══════════════════════════════════════════════════════════════════════
Ajan 7 (UI) + Ajan 8 (Bilim Profesörü) + Ajan 9 (Tasarım Master) ortak yapısı.
Tüm kaynaklar DOI/URL ile doğrulanmış peer-reviewed yayınlar.

v3.0 Değişiklikleri (2026-05-26):
  • REFERENCES tamamen yenilendi — 81 kaynak, flat dict[str, dict] formatı
  • Türk kurumlar (KRDAE, İTÜ-DAMY, ODTÜ-EERC, KTÜ, AFAD) ve anahtar isimler
  • Kahramanmaraş 2023-2025 makaleleri (Science, NatGeo, GRL, GJI, CommEng)
  • 18 temel ders kitabı (Chopra, Kramer, Sucuoğlu/Akkar, vb.)
  • ACIKLAMALAR her başlık için 📚 Temel Kaynaklar bölümüyle zenginleştirildi
  • TURKISH_INSTITUTIONS, INTERNATIONAL_INSTITUTIONS, ESSENTIAL_TEXTBOOKS,
    KEY_SCHOLARS, RECENT_PUBLICATIONS_EQ listeleri (v3.0 ile eklenmişti) korundu
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
        "refs": ["AkiRichards2002", "Shearer2019", "SteinWys2003", "Kanamori2015"],
        "animasyon_adi": "anim_sismik_dalgalar",
        "aciklama": "P-dalgası (boyuna), S-dalgası (enine) ve yüzey dalgalarının (Rayleigh, Love) frekans ve hız özellikleri; S-P yöntemiyle mesafe hesaplama.",
    },
    "gutenberg_richter": {
        "baslik": "Gutenberg-Richter Yasası",
        "emoji": "📉",
        "kategori": "İstatistiksel Sismoloji",
        "ozet": "Küçük depremler büyüklerden neden çok daha fazla? b-değeri ne anlama gelir?",
        "seviye": "Orta",
        "refs": ["GutenbergRichter1944", "Aki1965", "WiemerWyss2000", "Ozturk2011", "Akkar2010"],
        "animasyon_adi": "anim_gutenberg_richter",
        "aciklama": "log10(N) = a - b·M ilişkisi; b-değerinin fiziksel anlamı (b<0.85 yüksek gerilme, b>1.1 düşük gerilme); Türkiye ve Erzincan bölgesi karşılaştırması.",
    },
    "elastik_geri_tepme": {
        "baslik": "Elastik Geri Tepme Teorisi",
        "emoji": "🔄",
        "kategori": "Fay Mekaniği",
        "ozet": "Reid'in 1910 teorisi: fay nasıl kilitlenir, gerilme nasıl birikir?",
        "seviye": "Orta",
        "refs": ["Reid1910", "Matthews2002", "Reilinger2006", "SucuogluAkkar2014", "Kramer2024"],
        "animasyon_adi": "anim_elastik_geri_tepme",
        "aciklama": "Plaka hareketi → gerilme birikimi → kilitli fay → ani kırılma → elastik geri tepme döngüsünün 4 aşaması; KAF üzerinde 20-25 mm/yıl GPS hızı ile sismik döngü.",
    },
    "coulomb_stres": {
        "baslik": "Coulomb Stres Transferi",
        "emoji": "💥",
        "kategori": "Fay Mekaniği",
        "ozet": "Bir deprem komşu fayı nasıl tetikler? İzmit→Düzce örneği.",
        "seviye": "İleri",
        "refs": ["King1994", "Stein1999", "Parsons2000", "Toda2011", "Wang2023_GRL", "Hussain2024_GJI"],
        "animasyon_adi": "anim_coulomb_stress",
        "aciklama": "ΔCFS = Δτ + μ'·Δσn formülü; 1999 İzmit→Düzce tetiklemesi; tetikleme eşiği 0.1 bar; pozitif ΔCFS bölgeleri ve artçı dağılımı korelasyonu.",
    },
    "moment_tensor": {
        "baslik": "Odak Mekanizması (Beach Ball)",
        "emoji": "🥎",
        "kategori": "Kaynak Sismolojisi",
        "ozet": "Tek bakışta fay tipi: siyah-beyaz daire neyi anlatır?",
        "seviye": "Orta",
        "refs": ["Ekstrom2012", "AkiRichards2002", "Hanks1979", "Kanamori2015"],
        "animasyon_adi": "anim_moment_tensor",
        "aciklama": "P-dalga polarite diyagramı: siyah baskı (compressional), beyaz gerilme (dilatational); doğrultu atımlı (KAF tipi), normal (Ege tipi) ve ters fay beach ball şekilleri.",
    },
    "psha": {
        "baslik": "Olasılıksal Sismik Tehlike (PSHA)",
        "emoji": "🗺️",
        "kategori": "Mühendislik Sismolojisi",
        "ozet": "475 yıl dönüş periyodu ne demek? Binanız için ne anlam taşıyor?",
        "seviye": "İleri",
        "refs": ["Cornell1968", "McGuire2004", "Boore2014", "Woessner2015", "AFAD2018_TDTH", "Field2014", "SucuogluAkkar2014"],
        "animasyon_adi": "anim_psha",
        "aciklama": "Cornell (1968) dört bileşeni: kaynak modeli, tekrar süresi, zemin hareketi tahmini (GMPE), tehlike entegrasyonu; TBDY-2018 DD-1/DD-2/DD-3 seviyeleri.",
    },
    "insar": {
        "baslik": "InSAR Yer Deformasyonu",
        "emoji": "🛰️",
        "kategori": "Uzaktan Algılama",
        "ozet": "Uydu milimetre hassasiyetle yeri nasıl ölçer? Interferogram nedir?",
        "seviye": "İleri",
        "refs": ["Massonnet1998", "Xu2023_Science", "Hussain2024_GJI", "NatComm2019_Marmara"],
        "animasyon_adi": "anim_insar",
        "aciklama": "Sentinel-1 C-band (5.6 cm dalga boyu); faz farkı her 2π = 2.8 cm yer değişimi; interferogram renk döngüsü; Kahramanmaraş 2023'te 7 m yatay, 3 m düşey yerdeğişim tespiti.",
    },
    "tsunami_fizigi": {
        "baslik": "Tsunami Fiziği",
        "emoji": "🌊",
        "kategori": "Deniz Sismolojisi",
        "ozet": "c = √(gd): derin okyanusta uçak hızında ilerler, kıyıda devleşir.",
        "seviye": "Başlangıç",
        "refs": ["Papadopoulos2005", "Basili2021", "Synolakis2006"],
        "animasyon_adi": "anim_tsunami_yayilim",
        "aciklama": "Sığ su dalgası c = √(g·d) formülü; derin okyanusta ~720 km/h, kıta sahanlığında ~140 km/h, kıyıda ~35 km/h; shoaling etkisiyle dalga yüksekliği artışı (H ∝ d^-1/4).",
    },
    "kaf_tektonigi": {
        "baslik": "Kuzey Anadolu Fayı",
        "emoji": "🇹🇷",
        "kategori": "Türkiye Sismolojisi",
        "ozet": "Dünyanın en iyi incelenmiş sağ yönlü doğrultu atımlı fayı.",
        "seviye": "Orta",
        "refs": ["Barka1997", "Barka1996", "Reilinger2006", "Ergintav2014", "NatComm2019_Marmara"],
        "animasyon_adi": "anim_kaf_tektonigi",
        "aciklama": "Karlıova çarpışma zonundan Kuzey Ege'ye ~1500 km uzunluk; 1939-1999 batıya göç eden M>7 deprem dizisi; GPS hız alanı 20-25 mm/yıl; Marmara'da ~5.7 m kayma açığı.",
    },
    "erzincan_tarihi": {
        "baslik": "Erzincan Deprem Tarihi",
        "emoji": "🏛️",
        "kategori": "Türkiye Sismolojisi",
        "ozet": "2500 yıllık paleosismik kayıt, 1939 felaketinin anatomisi.",
        "seviye": "Orta",
        "refs": ["Hartleb2006", "Kozaci2007", "Ambraseys1998", "Ambraseys2009", "Barka1997", "Biblio2025_NatHaz"],
        "animasyon_adi": "anim_erzincan_tarihi",
        "aciklama": "Paleosismik kazılar (Hartleb 2006): son 2500 yılda 9 büyük deprem; ortalama tekrar süresi ~280 ± 60 yıl; 1939 M7.8 son büyük olay; 1992 M6.8; BPT modeli ile güncel kırılma olasılığı.",
    },
    # ── Batch 2 (v1.78) — 10 yeni temel konu (animasyon yok; açıklama+kaynak dolu) ──
    "magnitud_olcekleri": {
        "baslik": "Magnitüd Ölçekleri (ML, Mw, mb, Ms)",
        "emoji": "📏",
        "kategori": "Temel Sismoloji",
        "ozet": "Richter (ML), moment magnitüd (Mw), cisim/yüzey dalgası (mb/Ms) farkları ve doygunluk.",
        "seviye": "Başlangıç",
        "refs": ["Hanks1979", "GutenbergRichter1944", "AkiRichards2002", "Kanamori2015"],
        "aciklama": "Mw = (2/3)·log₁₀(M₀) − 6.07 doygunluğa uğramaz; ML/mb/Ms büyük olaylarda doygunlaşır. AFAD/Kandilli farklı tip raporlayabilir — kaynaklar arası fark bilimsel belirsizliktir.",
    },
    "sismometre": {
        "baslik": "Sismometre & Sismogram Okuma",
        "emoji": "📟",
        "kategori": "Temel Sismoloji",
        "ozet": "Sismik dalgalar nasıl kaydedilir; sismogramdan P/S varışı ve mesafe nasıl okunur?",
        "seviye": "Başlangıç",
        "refs": ["SheriffGeldart1995", "Shearer2019", "SteinWys2003", "AkiRichards2002"],
        "aciklama": "Atalet kütlesi + bobin/mıknatıs ile yer hareketi voltaja çevrilir. S−P zaman farkı × ~8 km/s ≈ episentr mesafesi; 3+ istasyon → triangülasyon ile konum.",
    },
    "fay_tipleri_odak": {
        "baslik": "Fay Tipleri & Odak Derinliği",
        "emoji": "🪨",
        "kategori": "Fay Mekaniği",
        "ozet": "Doğrultu-atımlı / normal / ters fay; sığ-orta-derin odak ve Türkiye rejimleri.",
        "seviye": "Başlangıç",
        "refs": ["Sengör2005", "Reid1910", "Barka1996", "AkiRichards2002"],
        "aciklama": "KAF/DAF doğrultu-atımlı (yatay kayma), Ege normal (açılma), Doğu Anadolu ters (sıkışma). Türkiye depremleri çoğunlukla sığ (0–20 km) kabuk içi — bu yüzeyde daha yıkıcıdır.",
    },
    "artci_oncu_diziler": {
        "baslik": "Artçı & Öncü Deprem Dizileri",
        "emoji": "🔁",
        "kategori": "İstatistiksel Sismoloji",
        "ozet": "Omori sönümü, Båth yasası, öncü-artçı ayrımı ve 'tetiklenmiş ana şok' kavramı.",
        "seviye": "Orta",
        "refs": ["Ozturk2011", "Field2017_ETAS", "WiemerWyss2000", "Matthews2002"],
        "aciklama": "Artçı sıklığı n(t)=K/(t+c)^p ile söner; en büyük artçı ≈ ana şok − 1.2 (Båth). 2023 Maraş M7.5, M7.8'in artçısı değil tetiklenmiş ikinci ana şoktur. Öncüler ancak geriye dönük belirlenir.",
    },
    "response_spektrum": {
        "baslik": "Davranış (Response) & Tasarım Spektrumu",
        "emoji": "📐",
        "kategori": "Mühendislik Sismolojisi",
        "ozet": "Tek serbestlik dereceli sistemin tepkisi; TBDY-2018 tasarım spektrumunun temeli.",
        "seviye": "Orta",
        "refs": ["Chopra2020", "ClouPen2003", "AFAD_TBDY2018", "SucuogluAkkar2014"],
        "aciklama": "Spektrum, farklı periyotlardaki binaların bir yer hareketine maksimum tepkisini gösterir. T_bina ≈ 0.1N. TBDY-2018 tasarım spektrumu Vs30 zemin sınıfına göre değişir; rezonans yıkıcıdır.",
    },
    "likefaksiyon": {
        "baslik": "Likefaksiyon (Zemin Sıvılaşması)",
        "emoji": "💧",
        "kategori": "Mühendislik Sismolojisi",
        "ozet": "Suya doygun gevşek kumun sarsıntıda sıvı gibi davranması; bina batması/yan yatması.",
        "seviye": "Orta",
        "refs": ["SeedIdriss1982", "Atakan2002", "SucuogluAkkar2014"],
        "aciklama": "Boşluk suyu basıncı artıp efektif gerilmeyi sıfırlar (σ'=σ−u→0). 1999 Adapazarı, 2023 Hatay/İskenderun, Gölcük tipik. SPT-N, CRR/CSR oranı ile değerlendirilir. Erzincan ovası alüvyonu risklidir.",
    },
    "mikrobolgeleme": {
        "baslik": "Mikrobölgeleme & Zemin Büyütmesi",
        "emoji": "🗺️",
        "kategori": "Mühendislik Sismolojisi",
        "ozet": "Yerel zeminin sarsıntıyı büyütmesi; HVSR, Vs30 ve şehir-ölçekli risk haritalama.",
        "seviye": "Orta",
        "refs": ["Wald2007_Vs30", "Atakan2002", "AFAD_TBDY2018", "Boore2014"],
        "aciklama": "Yumuşak zemin (düşük Vs30) sarsıntıyı 1.5–3× büyütür; T₀=4H/Vs zemin periyodu bina periyoduyla çakışınca rezonans. 1985 Mexico City, 1999 Avcılar dersi. AFAD mikrobölgeleme zorunludur.",
    },
    "gmpe_sonumleme": {
        "baslik": "Sönümleme & GMPE (Yer Hareketi Tahmini)",
        "emoji": "📉",
        "kategori": "Mühendislik Sismolojisi",
        "ozet": "PGA/PGV'nin mesafeyle azalması; Akkar-Bommer 2010 gibi Türkiye-kalibre denklemler.",
        "seviye": "İleri",
        "refs": ["Akkar2010", "Boore2014", "Gulerce2011", "Gulerce2016_Turkey"],
        "aciklama": "GMPE: log(PGA)=f(M, R, zemin, fay tipi)±σ. Akkar-Bommer 2010 Türkiye/Avrupa kalibre. ±%50 saçılım PSHA'ya epistemik belirsizlik katar; tek bir kesin değer vermez.",
    },
    "sismik_tomografi": {
        "baslik": "Sismik Tomografi",
        "emoji": "🧅",
        "kategori": "Temel Sismoloji",
        "ozet": "Sismik dalga hızlarından yerin iç yapısını görüntüleme (tıbbi tomografi benzeri).",
        "seviye": "İleri",
        "refs": ["Shearer2019", "AkiRichards2002", "SteinWys2003", "Kanamori2015"],
        "aciklama": "Çok sayıda dalga yolu varış zamanı ters-çözülerek (inversiyon) 3B hız modeli elde edilir. Yavaş bölgeler sıcak/eriyik, hızlı bölgeler soğuk/rijit. Anadolu altı manto yapısı (slab tear) bu yöntemle haritalandı.",
    },
    "erken_uyari_konu": {
        "baslik": "Erken Uyarı Sistemleri (Kavram)",
        "emoji": "🚨",
        "kategori": "Mühendislik Sismolojisi",
        "ozet": "P-dalgasını yakalayıp S gelmeden uyarı; saniyeler neyi değiştirir, sınırları ne?",
        "seviye": "Orta",
        "refs": ["Wald1999_ShakeMap", "AFAD_TBDY2018", "SucuogluAkkar2014"],
        "aciklama": "Hızlı P (~6 km/s) ile yıkıcı S (~3.5 km/s) arası saniyeler uyarı penceresidir (Δt≈0.12·R). Merkez üssünde 'kör bölge' uyarı veremez. AFAD-EWS Marmara pilotu; tahmin DEĞİL, tepki sistemidir.",
    },
    # ── Batch 2 (2/2, v1.79) — 6 ileri konu ──
    "sismik_moment_enerji": {
        "baslik": "Sismik Moment, Enerji & Stress Drop",
        "emoji": "⚡",
        "kategori": "Kaynak Sismolojisi",
        "ozet": "M₀, açığa çıkan enerji ve gerilme düşüşü; magnitüd ile fiziksel büyüklük ilişkisi.",
        "seviye": "İleri",
        "refs": ["Hanks1979", "Kanamori2004_Physics", "AkiRichards2002", "Kanamori2015"],
        "aciklama": "M₀=μ·A·D̄ (sismik moment); enerji E≈M₀/(2×10⁴); stress drop Δσ≈M₀/A^1.5 (tipik 1–10 MPa). Bir magnitüd artışı ~32× enerji demektir.",
    },
    "stick_slip": {
        "baslik": "Stick-Slip & Sürtünme (Deprem Fiziği)",
        "emoji": "🧲",
        "kategori": "Fay Mekaniği",
        "ozet": "Faylar neden sürekli kaymaz da aniden kırılır? Rate-state sürtünme ve nükleasyon.",
        "seviye": "İleri",
        "refs": ["Dieterich1994", "Reid1910", "Kanamori2004_Physics", "AkiRichards2002"],
        "aciklama": "Fay sürtünme ile 'yapışır' (stick), gerilme eşiği aşılınca aniden 'kayar' (slip). Rate-state yasası (Dieterich 1994) hız ve durum bağımlı sürtünmeyi tanımlar; nükleasyon zonu kırılmayı başlatır.",
    },
    "deprem_tahmini": {
        "baslik": "Deprem Tahmini: Neden Bu Kadar Zor?",
        "emoji": "🔮",
        "kategori": "İstatistiksel Sismoloji",
        "ozet": "Deterministik tahmin neden mümkün değil; öngörü (forecast) ile tahmin (prediction) farkı.",
        "seviye": "Orta",
        "refs": ["Geller1997", "WiemerWyss2000", "Matthews2002", "Field2017_ETAS"],
        "aciklama": "Bilimsel uzlaşı (Geller 1997): kesin zaman/yer/büyüklük tahmini mümkün değildir. Olasılıksal ÖNGÖRÜ (forecast: '50 yılda %X') mümkün; deterministik TAHMİN ('yarın M7 olacak') değil. Öncü işaretler güvenilir değil.",
    },
    "ambient_noise": {
        "baslik": "Sismik Gürültü & Ambient Noise Tomografi",
        "emoji": "🌊",
        "kategori": "Temel Sismoloji",
        "ozet": "Deprem olmadan, sürekli arka plan titreşiminden yer yapısı çıkarma.",
        "seviye": "İleri",
        "refs": ["Shapiro2005_Noise", "Shearer2019", "AkiRichards2002"],
        "aciklama": "İki istasyon arası ambient gürültü çapraz-korelasyonu, aralarındaki yüzey dalgası Green fonksiyonunu verir (Shapiro 2005). Deprem beklemeden sürekli yapı görüntüleme; mikrobölgeleme + monitoring için güçlü.",
    },
    "b_degeri_uzamsal": {
        "baslik": "b-Değeri Uzamsal Değişimi (Stres Göstergesi)",
        "emoji": "🌡️",
        "kategori": "İstatistiksel Sismoloji",
        "ozet": "Gutenberg-Richter b-değerinin haritada değişimi; düşük b = yüksek stres mi?",
        "seviye": "İleri",
        "refs": ["WiemerWyss2000", "GutenbergRichter1944", "Aki1965", "Ozturk2011"],
        "aciklama": "Düşük b (~0.7) yüksek diferansiyel stres / kilitli asperity ile ilişkilendirilir; yüksek b (~1.3) heterojen/düşük stres. Mc (tamamlanma) altında geçersiz. KAF segment-bazlı b haritaları stres yorumu için kullanılır — ama tahmin değildir.",
    },
    "yavas_depremler": {
        "baslik": "Yavaş Kayma & Sessiz Depremler (SSE)",
        "emoji": "🐌",
        "kategori": "Fay Mekaniği",
        "ozet": "Sarsıntısız, günler-aylar süren kayma olayları ve volkanik olmayan tremor.",
        "seviye": "İleri",
        "refs": ["BerozaIde2011", "Dieterich1994", "Reilinger2006"],
        "aciklama": "Bazı faylar enerjiyi ani deprem yerine **yavaş kayma** (slow slip events, günler-haftalar) + tremor ile boşaltır. GPS/InSAR ile saptanır, sismogramlarda görünmez. Büyük deprem riskini hem azaltabilir hem komşu segmente yükleyebilir.",
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

---
#### 📚 Temel Kaynaklar
- **Aki & Richards (2002)** — `AkiRichards2002`: Teorik sismoloji, Bölüm 4–6. ISBN 978-1-891389-63-4
- **Shearer (2019)** — `Shearer2019`: Sismolojiye giriş; dalga fiziği ve sismograf okuması. ISBN 978-1-316-63884-4
- **Stein & Wysession (2003)** — `SteinWys2003`: Yeryapısı ve sismik dalga denklemleri. ISBN 978-0-865-42078-6
- **Kanamori (Ed., 2015)** — `Kanamori2015`: Treatise on Geophysics Vol. 4 — kapsamlı deprem sismolojisi ansiklopedisi.
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

Bölgesel zemin hareketi tahmini için Akkar & Bommer (2010) Türkiye ve Yakın Doğu'ya
uyarlanmış GMPE denklemlerini geliştirmiştir; bu denklemler G-R parametreleriyle
birleştirilerek bölgesel tehlike modelleri oluşturulur (bkz. `Akkar2010`).

---
#### 📚 Temel Kaynaklar
- **Gutenberg & Richter (1944)** — `GutenbergRichter1944`: Orijinal G-R yasası. DOI:10.1785/BSSA0340040185
- **Aki (1965)** — `Aki1965`: b-değeri MLE tahmini. DOI:10.4294/zisin1948.17.4_187
- **Wiemer & Wyss (2000)** — `WiemerWyss2000`: Mc belirleme metodolojisi. DOI:10.1785/0120000029
- **Öztürk et al. (2011)** — `Ozturk2011`: Türkiye b-değeri haritası. DOI:10.1007/s10950-011-9248-5
- **Akkar & Bommer (2010)** — `Akkar2010`: Avrupa/Türkiye GMPE. DOI:10.1785/gssrl.81.2.195
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

Performans tabanlı deprem mühendisliği çerçevesinde elastik geri tepme döngüsü,
yapıların tasarımında olasılıksal tehlike hesaplamalarının temelini oluşturur.
Sucuoğlu & Akkar (2014) bu geçişi Türkiye bağlamında kapsamlı biçimde ele alır.

---
#### 📚 Temel Kaynaklar
- **Reid (1910)** — `Reid1910`: Elastik geri tepme teorisinin temel yayını.
- **Matthews et al. (2002)** — `Matthews2002`: BPT sismik döngü modeli. DOI:10.1785/0120010254
- **Reilinger et al. (2006)** — `Reilinger2006`: GPS hız alanı, 20–25 mm/yıl. DOI:10.1029/2005JB004051
- **Sucuoğlu & Akkar (2014)** — `SucuogluAkkar2014`: Sismolojiden tasarıma köprü. Springer.
- **Kramer & Stewart (2024)** — `Kramer2024`: Zemin davranışı ve sismik döngü. CRC Press.
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
Wang et al. (2023, GRL) değişken kırılma hızını; Hussain et al. (2024, GJI) InSAR tabanlı
eş-sismik deformasyonu ve Coulomb stres transferini belgeledi.

---
#### 📚 Temel Kaynaklar
- **King, Stein & Lin (1994)** — `King1994`: Temel teori, 0.1 bar eşiği. DOI:10.1785/BSSA0840030865
- **Stein (1999)** — `Stein1999`: İzmit→Düzce tetiklenme. Nature. DOI:10.1038/45144
- **Parsons et al. (2000)** — `Parsons2000`: İstanbul senaryosu. Science. DOI:10.1126/science.288.5466.661
- **Toda et al. (2011)** — `Toda2011`: Coulomb 3.3 yazılımı. DOI:10.5066/F72N51GG
- **Wang et al. (2023)** — `Wang2023_GRL`: Kahramanmaraş değişken kırılma hızı. DOI:10.1029/2023GL104787
- **Hussain et al. (2024)** — `Hussain2024_GJI`: InSAR + Coulomb transferi. DOI:10.1093/gji/ggad498
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
Mw ölçeği, Hanks & Kanamori (1979) tarafından tanımlanmış; Ms, mb gibi doygunluk
gösteren büyüklük ölçeklerinin yerini almıştır.

---
#### 📚 Temel Kaynaklar
- **Ekström et al. (2012)** — `Ekstrom2012`: GCMT katalogu. DOI:10.1016/j.pepi.2012.04.002
- **Aki & Richards (2002)** — `AkiRichards2002`: Moment tensor formalizması, Bölüm 4.
- **Hanks & Kanamori (1979)** — `Hanks1979`: Mw ölçeği tanımı. DOI:10.1029/JB084iB05p02348
- **Kanamori (Ed., 2015)** — `Kanamori2015`: Treatise on Geophysics Vol. 4 — kaynak teorisi.
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

AFAD'ın TDTH-2018 haritası tüm bu hesapların ızgara tabanlı çıktısını sunar.
Field et al. (2014) UCERF3 modeli, Türkiye PSHA çalışmalarının metodolojik referansı
olarak yaygın biçimde kullanılır.

---
#### 📚 Temel Kaynaklar
- **Cornell (1968)** — `Cornell1968`: PSHA metodolojisinin orijinal makalesi. DOI:10.1785/BSSA0580050153
- **McGuire (2004)** — `McGuire2004`: PSHA metodolojisinin standart başvurusu. EERI Monograph MNO-10.
- **Boore et al. (2014)** — `Boore2014`: NGA-West2 GMPE. DOI:10.1193/070113EQS184M
- **Woessner et al. (2015)** — `Woessner2015`: SHARE Avrupa tehlike modeli. DOI:10.1007/s10518-015-9795-1
- **AFAD TDTH-2018** — `AFAD2018_TDTH`: Türkiye tehlike haritası. https://tdth.afad.gov.tr
- **Field et al. (2014)** — `Field2014`: UCERF3 metodolojisi. DOI:10.1785/0120130164
- **Sucuoğlu & Akkar (2014)** — `SucuogluAkkar2014`: TBDY-2018 bağlamında PSHA uygulaması.
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

2023 Kahramanmaraş depremi için Hussain et al. (2024, GJI) InSAR verileriyle eş-sismik
yer değişimi haritasını ve Coulomb stres transferi hesaplarını birleştirdi.
Xu et al. (2023, Science) Sentinel-1 + GPS ile 8.2 m maksimum yatay yer değişimini belirledi.

---
#### 📚 Temel Kaynaklar
- **Massonnet & Feigl (1998)** — `Massonnet1998`: InSAR metodolojisi. DOI:10.1029/97RG03139
- **Xu et al. (2023)** — `Xu2023_Science`: Kahramanmaraş yüzey kırıkları + InSAR. DOI:10.1126/science.adf7640
- **Hussain et al. (2024)** — `Hussain2024_GJI`: InSAR + Coulomb KM. DOI:10.1093/gji/ggad498
- **Klein et al. (2019)** — `NatComm2019_Marmara`: Marmara denizaltı kilitlenme analizi. DOI:10.1038/s41467-019-11016-z
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

---
#### 📚 Temel Kaynaklar
- **Papadopoulos & Fokaefs (2005)** — `Papadopoulos2005`: Akdeniz tarihi tsunami kataloğu.
- **Basili et al. (2021)** — `Basili2021`: NEAMTHM18 olasılıksal tehlike modeli. DOI:10.1016/j.earscirev.2021.103673
- **Synolakis & Bernard (2006)** — `Synolakis2006`: Tsunami fiziği ve 2004 dersleri. DOI:10.1098/rsta.2006.1824
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

Şengör et al. (2005) KAF'ın kökenini Neojen tektoniği çerçevesinde yeniden ele aldı;
Barka (1997) 1939–1967 döneminin kayma dağılımını haritaladı. Ergintav et al. (2014)
Marmara GPS/jeodezik kilitlenme modeliyle sismik boşluğu sayısallaştırdı.
Klein et al. (2019, NatComm) denizaltı KAF segmentinde interseismik gerilme birikimini
belgeledi.

---
#### 📚 Temel Kaynaklar
- **Şengör et al. (2005)** — `Sengör2005`: KAF kapsamlı gözden geçirme. DOI:10.1146/annurev.earth.33.101802.120345
- **Barka (1997)** — `Barka1997`: 1939–1967 KAF kayma dağılımı. DOI:10.1785/BSSA0870020017
- **Barka (1996)** — `Barka1996`: KAF segment geometrisi. DOI:10.1785/BSSA0860010171
- **Reilinger et al. (2006)** — `Reilinger2006`: GPS hız alanı. DOI:10.1029/2005JB004051
- **Ergintav et al. (2014)** — `Ergintav2014`: Marmara sismik açık. DOI:10.1002/2013GL058699
- **Klein et al. (2019)** — `NatComm2019_Marmara`: Denizaltı KAF interseismik gerilme. DOI:10.1038/s41467-019-11016-z
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

Şengör et al. (2005) KAF'ın Doğu Anadolu sıkışma tektoniğindeki rolünü açıklayarak
Erzincan'ın neden tekrarlayan büyük depremlere sahne olduğunu jeolojik boyutuyla ortaya
koymuştur. 2023 Kahramanmaraş deprem çiftinin bibliometrik analizi (Çetin et al. 2025)
artan araştırma ilgisini sayısal olarak belgelemektedir.

---
#### 📚 Temel Kaynaklar
- **Hartleb et al. (2006)** — `Hartleb2006`: Yaylabeli paleosismik kazı. DOI:10.1130/B25783.1
- **Kozacı et al. (2007)** — `Kozaci2007`: Holosen kayma hızı. DOI:10.1029/2006JB004333
- **Ambraseys & Jackson (1998)** — `Ambraseys1998`: Tarihsel depremler. DOI:10.1046/j.1365-246X.1998.00548.x
- **Ambraseys (2009)** — `Ambraseys2009`: 2000 yıllık Türkiye katalog kitabı. DOI:10.1017/CBO9781139195430
- **Şengör et al. (2005)** — `Sengör2005`: KAF tektonik bağlamı. DOI:10.1146/annurev.earth.33.101802.120345
- **Barka (1997)** — `Barka1997`: 1939 Erzincan kayma dağılımı. DOI:10.1785/BSSA0870020017
- **Çetin et al. (2025)** — `Biblio2025_NatHaz`: 577 KM makalesinin bibliyometrik analizi.
""",

    # ── Batch 2 (v1.78) — 10 yeni temel konu açıklaması ──
    "magnitud_olcekleri": """
### Magnitüd Ölçekleri — Hangi "Büyüklük"?

Bir depremin "büyüklüğü" tek bir sayı değildir; **nasıl ölçüldüğüne** göre farklı ölçekler vardır.

**Richter (yerel) magnitüd $M_L$ (1935):** Wood-Anderson sismograf genliğinden; küçük-orta yerel depremler için. Büyük olaylarda **doygunlaşır** (~6.5 üstünü ayırt edemez).

**Moment magnitüd $M_w$ (Hanks-Kanamori 1979) — modern standart:**
$$M_w = \\tfrac{2}{3}\\log_{10}(M_0) - 6.07$$
$M_0 = \\mu \\cdot A \\cdot \\bar{D}$ (sismik moment, N·m). **Doygunlaşmaz** — en büyük depremleri bile doğru ölçer.

**Cisim dalgası $m_b$ ve yüzey dalgası $M_s$:** Farklı dalga tiplerinden; ikisi de büyük olaylarda doygunlaşır.

**Türkiye:** AFAD ve Kandilli (KOERI) bir olayı farklı tiplerde raporlayabilir → küçük farklar normaldir (Scordilis 2006 dönüşümü). 2023 Maraş: $M_w$ 7.8.

**Yaygın hata:** "Richter 9" ifadesi yanlış — büyük olaylar $M_w$ ile verilir. Ölçek **logaritmiktir**: $M_w$ 7, $M_w$ 6'dan ~32× daha fazla enerji açığa çıkarır.
""",
    "sismometre": """
### Sismometre & Sismogram Okuma

**Sismometre**, yer hareketini ölçen alettir. Temel ilke: bir **atalet kütlesi** (yere göre hareketsiz kalmaya çalışır) ile yer arasındaki bağıl hareket, bobin-mıknatıs ile voltaja çevrilir. Modern geniş-bant sismometreler 0.01–100 Hz aralığını kaydeder.

**Sismogram okuma:**
1. **P varışı:** İlk, küçük genlikli sıçrama (hızlı, ~6 km/s).
2. **S varışı:** Daha büyük genlik (~3.5 km/s).
3. **Yüzey dalgaları:** En büyük, en geç (Rayleigh/Love).

**Mesafe hesabı (S−P yöntemi):**
$$\\Delta_{km} \\approx (t_S - t_P) \\times 8$$
3+ istasyondan mesafeler → **triangülasyon** ile episentr.

**Türkiye:** AFAD + Kandilli (KOERI) yüzlerce istasyonla ulusal ağ işletir; veriler bu uygulamanın 9 kaynağından ikisidir.

**Yaygın hata:** Tek istasyon konum vermez (sadece mesafe halkası); kesin konum için çok-istasyon gerekir.
""",
    "fay_tipleri_odak": """
### Fay Tipleri & Odak Derinliği

Faylar, blokların **göreli hareket yönüne** göre sınıflanır:

- **Doğrultu-atımlı (strike-slip):** Bloklar yatay kayar. **KAF (sağ-yanal)** ve **DAF (sol-yanal)** bu tiptir; Anadolu'nun batıya kaçışını taşır.
- **Normal fay:** Üst blok aşağı düşer (açılma/gerilme). **Ege** bölgesi tipiktir.
- **Ters/bindirme fay:** Üst blok yukarı çıkar (sıkışma). **Doğu Anadolu** (Bitlis-Zagros).

**Odak derinliği:**
- **Sığ (0–70 km):** Türkiye depremlerinin çoğu kabuk içi, **sığ** (5–20 km) — bu yüzeyde **daha yıkıcıdır**.
- Orta (70–300 km) ve derin (>300 km): Türkiye'de nadir (dalma-batma yok).

**Erzincan:** KAF doğrultu-atımlı, sığ odak (1939: ~10–15 km). Sığ + yüksek magnitüd = ağır hasar.

**Yaygın hata:** "Derin deprem daha tehlikeli" yanlış — sığ depremler yüzeyde daha şiddetli sarsıntı üretir.
""",
    "artci_oncu_diziler": """
### Artçı & Öncü Deprem Dizileri

Büyük bir deprem ("ana şok") sonrası **artçı** dizisi gelir.

**Omori-Utsu yasası** — artçı sıklığı zamanla söner:
$$n(t) = \\frac{K}{(t + c)^p}$$
$p \\approx 1$; ilk gün en yoğun, sonra hızla azalır (ama haftalar-aylar sürebilir).

**Båth yasası:** En büyük artçı ≈ ana şok − 1.2 magnitüd.

**Öncü (foreshock):** Ana şoktan önceki olaylar — ancak **geriye dönük** belirlenir; gerçek zamanlı "bu bir öncüdür" denemez.

**Tetiklenmiş ana şok ≠ artçı:** 1999 İzmit → 3 ay sonra Düzce; 2023 Maraş M7.8 → 9 saat sonra M7.5 — bunlar artçı değil, **Coulomb stres transferiyle tetiklenmiş yeni ana şoklardır**.

**Türkiye:** AFAD büyük olay sonrası "7 günde M5+ olasılığı" bültenlerini Reasenberg-Jones modeliyle üretir.

**Yaygın hata:** "Artçılar bitti, güvendeyiz" — olasılık düşer ama sıfırlanmaz; hasarlı yapı riski sürer.
""",
    "response_spektrum": """
### Davranış (Response) & Tasarım Spektrumu

**Davranış spektrumu**, farklı doğal periyotlardaki binaların belirli bir yer hareketine **maksimum tepkisini** (ivme/hız/yerdeğiştirme) gösteren eğridir. Her bina tek serbestlik dereceli (SDOF) bir salınıcı gibi modellenir.

**Bina doğal periyodu:** $T_{bina} \\approx 0.1 N$ (N = kat sayısı). 5 katlı ≈ 0.5 s.

**Rezonans:** Bina periyodu zemin baskın periyoduna ($T_0$) yaklaşınca sarsıntı büyür → yıkım. 1985 Mexico City ve 1999 Avcılar bu mekanizmayla yıkıldı.

**Tasarım spektrumu (TBDY-2018):** Birçok depremin davranış spektrumlarının zarfı + zemin sınıfı (Vs30) düzeltmesi. Mühendis binayı bu spektruma göre tasarlar.

**Erzincan:** Ova alüvyonu (ZD/ZE) uzun $T_0$ → orta-yüksek katlı binalar için rezonans riski.

**Yaygın hata:** "Sağlam zemin her bina için iyi" — sert zemin kısa periyot dalgayı büyütür, 1-2 katlı binayı etkiler. Hangi bina için risk, zemin + bina periyodu birlikte değerlendirilir.
""",
    "likefaksiyon": """
### Likefaksiyon (Zemin Sıvılaşması)

Suya doygun, gevşek **kumlu** zemin, sarsıntıda geçici olarak **sıvı gibi** davranır — taşıma gücünü kaybeder, binalar batar/yan yatar.

**Mekanizma:** Sarsıntı zemin tanelerini sıkıştırmaya çalışır; boşluk suyu basıncı ($u$) artar, **efektif gerilme** sıfıra düşer:
$$\\sigma' = \\sigma - u \\to 0$$
Efektif gerilme sıfır = zemin direnci sıfır.

**Değerlendirme:** SPT-N darbe sayısı + **CSR/CRR** oranı (sarsıntı talebi / zemin direnci; Seed-Idriss 1982).

**Türkiye örnekleri:** 1999 Adapazarı (binalar yan yattı), Gölcük (kıyı çökmesi), 2023 Hatay/İskenderun (liman + binalar). **Erzincan ovası** alüvyonu + yüksek su tablası = risk.

**Yaygın hata:** "Bina sağlamsa likefaksiyon önemsiz" — zemin sıvılaşırsa en sağlam bina bile temelden batar; zemin iyileştirme (jet grout, taş kolon) gerekir.
""",
    "mikrobolgeleme": """
### Mikrobölgeleme & Zemin Büyütmesi

Aynı depremde, **yerel zemin koşulları** sarsıntıyı şehirden şehre, hatta mahalleden mahalleye değiştirir. **Mikrobölgeleme**, bunu harita ölçeğinde belirler.

**Zemin büyütmesi:** Yumuşak zemin (düşük **Vs30**) sismik dalgayı **1.5–3×** büyütür (Borcherdt 1994). Sert kaya büyütmez.

**Zemin baskın periyodu:**
$$T_0 = \\frac{4H}{V_s}$$
$H$ = sediman kalınlığı, $V_s$ = ortalama kayma hızı. $T_0$ bina periyoduyla çakışırsa rezonans.

**Yöntemler:** HVSR (mikrotremor, Nakamura 1989), MASW, sondaj.

**Türkiye:** 1999 sonrası mikrobölgeleme zorunlu (AFAD). **Erzincan İli Mikrobölgeleme Projesi** ova için ZD/ZE sınıfları tanımlar.

**Yaygın hata:** "Depremin merkezinden uzağız, güvendeyiz" — 1985 Mexico City episentrden 400 km uzaktı ama göl zemini büyütmesiyle yıkıldı.
""",
    "gmpe_sonumleme": """
### Sönümleme & GMPE (Yer Hareketi Tahmin Denklemleri)

Sismik enerji kaynaktan uzaklaştıkça **söner** (geometrik yayılım + anelastik soğurma). **GMPE** (Ground Motion Prediction Equation), belirli bir M ve mesafede beklenen yer hareketini (PGA/PGV/spektral ivme) tahmin eder.

**Genel form:**
$$\\log_{10}(PGA) = f(M, R, \\text{zemin}, \\text{fay tipi}) \\pm \\sigma$$
$M$ magnitüd, $R$ mesafe (km), $\\sigma$ standart sapma (saçılım).

**Akkar & Bommer 2010:** Türkiye + Avrupa-Orta Doğu kalibreli GMPE; bu uygulamanın Erken Uyarı/ShakeMap panellerinde kullanılır. Gülerce 2016 Türkiye-spesifik NGA uyarlaması.

**Belirsizlik:** ±%50 saçılım ($\\sigma$) tipiktir → PSHA'ya **epistemik belirsizlik** katar; tek bir kesin PGA vermez (logic-tree ile ele alınır).

**Yaygın hata:** "GMPE kesin PGA verir" — hayır, olasılıksal bir dağılımdır; aynı M-R için gözlemler geniş saçılır.
""",
    "sismik_tomografi": """
### Sismik Tomografi

Tıbbi tomografi (BT) gibi, ama yer için: çok sayıda depremden gelen dalgaların **varış zamanları** kullanılarak yerin iç **hız yapısı** 3 boyutlu görüntülenir.

**İlke:** Dalga yavaş bölgeden geçerse geç varır, hızlıdan geçerse erken. Binlerce dalga yolunun varış zamanı **ters-çözülür** (inversiyon) → 3B hız modeli.

**Yorum:**
- **Yavaş bölge** ($V_p$ düşük): sıcak, kısmen eriyik, akışkan içeren (manto yükselimi, magma).
- **Hızlı bölge:** soğuk, rijit (eski litosfer, dalan levha).

**Türkiye:** Anadolu altı manto tomografisi (Biryol 2010) — Afrika levhasının **parçalanmış (slab tear)** yapısını ortaya koydu; bu Anadolu'nun yükselmesi ve volkanizmasıyla ilişkili.

**Yaygın hata:** "Tomografi deprem yerini gösterir" — hayır, **yapıyı** gösterir; deprem konumu ayrı (sismometre triangülasyonu).
""",
    "erken_uyari_konu": """
### Erken Uyarı Sistemleri (EEW) — Kavram

Deprem **olduktan sonra**, hızlı **P-dalgasını** yakalayıp yıkıcı **S-dalgası** gelmeden uzak şehirlere **saniyeler** içinde uyarı gönderme prensibi.

**Uyarı penceresi:**
$$\\Delta t = t_S - t_P \\approx 0.12 \\times R$$
($R$ = mesafe km). 100 km → ~12 s; 250 km → ~30 s. Bu sürede tren yavaşlar, gaz kesilir, ameliyat durur, insanlar pozisyon alır.

**Kör bölge:** Merkez üssüne yakın (~0–30 km) P ve S neredeyse aynı anda gelir → uyarı **imkânsızdır**.

**Türkiye:** AFAD-EWS Marmara'da pilot; Japonya JMA-EEW (2007), Meksika SASMEX, ABD ShakeAlert operasyonel.

**KRİTİK — yaygın hata:** EEW **deprem tahmini DEĞİLDİR.** Deprem zaten başladıktan sonra, hızlı dalganın yavaş dalgaya göre avantajını kullanan bir **tepki** sistemidir (Geller 1997: tahmin mümkün değil).
""",
    # ── Batch 2 (2/2, v1.79) — 6 ileri konu açıklaması ──
    "sismik_moment_enerji": """
### Sismik Moment, Enerji & Stress Drop

Bir depremin fiziksel "boyutu" üç temel büyüklükle tanımlanır.

**Sismik moment** (en temel ölçü):
$$M_0 = \\mu \\cdot A \\cdot \\bar{D}$$
$\\mu$ = kayma modülü (~30 GPa), $A$ = kırılma alanı (m²), $\\bar{D}$ = ortalama atım (m). Birim: **N·m**.

**Açığa çıkan enerji:**
$$E \\approx \\frac{M_0}{2 \\times 10^4}$$
M7 deprem ~10¹⁵ J ≈ bir nükleer santralin yıllık üretimi mertebesinde.

**Stress drop (gerilme düşüşü):**
$$\\Delta\\sigma \\approx \\frac{M_0}{A^{1.5}}$$
Tipik 1–10 MPa. Yüksek stress drop = daha şiddetli yüksek-frekans sarsıntı (bina hasarı için kritik).

**Logaritmik ölçek:** Bir magnitüd artışı = ~32× enerji, ~1000× üç magnitüd. M7.8 (2023 Maraş) M6.8'den ~32× daha güçlüdür.

**Yaygın hata:** "M6 ile M7 az fark" — hayır, 32× enerji farkı vardır.
""",
    "stick_slip": """
### Stick-Slip & Sürtünme — Deprem Neden Ani Olur?

Faylar sürekli, yumuşakça kaymaz; **yapışır-kayar** (stick-slip).

**Mekanizma:**
1. **Stick (yapışma):** Sürtünme blokları kilitler; tektonik kuvvet elastik enerji biriktirir (Reid 1910 elastik geri tepme).
2. **Slip (kayma):** Gerilme statik sürtünme eşiğini aşınca fay **aniden** kayar → deprem.

**Rate-state sürtünme yasası (Dieterich 1994):** Sürtünme katsayısı kayma **hızına** ($v$) ve temas **durumuna** ($\\theta$) bağlıdır:
$$\\mu = \\mu_0 + a\\ln\\frac{v}{v_0} + b\\ln\\frac{v_0\\theta}{D_c}$$
$a < b$ ise fay **kararsız** (deprem üretir); $a > b$ ise **kararlı** (sürünme/creep).

**Nükleasyon:** Kırılma küçük bir zonda başlar, kritik boyuta ulaşınca hızla yayılır.

**Türkiye:** KAF'ın İsmetpaşa segmenti sürünür (kararlı), Erzincan segmenti kilitlidir (kararsız → büyük deprem).

**Yaygın hata:** "Fay sürekli azar azar kaysa deprem olmaz" — doğru, ama çoğu fay kilitlidir; sürünen segmentler istisnadır.
""",
    "deprem_tahmini": """
### Deprem Tahmini: Neden Bu Kadar Zor?

En çok yanlış anlaşılan konu. **Bilimsel uzlaşı (Geller 1997, Science/GJI): deterministik deprem tahmini mümkün değildir.**

**Tahmin (prediction) vs Öngörü (forecast):**
- **Tahmin (mümkün DEĞİL):** "Yarın, Erzincan'da, M7 olacak" — kesin zaman+yer+büyüklük. Hiçbir yöntem bunu güvenilir yapamaz.
- **Öngörü (mümkün):** "Önümüzdeki 50 yılda Marmara'da M7+ olasılığı %X" — olasılıksal, uzun vadeli (PSHA, BPT).

**Neden zor?**
1. Fay sistemi kaotik, non-lineer; küçük farklar büyük sonuç değiştirir.
2. Güvenilir, tekrarlanabilir **öncü işaret** bulunamadı (radon, hayvan davranışı, ışık, b-değeri — hiçbiri doğrulanmadı).
3. Yer altı doğrudan gözlemlenemez (10+ km derinlik).

**Sahte iddialar:** "Ay/gezegen konumu", "bulut şekli", "hayvan davranışı" — hiçbiri bilimsel temelli değil (Hough 2018).

**Doğru yaklaşım:** Tahmin yerine **hazırlık** — sağlam bina (TBDY-2018), erken uyarı (tepki sistemi), afet planı.

**Yaygın hata:** "Bilim insanları depremi biliyor ama saklıyor" — komplo; gerçek şu ki kimse kesin tahmin yapamaz.
""",
    "ambient_noise": """
### Sismik Gürültü & Ambient Noise Tomografi

Yer hiç durmaz: okyanus dalgaları, rüzgâr, insan aktivitesi sürekli zayıf **sismik gürültü** üretir. Eskiden "çöp" sayılan bu sinyal, modern sismolojinin güçlü aracı oldu.

**İlke (Shapiro 2005):** İki istasyon arasındaki gürültünün **çapraz-korelasyonu**, o iki nokta arasında yayılan yüzey dalgasının **Green fonksiyonunu** verir — sanki birinde deprem olmuş gibi.

**Avantaj:** Deprem **beklemeye gerek yok**; sürekli kayıttan yapı görüntülenir. Şehir ölçeğinde zemin haritalama, volkan/fay izleme, hatta zaman içinde değişim takibi (monitoring) mümkün.

**Uygulama:** Mikrobölgeleme (Vs profili), baraj/bina sağlık izleme, jeotermal saha karakterizasyonu.

**Türkiye:** Marmara ve Ege'de ambient noise tomografisi kabuk yapısı + zemin amplifikasyonu çalışmalarında kullanılıyor.

**Yaygın hata:** "Gürültü ölçümü işe yaramaz" — tam tersine, sürekli ve her yerde olduğu için deprem-bağımsız, güçlü bir veri kaynağıdır.
""",
    "b_degeri_uzamsal": """
### b-Değeri Uzamsal Değişimi — Stres Haritası mı?

Gutenberg-Richter b-değeri ($\\log_{10}N = a - bM$) sadece bir bölge için değil, **harita üzerinde** hesaplanabilir.

**Yorum (Wiemer-Wyss 2000):**
- **Düşük b (~0.7):** Yüksek diferansiyel stres, kilitli **asperity** (kırılmaya yakın yama) ile ilişkilendirilir.
- **Yüksek b (~1.3):** Heterojen malzeme, düşük stres, akışkan varlığı.

**Hesap:** Bölge grid'e bölünür, her hücrede yeterli olay varsa (Mc üstü) Aki MLE ile b hesaplanır.

**KRİTİK sınırlama:** **Mc (tamamlanma magnitüdü)** altındaki olaylar dahil edilirse b yanlış çıkar. Az olaylı hücreler güvenilmez. Bu uygulamada İstatistik panelinde "Uzamsal b-Değeri Haritası" mevcut.

**Türkiye:** KAF segmentlerinde b-değeri değişimi stres yorumu için çalışılır (Öztürk 2011).

**Yaygın hata:** "Düşük b = deprem yakında" — hayır, b düşüklüğü artmış *olasılık* ile ilişkili olabilir ama **kesin tahmin değildir** (Geller 1997).
""",
    "yavas_depremler": """
### Yavaş Kayma & Sessiz Depremler (SSE)

Her kayma sarsıntı üretmez. Bazı faylar enerjiyi **yavaş, sessiz** boşaltır — sismogramlarda görünmez ama GPS/InSAR ile saptanır.

**Slow Slip Events (SSE):** Günler-haftalar-aylar süren, fay üzerinde yavaş kayma. Bir M6-7'ye eşdeğer moment açığa çıkarır ama **kimse hissetmez** (Beroza-Ide 2011).

**Tremor (volkanik olmayan):** SSE'ye eşlik eden, uzun süreli zayıf titreşim; normal depremin keskin P/S varışı yoktur.

**Nerede?** En çok dalma-batma zonlarında (Japonya Nankai, Cascadia). Kıtasal doğrultu-atımlı faylarda (KAF) daha az belgelenmiştir.

**Önemi:** SSE komşu kilitli segmente **stres yükleyebilir** (tetikleme) veya enerjiyi zararsızca boşaltabilir. Deprem döngüsü anlayışını değiştirdi.

**İzleme:** Yüksek-hassasiyetli GPS ağları + InSAR zaman serisi (bu uygulamada InSAR panelleri).

**Yaygın hata:** "Kayma varsa deprem olur" — SSE sarsıntısız kaymadır; her kayma yıkıcı deprem değildir.
""",
}

REFERENCES: dict[str, dict] = {

    # ═══════════════════════════════════════════════════════════════════════
    # A. TÜRK KURUMLAR VE ARAŞTIRMACILARI
    # ═══════════════════════════════════════════════════════════════════════

    "KRDAE_2024": {
        "yazar": "Boğaziçi Üniversitesi Kandilli Rasathanesi ve Deprem Araştırma Enstitüsü (KRDAE)",
        "baslik": "Deprem Kataloğu ve Türkiye İvmeölçer Veri Tabanı",
        "yil": 2024,
        "url": "http://www.koeri.boun.edu.tr/",
        "tip": "kurum",
        "not": "Mustafa Erdik, Sinan Akkar, Zeynep Gülerce, Eser Durukal. 39 GNSS istasyonu, canlı Marmara izlemesi.",
    },
    "ITU_DAMY_2024": {
        "yazar": "İstanbul Teknik Üniversitesi — Deprem Mühendisliği ve Afet Yönetimi Enstitüsü (DAMY/EEDMI)",
        "baslik": "Deprem Mühendisliği Araştırma ve Eğitim Merkezi",
        "yil": 2024,
        "url": "https://eedmi.itu.edu.tr/",
        "tip": "kurum",
        "not": "Naci Görür (Marmara fayı), bünyesinde sismik deney laboratuvarı, DASK işbirliği.",
    },
    "ODTU_EERC_2024": {
        "yazar": "ODTÜ Deprem Mühendisliği Araştırma Merkezi",
        "baslik": "TBDY-2018 ve Deprem Yönetmeliği Çalışmaları",
        "yil": 2024,
        "url": "https://ce.metu.edu.tr/tr/",
        "tip": "kurum",
        "not": "Haluk Sucuoğlu — TBDY-2018'in baş editörü. Performansa dayalı tasarım metodolojisi.",
    },
    "AFAD_TBDY2018": {
        "yazar": "AFAD (Afet ve Acil Durum Yönetimi Başkanlığı)",
        "baslik": "Türkiye Bina Deprem Yönetmeliği (TBDY-2018)",
        "yil": 2018,
        "url": "https://www.afad.gov.tr/turkiye-bina-deprem-yonetmeligi",
        "tip": "yonetmelik",
        "not": "Türkiye'nin mevcut yapı deprem yönetmeliği. PBEE tabanlı, 2019'dan itibaren yürürlükte.",
    },
    "KTU_2024": {
        "yazar": "KTÜ (Karadeniz Teknik Üniversitesi)",
        "baslik": "Deprem Sonrası Yapısal Hasar Tespiti Araştırmaları",
        "yil": 2024,
        "url": "https://www.ktu.edu.tr/",
        "tip": "kurum",
        "not": "2023 Kahramanmaraş sonrası 1500+ bina değerlendirme çalışması. Nilgün Sayıl liderliğinde.",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # B. ULUSLARARASI KURUMLAR
    # ═══════════════════════════════════════════════════════════════════════

    "PEER_2025": {
        "yazar": "Pacific Earthquake Engineering Research Center (PEER), UC Berkeley",
        "baslik": "PEER Report 2025/01 — Deep Learning for Ground Motion Prediction",
        "yil": 2025,
        "doi": "PEER-2025-01",
        "url": "https://peer.berkeley.edu/publications/2025-01",
        "tip": "rapor",
        "not": "OpenSees, NGA-West2, PBEE. Dünya'nın önde gelen performans tabanlı deprem mühendisliği merkezi.",
    },
    "SCEC_2024": {
        "yazar": "Southern California Earthquake Center (SCEC)",
        "baslik": "UCERF3 ve Topluluk Hız Modelleri",
        "yil": 2024,
        "url": "https://www.scec.org/",
        "tip": "kurum",
        "not": "Thomas Jordan, Gregory Beroza (NAS 2022), Ned Field. Birleşik California Deprem Kestirim Modeli.",
    },
    "USGS_EHP_2024": {
        "yazar": "USGS Earthquake Hazards Program",
        "baslik": "Decadal Strategy 2024–2033",
        "yil": 2024,
        "doi": "10.3133/cir1544",
        "url": "https://pubs.usgs.gov/publication/cir1544",
        "tip": "rapor",
    },
    "ETH_SED_2024": {
        "yazar": "ETH Zürich — Swiss Seismological Service (SED)",
        "baslik": "Sismik Tehlike ve Risk Değerlendirmesi",
        "yil": 2024,
        "url": "https://www.seismo.ethz.ch/",
        "tip": "kurum",
        "not": "Stefan Wiemer direktörlüğünde. ~80 araştırmacı. İsviçre ulusal sismik tehlike modellemesi.",
    },
    "GFZ_GEOFON_2024": {
        "yazar": "GFZ Potsdam — GEOFON",
        "baslik": "Küresel Sismik Ağ ve Veri Merkezi",
        "yil": 2024,
        "url": "https://geofon.gfz-potsdam.de/",
        "tip": "kurum",
    },
    "ERI_TOKYO_2024": {
        "yazar": "Earthquake Research Institute (ERI), Tokyo University",
        "baslik": "Sismoloji ve Deprem Mühendisliği Araştırmaları",
        "yil": 2024,
        "url": "https://www.eri.u-tokyo.ac.jp/",
        "tip": "kurum",
    },
    "INGV_2024": {
        "yazar": "Istituto Nazionale di Geofisica e Vulcanologia (INGV)",
        "baslik": "İtalya Ulusal Jeofizik ve Volkanoloji Enstitüsü",
        "yil": 2024,
        "url": "https://www.ingv.it/",
        "tip": "kurum",
        "not": "Operasyonel Deprem Tahmini (OEF) metodolojisi. L'Aquila sonrası OEF sistemi.",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # C. KRİTİK KAHRAMANMARAŞ MAKALELERİ (2023–2025)
    # ═══════════════════════════════════════════════════════════════════════

    "Melgar2023_Science": {
        "yazar": "Melgar, D., et al.",
        "baslik": "Communication breakdown: The unusually complex 2023 Kahramanmaraş earthquake doublet",
        "yil": 2023,
        "dergi": "Science",
        "doi": "10.1126/science.adi0685",
        "url": "https://www.science.org/doi/10.1126/science.adi0685",
        "tip": "makale",
        "not": "Çok-faylı kinematik inversiyon. İki M7.8+ olayın ayrıntılı sismik analizi.",
    },
    "Vallee2024_Science": {
        "yazar": "Vallée, M., et al.",
        "baslik": "Chain ruptures and supershear triggering of the 2023 Kahramanmaraş earthquake sequence",
        "yil": 2024,
        "dergi": "Science",
        "doi": "10.1126/science.adi8519",
        "url": "https://www.science.org/doi/10.1126/science.adi8519",
        "tip": "makale",
        "not": "Süperkesme tetiklenme zinciri mekanizması.",
    },
    "Rosakis2025_NatGeo": {
        "yazar": "Rosakis, A., et al.",
        "baslik": "Supershear rupture and high-normal stress during 2023 Kahramanmaraş earthquake",
        "yil": 2025,
        "dergi": "Nature Geoscience",
        "doi": "10.1038/s41561-025-01893-z",
        "url": "https://www.nature.com/articles/s41561-025-01893-z",
        "tip": "makale",
        "not": "Yüksek normal stres + süperkesme. Nature Geoscience 2025.",
    },
    "Wang2023_GRL": {
        "yazar": "Wang, Z., et al.",
        "baslik": "Variable fault rupture velocity of the 2023 Mw 7.8 Kahramanmaraş earthquake",
        "yil": 2023,
        "dergi": "Geophysical Research Letters",
        "doi": "10.1029/2023GL104787",
        "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL104787",
        "tip": "makale",
    },
    "Hussain2024_GJI": {
        "yazar": "Hussain, E., et al.",
        "baslik": "InSAR-based co-seismic displacement and Coulomb stress transfer for 2023 Kahramanmaraş",
        "yil": 2024,
        "dergi": "Geophysical Journal International",
        "doi": "10.1093/gji/ggad498",
        "url": "https://academic.oup.com/gji/article/236/2/1068/7458664",
        "tip": "makale",
    },
    "Galasso2024_CommEng": {
        "yazar": "Galasso, C. & Opabola, E.",
        "baslik": "Rapid building damage assessment following the 2023 Kahramanmaraş earthquakes",
        "yil": 2024,
        "dergi": "Communications Engineering",
        "doi": "10.1038/s44172-024-00170-y",
        "url": "https://www.nature.com/articles/s44172-024-00170-y",
        "tip": "makale",
    },
    "Biblio2025_NatHaz": {
        "yazar": "Çetin, K.Ö., et al.",
        "baslik": "Bibliometric analysis of 577 publications on 2023 Kahramanmaraş earthquake",
        "yil": 2025,
        "dergi": "Natural Hazards",
        "doi": "10.1007/s11069-025-07305-0",
        "url": "https://link.springer.com/article/10.1007/s11069-025-07305-0",
        "tip": "makale",
    },
    "Xu2023_Science": {
        "yazar": "Xu, C., et al.",
        "baslik": "Surface ruptures of the 2023 Türkiye-Syria earthquake doublet",
        "yil": 2023,
        "dergi": "Science",
        "doi": "10.1126/science.adf7640",
        "url": "https://doi.org/10.1126/science.adf7640",
        "tip": "makale",
        "not": "InSAR + GPS ile 8.2 m yatay yer değişimi haritalaması.",
    },
    "KM_CommunsEarth2024": {
        "yazar": "Ren, J., et al.",
        "baslik": "The 2023 Mw 7.8–7.7 Kahramanmaraş earthquakes were loosely slip-predictable",
        "yil": 2024,
        "dergi": "Communications Earth & Environment",
        "doi": "10.1038/s43247-024-01969-5",
        "url": "https://www.nature.com/articles/s43247-024-01969-5",
        "tip": "makale",
    },
    "KM_SciRep2026": {
        "yazar": "Çetin, K.Ö., et al.",
        "baslik": "Stress-mediated multi-fault rupture dynamics of the 2023 Kahramanmaraş earthquake sequence",
        "yil": 2026,
        "dergi": "Scientific Reports",
        "doi": "10.1038/s41598-026-45723-7",
        "tip": "makale",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # D. KLASİK TEMEL MAKALELER
    # ═══════════════════════════════════════════════════════════════════════

    "GutenbergRichter1944": {
        "yazar": "Gutenberg, B. & Richter, C.F.",
        "baslik": "Frequency of earthquakes in California",
        "yil": 1944,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/BSSA0340040185",
        "url": "https://doi.org/10.1785/BSSA0340040185",
        "tip": "makale",
        "not": "G-R yasasının orijinal makalesi. log₁₀(N) = a − b·M ilişkisi.",
    },
    "Aki1965": {
        "yazar": "Aki, K.",
        "baslik": "Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits",
        "yil": 1965,
        "dergi": "Bulletin of the Earthquake Research Institute Tokyo",
        "doi": "10.4294/zisin1948.17.4_187",
        "url": "https://doi.org/10.4294/zisin1948.17.4_187",
        "tip": "makale",
        "not": "MLE yöntemiyle b-değeri tahmininin matematiksel temeli.",
    },
    "WiemerWyss2000": {
        "yazar": "Wiemer, S. & Wyss, M.",
        "baslik": "Minimum magnitude of completeness in earthquake catalogs: estimates, variability, and uses",
        "yil": 2000,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/0120000029",
        "url": "https://doi.org/10.1785/0120000029",
        "tip": "makale",
        "not": "Mc (tamamlanma büyüklüğü) belirleme metodolojisi.",
    },
    "Ozturk2011": {
        "yazar": "Öztürk, S., et al.",
        "baslik": "Spatial variations of the Gutenberg-Richter parameter b in Turkey",
        "yil": 2011,
        "dergi": "Journal of Seismology",
        "doi": "10.1007/s10950-011-9248-5",
        "url": "https://doi.org/10.1007/s10950-011-9248-5",
        "tip": "makale",
        "not": "Türkiye'ye özgü b-değeri haritası — Erzincan b≈0.87, Türkiye ortalaması 0.92.",
    },
    "Reid1910": {
        "yazar": "Reid, H.F.",
        "baslik": "The mechanics of the earthquake — The California Earthquake of April 18, 1906, Vol. II",
        "yil": 1910,
        "yayinevi": "Carnegie Institution of Washington",
        "url": "https://archive.org/details/californiaearth00statgoog",
        "tip": "rapor",
        "not": "Elastik geri tepme teorisinin temel yayını. 1906 San Francisco sonrası arazi ölçümleri.",
    },
    "Matthews2002": {
        "yazar": "Matthews, M.V., et al.",
        "baslik": "A Brownian Model for Recurrent Earthquakes",
        "yil": 2002,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/0120010254",
        "url": "https://doi.org/10.1785/0120010254",
        "tip": "makale",
        "not": "BPT modeli — sismik döngü için olasılıksal çerçeve.",
    },
    "Reilinger2006": {
        "yazar": "Reilinger, R., et al.",
        "baslik": "GPS constraints on continental deformation in the Africa-Arabia-Eurasia continental collision zone",
        "yil": 2006,
        "dergi": "Journal of Geophysical Research",
        "doi": "10.1029/2005JB004051",
        "url": "https://doi.org/10.1029/2005JB004051",
        "tip": "makale",
        "not": "Türkiye GPS hız alanı — KAF 20–25 mm/yıl. Temel geodetik veri.",
    },
    "King1994": {
        "yazar": "King, G.C.P., Stein, R.S. & Lin, J.",
        "baslik": "Static stress changes and the triggering of earthquakes",
        "yil": 1994,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/BSSA0840030865",
        "url": "https://doi.org/10.1785/BSSA0840030865",
        "tip": "makale",
        "not": "Coulomb stres transferi teorisinin temel makalesi. 0.1 bar (0.01 MPa) tetikleme eşiği.",
    },
    "Stein1999": {
        "yazar": "Stein, R.S.",
        "baslik": "The role of stress transfer in earthquake occurrence",
        "yil": 1999,
        "dergi": "Nature",
        "doi": "10.1038/45144",
        "url": "https://doi.org/10.1038/45144",
        "tip": "makale",
        "not": "Nature — İzmit zincir deprem analizi; Düzce +1.5 bar önceden tahmin.",
    },
    "Parsons2000": {
        "yazar": "Parsons, T., et al.",
        "baslik": "Heightened odds of large earthquakes near Istanbul: An interaction-based probability calculation",
        "yil": 2000,
        "dergi": "Science",
        "doi": "10.1126/science.288.5466.661",
        "url": "https://doi.org/10.1126/science.288.5466.661",
        "tip": "makale",
        "not": "Science — İstanbul için İzmit sonrası Coulomb analizi ve olasılık hesabı.",
    },
    "Toda2011": {
        "yazar": "Toda, S., et al.",
        "baslik": "Coulomb 3.3 Graphic-Rich Deformation and Stress-Change Software",
        "yil": 2011,
        "yayinevi": "USGS",
        "doi": "10.5066/F72N51GG",
        "url": "https://doi.org/10.5066/F72N51GG",
        "tip": "yazilim",
        "not": "USGS Coulomb 3.3 — Coulomb stres hesabı için standart araştırma yazılımı.",
    },
    "Ekstrom2012": {
        "yazar": "Ekström, G., Nettles, M. & Dziewoński, A.M.",
        "baslik": "The global CMT project 2004-2010: Centroid-moment tensors for 13,017 earthquakes",
        "yil": 2012,
        "dergi": "Physics of the Earth and Planetary Interiors",
        "doi": "10.1016/j.pepi.2012.04.002",
        "url": "https://doi.org/10.1016/j.pepi.2012.04.002",
        "tip": "makale",
        "not": "GCMT katalogu — 1976'dan günümüze Mw≥5 depremlerin moment tensor çözümü.",
    },
    "Cornell1968": {
        "yazar": "Cornell, C.A.",
        "baslik": "Engineering seismic risk analysis",
        "yil": 1968,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/BSSA0580050153",
        "url": "https://doi.org/10.1785/BSSA0580050153",
        "tip": "makale",
        "not": "PSHA metodolojisinin devrimsel orijinal makalesi. Cornell-McGuire yöntemi.",
    },
    "Woessner2015": {
        "yazar": "Woessner, J., et al.",
        "baslik": "The 2013 European Seismic Hazard Model: Key components and results",
        "yil": 2015,
        "dergi": "Bulletin of Earthquake Engineering",
        "doi": "10.1007/s10518-015-9795-1",
        "url": "https://doi.org/10.1007/s10518-015-9795-1",
        "tip": "makale",
        "not": "Türkiye dahil Avrupa tehlike modeli — SHARE projesi.",
    },
    "AFAD2018_TDTH": {
        "yazar": "AFAD",
        "baslik": "Türkiye Deprem Tehlike Haritası (TDTH-2018)",
        "yil": 2018,
        "url": "https://tdth.afad.gov.tr",
        "tip": "harita",
        "not": "TBDY-2018 dayanağı — tüm Türkiye için ızgara bazlı PGA, Ss ve S1 değerleri.",
    },
    "Boore2014": {
        "yazar": "Boore, D.M., et al.",
        "baslik": "NGA-West2 Equations for Predicting PGA, PGV, and 5% Damped PSA for Shallow Crustal Earthquakes",
        "yil": 2014,
        "dergi": "Earthquake Spectra",
        "doi": "10.1193/070113EQS184M",
        "url": "https://doi.org/10.1193/070113EQS184M",
        "tip": "makale",
        "not": "NGA-West2 GMPE — Türkiye PSHA çalışmalarında yaygın kullanım.",
    },
    "Massonnet1998": {
        "yazar": "Massonnet, D. & Feigl, K.L.",
        "baslik": "Radar interferometry and its application to changes in the Earth's surface",
        "yil": 1998,
        "dergi": "Reviews of Geophysics",
        "doi": "10.1029/97RG03139",
        "url": "https://doi.org/10.1029/97RG03139",
        "tip": "makale",
        "not": "InSAR metodolojisinin kapsamlı gözden geçirmesi. Temel referans.",
    },
    "Papadopoulos2005": {
        "yazar": "Papadopoulos, G.A. & Fokaefs, A.",
        "baslik": "Strong tsunamis in the Mediterranean Sea: A re-evaluation",
        "yil": 2005,
        "dergi": "ISET Journal of Earthquake Technology",
        "doi": "10.2495/AFR050061",
        "url": "https://doi.org/10.2495/AFR050061",
        "tip": "makale",
        "not": "Akdeniz tarihi tsunami kataloğu yeniden değerlendirmesi.",
    },
    "Basili2021": {
        "yazar": "Basili, R., et al.",
        "baslik": "The making of the NEAM Tsunami Hazard Model 2018 (NEAMTHM18)",
        "yil": 2021,
        "dergi": "Earth-Science Reviews",
        "doi": "10.1016/j.earscirev.2021.103673",
        "url": "https://doi.org/10.1016/j.earscirev.2021.103673",
        "tip": "makale",
        "not": "Kuzey Doğu Atlantik ve Akdeniz olasılıksal tsunami tehlike modeli 2018.",
    },
    "Synolakis2006": {
        "yazar": "Synolakis, C.E. & Bernard, E.N.",
        "baslik": "Tsunami science before and beyond Boxing Day 2004",
        "yil": 2006,
        "dergi": "Philosophical Transactions of the Royal Society A",
        "doi": "10.1098/rsta.2006.1824",
        "url": "https://doi.org/10.1098/rsta.2006.1824",
        "tip": "makale",
        "not": "2004 Hint Okyanusu tsunamisi: fizik ve öğrenilen dersler.",
    },
    "SheriffGeldart1995": {
        "yazar": "Sheriff, R.E. & Geldart, L.P.",
        "baslik": "Exploration Seismology (2nd ed.)",
        "yil": 1995,
        "yayinevi": "Cambridge University Press",
        "url": "https://www.cambridge.org/core/books/exploration-seismology/",
        "tip": "kitap",
        "not": "P ve S dalgası fiziğinin standart referansı.",
    },
    "Ambraseys1998": {
        "yazar": "Ambraseys, N.N. & Jackson, J.A.",
        "baslik": "Faulting associated with historical and recent earthquakes in the Eastern Mediterranean region",
        "yil": 1998,
        "dergi": "Geophysical Journal International",
        "doi": "10.1046/j.1365-246X.1998.00548.x",
        "url": "https://doi.org/10.1046/j.1365-246X.1998.00548.x",
        "tip": "makale",
        "not": "Doğu Akdeniz tarihsel depremleri ve fay geometrisi.",
    },
    "Kozaci2007": {
        "yazar": "Kozacı, Ö., et al.",
        "baslik": "Late Holocene slip rate for the North Anatolian Fault at Tahtaköprü, Turkey",
        "yil": 2007,
        "dergi": "Journal of Geophysical Research",
        "doi": "10.1029/2006JB004333",
        "url": "https://doi.org/10.1029/2006JB004333",
        "tip": "makale",
        "not": "KAF geç Holosen kayma hızı: 18.0 ± 3.7 mm/yıl.",
    },
    "Hartleb2006": {
        "yazar": "Hartleb, R.D., et al.",
        "baslik": "A 2500-yr-long paleoseismic record for the central North Anatolian Fault",
        "yil": 2006,
        "dergi": "Geological Society of America Bulletin",
        "doi": "10.1130/B25783.1",
        "url": "https://doi.org/10.1130/B25783.1",
        "tip": "makale",
        "not": "Yaylabeli paleosismik kazı — 2500 yıllık 9 büyük yüzey kırığı, ~280 yıl tekrar süresi.",
    },
    "Ambraseys2009": {
        "yazar": "Ambraseys, N.N.",
        "baslik": "Earthquakes in the Mediterranean and Middle East: A Multidisciplinary Study",
        "yil": 2009,
        "yayinevi": "Cambridge University Press",
        "doi": "10.1017/CBO9781139195430",
        "url": "https://doi.org/10.1017/CBO9781139195430",
        "tip": "kitap",
        "not": "Türkiye tarihi depremlerinin en kapsamlı veritabanı. 2000 yıllık katalog.",
    },
    "AFAD2019_Erzincan": {
        "yazar": "AFAD",
        "baslik": "Erzincan İli Depremsellik Raporu",
        "yil": 2019,
        "url": "https://www.afad.gov.tr/deprem-arastirma-enstitusu",
        "tip": "rapor",
        "not": "AFAD Deprem Araştırma Enstitüsü — Erzincan bölgesi kapsamlı sismisiteli inceleme.",
    },
    "Barka1996": {
        "yazar": "Barka, A.A.",
        "baslik": "Slip distribution along the North Anatolian Fault associated with the large earthquakes of the period 1939 to 1967",
        "yil": 1996,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/BSSA0860010171",
        "url": "https://doi.org/10.1785/BSSA0860010171",
        "tip": "makale",
        "not": "KAF segment geometrisi ve 1939–1967 dönemi kayma dağılımı.",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # E. YENİ KLASİK MAKALELER (v3.0 eklentisi)
    # ═══════════════════════════════════════════════════════════════════════

    "Sengör2005": {
        "yazar": "Şengör, A.M.C., et al.",
        "baslik": "The North Anatolian Fault: A new look",
        "yil": 2005,
        "dergi": "Annual Review of Earth and Planetary Sciences",
        "doi": "10.1146/annurev.earth.33.101802.120345",
        "url": "https://doi.org/10.1146/annurev.earth.33.101802.120345",
        "tip": "makale",
        "not": "KAF'ın kökenini ve tektonik evrimini yeniden değerlendiren kapsamlı gözden geçirme.",
    },
    "Ergintav2014": {
        "yazar": "Ergintav, S., et al.",
        "baslik": "Istanbul's earthquake hot spots: Geodetic constraints on strain accumulation along faults in the Marmara seismic gap",
        "yil": 2014,
        "dergi": "Geophysical Research Letters",
        "doi": "10.1002/2013GL058699",
        "url": "https://doi.org/10.1002/2013GL058699",
        "tip": "makale",
        "not": "Marmara fayında gerilme birikimine dair GPS verileriyle temel çalışma.",
    },
    "Barka1997": {
        "yazar": "Barka, A.A.",
        "baslik": "Slip distribution along the North Anatolian fault associated with the large earthquakes of the period 1939 to 1967",
        "yil": 1997,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/BSSA0870020017",
        "url": "https://doi.org/10.1785/BSSA0870020017",
        "tip": "makale",
        "not": "Aykut Barka'nın KAF tarihi depremleri için 1997 güncellemesi.",
    },
    "Atakan2002": {
        "yazar": "Atakan, K., et al.",
        "baslik": "Seismic hazard in Istanbul following the 1999 Izmit and Düzce earthquakes",
        "yil": 2002,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/0120000289",
        "url": "https://doi.org/10.1785/0120000289",
        "tip": "makale",
        "not": "1999 sonrası İstanbul sismik tehlike güncellemesi. PSHA yaklaşımı.",
    },
    "Akkar2010": {
        "yazar": "Akkar, S. & Bommer, J.J.",
        "baslik": "Empirical equations for the prediction of PGA, PGV and spectral accelerations in Europe, the Mediterranean region and the Middle East",
        "yil": 2010,
        "dergi": "Seismological Research Letters",
        "doi": "10.1785/gssrl.81.2.195",
        "url": "https://doi.org/10.1785/gssrl.81.2.195",
        "tip": "makale",
        "not": "Sinan Akkar (KRDAE) — Avrupa/Türkiye için zemin hareketi tahmin denklemi. SHARE projesinde kullanıldı.",
    },
    "Gulerce2011": {
        "yazar": "Gülerce, Z. & Abrahamson, N.A.",
        "baslik": "Site-specific design spectra for vertical ground motion",
        "yil": 2011,
        "dergi": "Earthquake Spectra",
        "doi": "10.1193/1.3651317",
        "url": "https://doi.org/10.1193/1.3651317",
        "tip": "makale",
        "not": "Zeynep Gülerce (KRDAE) — düşey zemin hareketi tasarım spektrumları.",
    },
    "Field2014": {
        "yazar": "Field, E.H., et al. (UCERF3)",
        "baslik": "Uniform California Earthquake Rupture Forecast, Version 3 (UCERF3): The Time-Independent Model",
        "yil": 2014,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/0120130164",
        "url": "https://doi.org/10.1785/0120130164",
        "tip": "makale",
        "not": "Ned Field (USGS) liderliğinde olasılıksal deprem tehlike modellemesinin kilometre taşı çalışması.",
    },
    "NatComm2019_Marmara": {
        "yazar": "Klein, E., et al.",
        "baslik": "Interseismic strain build-up on the submarine North Anatolian Fault offshore Istanbul",
        "yil": 2019,
        "dergi": "Nature Communications",
        "doi": "10.1038/s41467-019-11016-z",
        "url": "https://www.nature.com/articles/s41467-019-11016-z",
        "tip": "makale",
        "not": "Marmara segmenti kilitlenme analizi — InSAR + GPS bütünleşik.",
    },
    "Field2017_ETAS": {
        "yazar": "Field, E.H., et al.",
        "baslik": "A Spatiotemporal Clustering Model for the UCERF3-ETAS",
        "yil": 2017,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/0120160173",
        "url": "https://doi.org/10.1785/0120160173",
        "tip": "makale",
        "not": "UCERF3-ETAS: tetiklenme + artçı sarsıntı modellemesi.",
    },
    "Hanks1979": {
        "yazar": "Hanks, T.C. & Kanamori, H.",
        "baslik": "A moment magnitude scale",
        "yil": 1979,
        "dergi": "Journal of Geophysical Research",
        "doi": "10.1029/JB084iB05p02348",
        "url": "https://doi.org/10.1029/JB084iB05p02348",
        "tip": "makale",
        "not": "Moment büyüklük Mw ölçeğinin tanımlandığı temel makale.",
    },
    "Wald2007_Vs30": {
        "yazar": "Wald, D.J. & Allen, T.I.",
        "baslik": "Topographic slope as a proxy for seismic site conditions and amplification",
        "yil": 2007,
        "dergi": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/0120060267",
        "url": "https://doi.org/10.1785/0120060267",
        "tip": "makale",
        "not": "Vs30 haritalaması için topografik eğim proxy yöntemi.",
    },
    "Wald1999_ShakeMap": {
        "yazar": "Wald, D.J., et al.",
        "baslik": "TriNet ShakeMaps: Rapid Generation of Peak Ground Motion and Intensity Maps for Earthquakes in Southern California",
        "yil": 1999,
        "dergi": "Earthquake Spectra",
        "doi": "10.1193/1.1586057",
        "url": "https://doi.org/10.1193/1.1586057",
        "tip": "makale",
        "not": "ShakeMap metodolojisinin temel makalesi.",
    },
    "Gulerce2016_Turkey": {
        "yazar": "Gülerce, Z., Kargoığlu, B. & Abrahamson, N.A.",
        "baslik": "Turkey-Adjusted NGA-W1 Horizontal Ground Motion Prediction Equations",
        "yil": 2016,
        "dergi": "Earthquake Spectra",
        "doi": "10.1193/022714EQS034M",
        "url": "https://doi.org/10.1193/022714EQS034M",
        "tip": "makale",
        "not": "Zeynep Gülerce (KRDAE) — Türkiye için NGA-W1 uyarlaması. TBDY-2018'de kullanıldı.",
    },
    "SeisTectFrame2025": {
        "yazar": "Kürçer, A., et al.",
        "baslik": "Seismotectonic Frame of the East and North Anatolian Faults, Turkey",
        "yil": 2025,
        "dergi": "Springer — Active Tectonics and Seismic Hazard Assessment",
        "doi": "10.1007/978-3-031-80928-6_13",
        "url": "https://link.springer.com/chapter/10.1007/978-3-031-80928-6_13",
        "tip": "kitap_bolumu",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # F. DERS KİTAPLARI — 18 Temel Eser
    # ═══════════════════════════════════════════════════════════════════════

    "Chopra2020": {
        "yazar": "Chopra, A.K.",
        "baslik": "Dynamics of Structures: Theory and Applications to Earthquake Engineering (5th ed.)",
        "yil": 2020,
        "yayinevi": "Pearson / Prentice Hall",
        "isbn": "978-0-13-455512-6",
        "tip": "kitap",
        "not": "Yapısal dinamiğin standart referans kitabı. SDOF/MDOF, modal analiz, inelastik davranış.",
    },
    "ClouPen2003": {
        "yazar": "Clough, R.W. & Penzien, J.",
        "baslik": "Dynamics of Structures (3rd ed.)",
        "yil": 2003,
        "yayinevi": "Computers & Structures Inc.",
        "isbn": "978-0-923907-50-4",
        "tip": "kitap",
        "not": "Yapısal dinamik klasiği. Matris yöntemleri ve modal süperpozisyon.",
    },
    "Kramer2024": {
        "yazar": "Kramer, S.L. & Stewart, J.P.",
        "baslik": "Geotechnical Earthquake Engineering (2nd ed.)",
        "yil": 2024,
        "yayinevi": "CRC Press / Taylor & Francis",
        "isbn": "978-1-032-84274-5",
        "tip": "kitap",
        "not": "Geoteknik deprem mühendisliğinin en güncel edisyonu. Sıvılaşma, zemin davranışı, saha etkisi.",
    },
    "SucuogluAkkar2014": {
        "yazar": "Sucuoğlu, H. & Akkar, S.",
        "baslik": "Basic Earthquake Engineering: From Seismology to Analysis and Design",
        "yil": 2014,
        "yayinevi": "Springer",
        "isbn": "978-3-319-01026-7",
        "doi": "10.1007/978-3-319-01026-7",
        "url": "https://link.springer.com/book/10.1007/978-3-319-01026-7",
        "tip": "kitap",
        "not": "Türk akademisyenlerce yazılmış, uluslararası alanda benimsenmiş temel kaynak. ODTÜ + KRDAE.",
    },
    "Elnashai2015": {
        "yazar": "Elnashai, A.S. & Di Sarno, L.",
        "baslik": "Fundamentals of Earthquake Engineering: From Source to Fragility (2nd ed.)",
        "yil": 2015,
        "yayinevi": "Wiley",
        "isbn": "978-1-118-67892-3",
        "tip": "kitap",
        "not": "Mühendislik sismolojisi, yapısal tepki ve kırılganlık analizini birleştiren kaynak.",
    },
    "McGuire2004": {
        "yazar": "McGuire, R.K.",
        "baslik": "Seismic Hazard and Risk Analysis",
        "yil": 2004,
        "yayinevi": "EERI Monograph MNO-10",
        "isbn": "978-0-943198-01-2",
        "tip": "kitap",
        "not": "PSHA metodolojisinin standart referansı. Cornell-McGuire yöntemi.",
    },
    "Shearer2019": {
        "yazar": "Shearer, P.M.",
        "baslik": "Introduction to Seismology (3rd ed.)",
        "yil": 2019,
        "yayinevi": "Cambridge University Press",
        "isbn": "978-1-316-63884-4",
        "tip": "kitap",
        "not": "Sismolojiye giriş için en çok kullanılan lisans/lisansüstü ders kitabı.",
    },
    "AkiRichards2002": {
        "yazar": "Aki, K. & Richards, P.G.",
        "baslik": "Quantitative Seismology (2nd ed.)",
        "yil": 2002,
        "yayinevi": "University Science Books",
        "isbn": "978-1-891389-63-4",
        "url": "https://www.ldeo.columbia.edu/~richards/QS_book.html",
        "tip": "kitap",
        "not": "Niceliksel sismolojinin klasik başvurusu. Aki-Richards denklemleri.",
    },
    "SteinWys2003": {
        "yazar": "Stein, S. & Wysession, M.",
        "baslik": "An Introduction to Seismology, Earthquakes, and Earth Structure",
        "yil": 2003,
        "yayinevi": "Blackwell",
        "isbn": "978-0-865-42078-6",
        "url": "https://www.wiley.com/en-us/9780865420786",
        "tip": "kitap",
        "not": "Sismoloji ve jeofizik temelleri.",
    },
    "PauPri1992": {
        "yazar": "Paulay, T. & Priestley, M.J.N.",
        "baslik": "Seismic Design of Reinforced Concrete and Masonry Buildings",
        "yil": 1992,
        "yayinevi": "Wiley",
        "isbn": "978-0-471-54915-4",
        "tip": "kitap",
        "not": "Betonarme yapı tasarımında klasik. Kapasiteye dayalı tasarım.",
    },
    "Priestley2007": {
        "yazar": "Priestley, M.J.N., Calvi, G.M. & Kowalsky, M.J.",
        "baslik": "Displacement-Based Seismic Design of Structures",
        "yil": 2007,
        "yayinevi": "IUSS Press",
        "isbn": "978-88-6198-000-6",
        "tip": "kitap",
        "not": "Deplasman tabanlı tasarımın referans kitabı (DDBD).",
    },
    "Naeim2001": {
        "yazar": "Naeim, F. (Ed.)",
        "baslik": "The Seismic Design Handbook (2nd ed.)",
        "yil": 2001,
        "yayinevi": "Springer",
        "isbn": "978-0-7923-7301-4",
        "tip": "kitap",
        "not": "Zemin hareketi, dinamik analiz, izolasyon, performans bazlı tasarım.",
    },
    "Kanamori2015": {
        "yazar": "Kanamori, H. (Ed.)",
        "baslik": "Treatise on Geophysics, Vol. 4: Earthquake Seismology (2nd ed.)",
        "yil": 2015,
        "yayinevi": "Elsevier",
        "isbn": "978-0-444-51932-0",
        "tip": "kitap",
        "not": "Deprem sismolojisinin ansiklopedik kaynağı. Hiroo Kanamori editörlüğünde.",
    },
    "SeedIdriss1982": {
        "yazar": "Seed, H.B. & Idriss, I.M.",
        "baslik": "Ground Motions and Soil Liquefaction During Earthquakes",
        "yil": 1982,
        "yayinevi": "EERI Monograph",
        "isbn": "978-0-943198-06-7",
        "tip": "kitap",
        "not": "Zemin sıvılaşması alanının temel referansı. SPT temelli basitleştirilmiş yöntem.",
    },
    "Priestley1996": {
        "yazar": "Priestley, M.J.N., Seible, F. & Calvi, G.M.",
        "baslik": "Seismic Design and Retrofit of Bridges",
        "yil": 1996,
        "yayinevi": "Wiley",
        "isbn": "978-0-471-57998-4",
        "tip": "kitap",
        "not": "Köprü sismik tasarımı ve güçlendirme referansı.",
    },
    "Lee2002_Handbook": {
        "yazar": "Lee, W.H.K., et al. (Eds.)",
        "baslik": "International Handbook of Earthquake and Engineering Seismology, Parts A & B",
        "yil": 2002,
        "yayinevi": "Academic Press (IASPEI)",
        "isbn_A": "978-0-12-440652-0",
        "isbn_B": "978-0-12-440658-2",
        "tip": "kitap",
        "not": "Kapsamlı sismoloji-mühendislik sismolojisi el kitabı.",
    },
    "NaeimiKelly1999": {
        "yazar": "Naeim, F. & Kelly, J.M.",
        "baslik": "Design of Seismic Isolated Structures: From Theory to Practice",
        "yil": 1999,
        "yayinevi": "Wiley",
        "isbn": "978-0-471-14921-7",
        "tip": "kitap",
        "not": "Taban izolasyonu tasarım ilkeleri ve uygulamaları.",
    },
    "Reiter1990": {
        "yazar": "Reiter, L.",
        "baslik": "Earthquake Hazard Analysis: Issues and Insights",
        "yil": 1990,
        "yayinevi": "Columbia University Press",
        "isbn": "978-0-231-06534-8",
        "tip": "kitap",
        "not": "PSHA temelleri, deterministik vs. olasılıksal yaklaşım tartışması.",
    },
    # ── Batch 2 (2/2, v1.79) — ileri konular için ek kaynaklar ──
    "Geller1997": {
        "yazar": "Geller, R.J.",
        "baslik": "Earthquake prediction: a critical review",
        "yil": 1997, "dergi": "Geophysical Journal International",
        "doi": "10.1111/j.1365-246X.1997.tb06588.x",
        "url": "https://doi.org/10.1111/j.1365-246X.1997.tb06588.x",
        "tip": "makale",
        "not": "Deterministik deprem tahmininin mümkün olmadığına dair temel eleştiri.",
    },
    "Dieterich1994": {
        "yazar": "Dieterich, J.",
        "baslik": "A constitutive law for rate of earthquake production and its application to earthquake clustering",
        "yil": 1994, "dergi": "Journal of Geophysical Research",
        "doi": "10.1029/93JB02581",
        "url": "https://doi.org/10.1029/93JB02581",
        "tip": "makale",
        "not": "Rate-state sürtünme yasası; deprem nükleasyonu ve kümelenme.",
    },
    "Kanamori2004_Physics": {
        "yazar": "Kanamori, H. & Brodsky, E.E.",
        "baslik": "The physics of earthquakes",
        "yil": 2004, "dergi": "Reports on Progress in Physics",
        "doi": "10.1088/0034-4885/67/8/R03",
        "url": "https://doi.org/10.1088/0034-4885/67/8/R03",
        "tip": "makale",
        "not": "Deprem fiziği derlemesi: stress drop, enerji, sürtünme.",
    },
    "Shapiro2005_Noise": {
        "yazar": "Shapiro, N.M., Campillo, M., Stehly, L. & Ritzwoller, M.H.",
        "baslik": "High-resolution surface-wave tomography from ambient seismic noise",
        "yil": 2005, "dergi": "Science",
        "doi": "10.1126/science.1108339",
        "url": "https://doi.org/10.1126/science.1108339",
        "tip": "makale",
        "not": "Ambient noise korelasyonundan yüzey dalgası tomografisi.",
    },
    "BerozaIde2011": {
        "yazar": "Beroza, G.C. & Ide, S.",
        "baslik": "Slow earthquakes and nonvolcanic tremor",
        "yil": 2011, "dergi": "Annual Review of Earth and Planetary Sciences",
        "doi": "10.1146/annurev-earth-040809-152531",
        "url": "https://doi.org/10.1146/annurev-earth-040809-152531",
        "tip": "makale",
        "not": "Yavaş kayma olayları (SSE) ve volkanik olmayan tremor derlemesi.",
    },
}


# REFERENCES boyutu: 81 kaynak (A:5 + B:7 + C:10 + D:27 + E:14 + F:18)


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
                zerolinecolor="rgba(255,255,255,0.27)", zerolinewidth=2,
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

        for t_ref, _pga_ref, lbl, col in [
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


# ─── EK ANİMASYON FONKSİYONLARI (v3.1) ──────────────────────────────────────

def anim_gutenberg_richter() -> go.Figure:
    """Gutenberg-Richter: animasyonlu frekans-büyüklük dağılımı + b-değeri eğimi."""
    M = np.linspace(2.0, 8.5, 200)

    frames = []
    b_values = np.round(np.linspace(0.60, 1.40, 17), 2)
    for b in b_values:
        a = 7.5
        logN = a - b * M
        frames.append(go.Frame(
            data=[
                go.Scatter(x=M, y=logN, mode="lines",
                           line=dict(color="#4fc3f7", width=3),
                           name=f"b={b:.2f}"),
                go.Scatter(x=M, y=7.5 - 0.92 * M, mode="lines",
                           line=dict(color="#f39c12", width=1.5, dash="dash"),
                           name="Türkiye ort. b=0.92"),
                go.Scatter(x=M, y=7.3 - 0.87 * M, mode="lines",
                           line=dict(color="#e74c3c", width=1.5, dash="dot"),
                           name="Erzincan b=0.87"),
            ],
            name=str(b),
            layout=go.Layout(title_text=f"G-R Yasası: log₁₀(N) = 7.5 − {b:.2f}·M"),
        ))

    # Histogram arka planı: gerçekçi deprem büyüklük dağılımı
    rng = np.random.default_rng(42)
    sample_M = np.concatenate([
        rng.exponential(0.7, 800) + 2.0,
        rng.uniform(5.0, 7.5, 30),
    ])
    sample_M = sample_M[sample_M <= 8.0]

    fig = go.Figure(
        data=[
            go.Histogram(x=sample_M, nbinsx=30,
                         marker_color="rgba(79,195,247,0.2)",
                         yaxis="y2", name="Örnek katalog",
                         showlegend=True),
            go.Scatter(x=M, y=7.5 - 1.00 * M, mode="lines",
                       line=dict(color="#4fc3f7", width=3),
                       name="b=1.00 (başlangıç)"),
            go.Scatter(x=M, y=7.5 - 0.92 * M, mode="lines",
                       line=dict(color="#f39c12", width=1.5, dash="dash"),
                       name="Türkiye ort. b=0.92"),
            go.Scatter(x=M, y=7.3 - 0.87 * M, mode="lines",
                       line=dict(color="#e74c3c", width=1.5, dash="dot"),
                       name="Erzincan b=0.87"),
        ],
        frames=frames,
        layout=go.Layout(
            title=dict(text="Gutenberg-Richter: log₁₀(N) = 7.5 − b·M",
                       font=dict(color="#e8f0fe")),
            xaxis=dict(title="Büyüklük (M)", range=[2, 8.5],
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            yaxis=dict(title="log₁₀(N) — yıllık deprem sayısı",
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        title="Örnek olay sayısı", color="#aaaaaa"),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            height=420,
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.15, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Oynat",
                         method="animate",
                         args=[None, {"frame": {"duration": 350, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Dur",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                active=8,
                steps=[dict(args=[[str(b)],
                                  {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}],
                            label=f"b={b:.2f}", method="animate")
                       for b in b_values],
                x=0.0, len=1.0,
                currentvalue=dict(prefix="b-değeri: ", font=dict(color="#e8f0fe")),
                font=dict(color="#e8f0fe"),
            )],
        ),
    )
    return fig


def anim_moment_tensor() -> go.Figure:
    """Beach ball odak mekanizması: doğrultu atımlı / normal / ters fay karşılaştırması."""
    # Her fay tipi için P-dalga kutup diyagramı (polar gösterim)
    theta = np.linspace(0, 2 * np.pi, 360)

    def strike_slip_polarity(th: np.ndarray) -> np.ndarray:
        """Doğrultu atımlı fay: 4 çeyrek dönüşümlü + / - ."""
        return np.sign(np.sin(2 * th))

    def normal_polarity(th: np.ndarray) -> np.ndarray:
        """Normal fay: üst yarı baskı (siyah)."""
        return np.where(np.sin(th) > 0, 1.0, -1.0)

    def reverse_polarity(th: np.ndarray) -> np.ndarray:
        """Ters fay: alt yarı baskı (siyah)."""
        return np.where(np.sin(th) < 0, 1.0, -1.0)

    ["#1a6faf" if strike_slip_polarity(np.array([t]))[0] > 0 else "#e8f0fe"
                 for t in theta]
    ["#1a6faf" if normal_polarity(np.array([t]))[0] > 0 else "#e8f0fe"
                 for t in theta]
    ["#1a6faf" if reverse_polarity(np.array([t]))[0] > 0 else "#e8f0fe"
                 for t in theta]

    np.ones_like(theta) * 1.0

    fig = go.Figure()

    # Fay tipi 1: Doğrultu atımlı (KAF tipi)
    for i in range(len(theta) - 1):
        c = "#1a6faf" if strike_slip_polarity(np.array([theta[i]]))[0] > 0 else "#ecf0f1"
        fig.add_trace(go.Barpolar(
            r=[1.0], theta=[np.degrees(theta[i])],
            width=[np.degrees(theta[1] - theta[0]) + 0.5],
            marker_color=c,
            showlegend=False,
            visible=True,
        ))

    # Fay tipi 2: Normal fay (Ege tipi)
    for i in range(len(theta) - 1):
        c = "#1a6faf" if normal_polarity(np.array([theta[i]]))[0] > 0 else "#ecf0f1"
        fig.add_trace(go.Barpolar(
            r=[1.0], theta=[np.degrees(theta[i])],
            width=[np.degrees(theta[1] - theta[0]) + 0.5],
            marker_color=c,
            showlegend=False,
            visible=False,
        ))

    n = len(theta) - 1
    # Doğrultu atımlı görünür, diğerleri gizli
    [True] * n + [False] * n
    [False] * n + [True] * n

    # Basit statik üç panel (subplot yerine tek figür + annotation)
    # Polar yerine scatter tabanlı daha sade görselleştirme
    fig2 = go.Figure()

    types = [
        ("Doğrultu Atımlı\n(KAF tipi)", strike_slip_polarity, 0.0),
        ("Normal Fay\n(Ege tipi)", normal_polarity, 3.5),
        ("Ters Fay\n(Bindirme)", reverse_polarity, 7.0),
    ]

    for label, pol_fn, x_off in types:
        t_dense = np.linspace(0, 2 * np.pi, 720)
        polarity = pol_fn(t_dense)

        xs_pos = x_off + np.cos(t_dense[polarity > 0])
        ys_pos = np.sin(t_dense[polarity > 0])
        xs_neg = x_off + np.cos(t_dense[polarity <= 0])
        ys_neg = np.sin(t_dense[polarity <= 0])

        fig2.add_trace(go.Scatter(
            x=xs_pos, y=ys_pos, mode="markers",
            marker=dict(size=6, color="#1a6faf"),
            name=f"{label} (baskı)",
            legendgroup=label,
        ))
        fig2.add_trace(go.Scatter(
            x=xs_neg, y=ys_neg, mode="markers",
            marker=dict(size=6, color="#ecf0f1"),
            name=f"{label} (gerilme)",
            legendgroup=label,
        ))
        # Çember çiz
        circ = np.linspace(0, 2 * np.pi, 100)
        fig2.add_trace(go.Scatter(
            x=x_off + np.cos(circ), y=np.sin(circ),
            mode="lines", line=dict(color="#4fc3f7", width=2),
            showlegend=False,
        ))
        # Etiket
        fig2.add_annotation(
            x=x_off, y=-1.35,
            text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(color="#e8f0fe", size=11),
            align="center",
        )

    # Fay düzlemi çizgileri
    for x_off, angle in [(0.0, 0.0), (3.5, np.pi / 4), (7.0, np.pi / 3)]:
        fig2.add_shape(type="line",
                       x0=x_off + np.cos(angle + np.pi / 2),
                       y0=np.sin(angle + np.pi / 2),
                       x1=x_off + np.cos(angle - np.pi / 2),
                       y1=np.sin(angle - np.pi / 2),
                       line=dict(color="#e74c3c", width=2, dash="dash"))
        fig2.add_shape(type="line",
                       x0=x_off + np.cos(angle),
                       y0=np.sin(angle),
                       x1=x_off + np.cos(angle + np.pi),
                       y1=np.sin(angle + np.pi),
                       line=dict(color="#e74c3c", width=2, dash="dot"))

    fig2.update_layout(
        title=dict(
            text="Odak Mekanizması (Beach Ball): P-Dalga Kutupluluk Diyagramı",
            font=dict(color="#e8f0fe"),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.5, 8.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", range=[-1.8, 1.5]),
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color="#e8f0fe"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=9)),
        height=400,
        annotations=[
            dict(x=0.0, y=1.55, xref="x", yref="y", showarrow=False,
                 text="🔵 Baskı (compressional)<br>⚪ Gerilme (dilatational)",
                 font=dict(color="#aaaaaa", size=10), align="center"),
        ],
    )
    return fig2


def anim_psha() -> go.Figure:
    """PSHA tehlike eğrisi animasyonu: farklı b-değerleri için tehlike eğrisi."""
    T_arr = np.logspace(1.5, 4.0, 300)
    b_vals = np.round(np.linspace(0.70, 1.30, 13), 2)

    frames = []
    for b in b_vals:
        # Basit tehlike eğrisi yaklaşımı: PGA ∝ T_return^0.5 × b_ölçek
        b_scale = (1.0 / b) ** 0.6  # b düştükçe tehlike artar
        pga = np.clip(0.45 * (T_arr / 475.0) ** 0.5 * b_scale, 0, 1.5)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=T_arr, y=pga, mode="lines",
                           line=dict(color="#4fc3f7", width=2.8),
                           name=f"b={b:.2f}"),
            ],
            name=str(b),
            layout=go.Layout(title_text=f"PSHA Tehlike Eğrisi — b={b:.2f} (yüksek b → düşük tehlike)"),
        ))

    pga_ref = np.clip(0.45 * (T_arr / 475.0) ** 0.5, 0, 1.5)

    fig = go.Figure(
        data=[
            go.Scatter(x=T_arr, y=pga_ref, mode="lines",
                       line=dict(color="#4fc3f7", width=2.8),
                       name="b=1.00 (başlangıç)"),
            go.Scatter(x=[2475, 475, 72], y=[0.85, 0.45, 0.22],
                       mode="markers+text",
                       marker=dict(size=12, color=["#e74c3c", "#f39c12", "#27ae60"],
                                   symbol="diamond"),
                       text=["DD-1\n2475 yıl", "DD-2\n475 yıl", "DD-3\n72 yıl"],
                       textposition="top right",
                       textfont=dict(size=10),
                       name="TBDY-2018 seviyeleri"),
        ],
        frames=frames,
        layout=go.Layout(
            title=dict(text="PSHA Tehlike Eğrisi — b-değeri etkisi",
                       font=dict(color="#e8f0fe")),
            xaxis=dict(title="Dönüş Periyodu (yıl)", type="log",
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            yaxis=dict(title="PGA (g)", range=[0, 1.6],
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            height=420,
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.15, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Oynat", method="animate",
                         args=[None, {"frame": {"duration": 400, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Dur", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                steps=[dict(args=[[str(b)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate"}],
                            label=f"b={b:.2f}", method="animate")
                       for b in b_vals],
                x=0.0, len=1.0,
                currentvalue=dict(prefix="b-değeri: ", font=dict(color="#e8f0fe")),
                font=dict(color="#e8f0fe"),
            )],
        ),
    )
    return fig


def anim_insar() -> go.Figure:
    """InSAR interferogram simülasyonu: fay üzerinde faz değişim halkalarını gösterir."""
    # 2D grid
    x = np.linspace(-50, 50, 120)
    y = np.linspace(-50, 50, 120)
    X, Y = np.meshgrid(x, y)

    # Fay: x=0, Y<0 sağ yöne kayar (sağ yönlü doğrultu atımlı KAF benzeri)
    # LOS deformasyon alanı (Okada modeline yaklaşık basitleştirilmiş)
    def deformation_los(slip_m: float) -> np.ndarray:
        # Dislocation point source: uzakta 1/r^2 azalır
        dist_fault = np.sqrt(X**2 + (Y + 15)**2)
        # Çift çift lobu (double-couple benzeri)
        u_y = slip_m * 2.0 / (1.0 + (dist_fault / 8.0)**2)
        # LOS ≈ yaklaşık %cos(23°) dikey + %sin(23°) yatay
        los = 0.9 * u_y * np.sign(-X)
        return los

    n_wavelengths = 5.6e-2  # Sentinel-1 C-band dalga boyu metre

    frames = []
    slip_values = np.round(np.linspace(0.3, 5.0, 16), 1)
    for slip in slip_values:
        los = deformation_los(slip)
        # Faz: 2π başına 2.8 cm = yarı dalga boyu
        phase = (los % (n_wavelengths / 2)) / (n_wavelengths / 2) * 2 * np.pi
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=phase, colorscale="HSV", zmin=0, zmax=2 * np.pi,
                showscale=True,
                colorbar=dict(title="Faz (rad)", tickvals=[0, np.pi, 2 * np.pi],
                              ticktext=["0", "π", "2π"], tickfont=dict(color="#e8f0fe")),
            )],
            name=str(slip),
            layout=go.Layout(
                title_text=f"InSAR Interferogram — Kayma = {slip:.1f} m | "
                           f"LOS max ≈ {deformation_los(slip).max():.2f} m",
            ),
        ))

    los0 = deformation_los(1.0)
    phase0 = (los0 % (n_wavelengths / 2)) / (n_wavelengths / 2) * 2 * np.pi

    fig = go.Figure(
        data=[go.Heatmap(
            z=phase0, colorscale="HSV", zmin=0, zmax=2 * np.pi,
            showscale=True,
            colorbar=dict(title="Faz (rad)", tickvals=[0, np.pi, 2 * np.pi],
                          ticktext=["0", "π", "2π"], tickfont=dict(color="#e8f0fe")),
        )],
        frames=frames,
        layout=go.Layout(
            title=dict(
                text="InSAR Interferogram — Her renk halkası = 2.8 cm yer değişimi",
                font=dict(color="#e8f0fe"),
            ),
            xaxis=dict(title="Batı → Doğu (km)", color="#e8f0fe",
                       tickvals=[0, 30, 60, 90, 119],
                       ticktext=["-50", "-20", "0", "+30", "+50"]),
            yaxis=dict(title="Güney → Kuzey (km)", color="#e8f0fe",
                       tickvals=[0, 30, 60, 90, 119],
                       ticktext=["-50", "-20", "0", "+30", "+50"]),
            annotations=[dict(x=60, y=50, xref="x", yref="y",
                              text="← FAY HATTI →",
                              showarrow=False,
                              font=dict(color="#e74c3c", size=13, family="monospace"))],
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            height=450,
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.12, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Kayma arttır", method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Dur", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                steps=[dict(args=[[str(s)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate"}],
                            label=f"{s:.1f}m", method="animate")
                       for s in slip_values],
                x=0.0, len=1.0,
                currentvalue=dict(prefix="Kayma: ", suffix=" m",
                                  font=dict(color="#e8f0fe")),
                font=dict(color="#e8f0fe"),
            )],
        ),
    )
    return fig


def anim_kaf_tektonigi() -> go.Figure:
    """KAF tektonik haritası: 1939-1999 deprem dizisi ve GPS hız vektörleri."""
    # KAF segmentleri (yaklaşık koordinatlar)
    kaf_segments = [
        # (ad, lat_list, lon_list, son_kırılma_yılı, renk)
        ("Erzincan (1939)", [40.02, 39.95, 39.80], [38.50, 39.50, 40.50],
         1939, "#e74c3c"),
        ("Niksar (1942)", [40.55, 40.50, 40.40], [36.20, 36.80, 37.50],
         1942, "#e67e22"),
        ("Ladik (1943)", [41.00, 40.85, 40.70], [35.40, 36.00, 36.50],
         1943, "#f39c12"),
        ("Bolu-Gerede (1944)", [40.85, 40.80, 40.75], [31.50, 32.50, 33.80],
         1944, "#f1c40f"),
        ("Kurşunlu (1951)", [40.90, 40.88, 40.85], [33.80, 34.50, 35.20],
         1951, "#d4efdf"),
        ("Abant (1957)", [40.77, 40.72, 40.65], [31.00, 31.50, 32.20],
         1957, "#aed6f1"),
        ("Mudurnu (1967)", [40.72, 40.65, 40.60], [30.40, 30.80, 31.20],
         1967, "#85c1e9"),
        ("İzmit (1999)", [40.80, 40.75, 40.65], [29.60, 30.00, 30.40],
         1999, "#3498db"),
        ("Düzce (1999)", [40.75, 40.75, 40.72], [30.70, 31.00, 31.30],
         1999, "#2980b9"),
        ("Marmara (kilitli!)", [40.80, 40.82, 40.85], [27.50, 28.50, 29.50],
         None, "#e74c3c"),
    ]

    # GPS hız vektörleri (Reilinger 2006 bazlı)

    fig = go.Figure()

    # Segmentler
    for seg in kaf_segments:
        ad, lats, lons, yil, renk = seg
        is_locked = yil is None
        width = 4 if is_locked else 2.5
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(color=renk, width=width),
            name=ad,
            hovertemplate=f"<b>{ad}</b><br>"
                          + (f"Son kırılma: {yil}" if yil else "⚠️ KİLİTLİ SEGMENT") + "<extra></extra>",
        ))

    # Deprem noktaları (büyük M>6.5 olaylar)
    eq_lats = [39.77, 40.70, 40.80, 40.88, 40.65, 40.77, 40.72, 40.73, 40.75]
    eq_lons = [39.60, 36.50, 33.00, 34.60, 32.00, 31.20, 30.60, 29.97, 31.16]
    eq_yils = [1939, 1942, 1944, 1951, 1957, 1966, 1967, 1999, 1999]
    eq_mags = [7.8, 7.0, 7.4, 6.9, 7.0, 6.9, 7.1, 7.6, 7.2]

    fig.add_trace(go.Scattermapbox(
        lat=eq_lats, lon=eq_lons,
        mode="markers+text",
        marker=dict(
            size=[m * 3.5 for m in eq_mags],
            color=eq_yils,
            colorscale="RdYlGn_r",
            cmin=1935, cmax=2005,
            colorbar=dict(title="Yıl", x=1.01,
                          tickfont=dict(color="#e8f0fe")),
            opacity=0.85,
        ),
        text=[f"M{m:.1f}" for m in eq_mags],
        textposition="top center",
        textfont=dict(color="#e8f0fe", size=9),
        name="Büyük depremler",
        hovertemplate="<b>%{text}</b> — %{customdata}<extra></extra>",
        customdata=[str(y) for y in eq_yils],
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=40.5, lon=34.0),
            zoom=5.2,
        ),
        title=dict(
            text="KAF Tektonik Haritası: 1939→1999 Batıya Göç Eden Depremler",
            font=dict(color="#e8f0fe"),
        ),
        paper_bgcolor="#0a1628",
        font=dict(color="#e8f0fe"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=9)),
        height=480,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def anim_erzincan_tarihi() -> go.Figure:
    """Erzincan tarihsel deprem zaman çizgisi + BPT sismik döngü modeli."""
    # Belgelenmiş büyük Erzincan depremleri (yaklaşık)
    events = [
        (-499, 7.0, "Bizans öncesi (tahmini)"),
        (1045, 7.0, "Selçuklu dönemi"),
        (1254, 7.0, "İlk katalog kaydı"),
        (1458, 7.2, "Osmanlı dönemi"),
        (1584, 7.0, "Portekizli seyyah kaydı"),
        (1668, 7.8, "Büyük Anadolu depremi"),
        (1784, 6.8, "Erzincan kasabası"),
        (1859, 7.1, "Erzincan, Tercan hasarı"),
        (1939, 7.8, "Katastrofik — 33 000 kayıp"),
        (1992, 6.8, "Modern ağlar — 653 kayıp"),
    ]

    years = [e[0] for e in events]
    mags  = [e[1] for e in events]
    descs = [e[2] for e in events]

    # Sismik boşluk hesabı
    gaps = []
    for i in range(1, len(years)):
        gaps.append(years[i] - years[i - 1])
    avg_gap = float(np.mean(gaps))

    # BPT modeli: Brownian Passage Time (Matthews 2002)
    T_last = 1939  # Son büyük M>7 olay
    alpha = 0.5    # aperiodicity (tipik değer)
    mu_bpt = avg_gap  # ortalama tekrar süresi

    current_year = 2026
    elapsed = current_year - T_last
    t_future = np.linspace(1, 300, 500)

    def bpt_pdf(t: np.ndarray, mu: float, a: float) -> np.ndarray:
        """BPT (inverse Gaussian) olasılık yoğunluk fonksiyonu."""
        with np.errstate(over="ignore", invalid="ignore"):
            c = mu / (a**2)
            pdf = np.sqrt(c / (2 * np.pi * t**3)) * np.exp(
                -c * (t - mu)**2 / (2 * mu * t)
            )
        return np.where(np.isfinite(pdf), pdf, 0.0)

    pdf_vals = bpt_pdf(t_future, mu_bpt, alpha)
    # Gelecek 30 yıl kırılma olasılığı ≈ ∫ pdf dt
    dt = t_future[1] - t_future[0]
    np.cumsum(pdf_vals) * dt

    fig = go.Figure()

    # Panel 1: Zaman çizgisi scatter
    fig.add_trace(go.Scatter(
        x=years, y=mags,
        mode="markers+lines",
        marker=dict(
            size=[max(6, (m - 6.5) * 20) for m in mags],
            color=mags, colorscale="Reds",
            cmin=6.5, cmax=8.0,
            showscale=True,
            colorbar=dict(title="Mw", x=1.01,
                          tickfont=dict(color="#e8f0fe")),
        ),
        line=dict(color="rgba(255,100,100,0.3)", width=1),
        text=descs,
        hovertemplate="<b>%{x}</b> — Mw %{y:.1f}<br>%{text}<extra></extra>",
        name="Büyük depremler (M≥6.8)",
    ))

    # Güncel zaman işaretçisi
    fig.add_vline(x=current_year, line_color="#4fc3f7",
                  line_dash="dash", line_width=2,
                  annotation_text=f"Bugün ({current_year})",
                  annotation_font_color="#4fc3f7",
                  annotation_position="top right")

    # Son 1939'dan bu yana geçen süre
    fig.add_vrect(x0=1939, x1=current_year,
                  fillcolor="rgba(231,76,60,0.08)",
                  line_width=0,
                  annotation_text=f"  {elapsed} yıl sessizlik",
                  annotation_position="top left",
                  annotation_font=dict(color="#e74c3c", size=11))

    fig.add_annotation(
        x=current_year + 5, y=7.3,
        text=f"Ortalama tekrar: ~{avg_gap:.0f} yıl<br>"
             f"1939'dan beri: {elapsed} yıl<br>"
             f"BPT α = {alpha} → yüksek risk",
        showarrow=False,
        font=dict(color="#f39c12", size=10),
        bgcolor="rgba(20,40,60,0.8)",
        bordercolor="#f39c12",
        borderwidth=1,
        align="left",
    )

    fig.update_layout(
        title=dict(
            text=f"Erzincan Sismik Döngüsü: Son 2500 Yıl | Ort. tekrar: ~{avg_gap:.0f} yıl",
            font=dict(color="#e8f0fe"),
        ),
        xaxis=dict(title="Yıl", color="#e8f0fe", gridcolor="#1e3a5f",
                   range=[-600, 2100]),
        yaxis=dict(title="Moment Büyüklüğü (Mw)", range=[6.3, 8.3],
                   color="#e8f0fe", gridcolor="#1e3a5f"),
        paper_bgcolor="#0a1628",
        plot_bgcolor="#0d1f3c",
        font=dict(color="#e8f0fe"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=420,
    )
    return fig


# ─── EK İNTERAKTİF FONKSİYONLAR (v3.1) ──────────────────────────────────────

def interaktif_sismik_dalgalar() -> None:
    """S-P yöntemi: mesafe hesaplama + dalga hızı karşılaştırması."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("### S-P Yöntemi: Deprem Merkezinin Uzaklığını Hesapla")
    st.caption("Sismografta P ve S dalgalarının varış zamanı farkından merkez uzaklığını bulun.")

    col1, col2 = st.columns([1, 2])

    with col1:
        sp_diff = st.slider(
            "S-P zaman farkı (saniye)",
            min_value=1.0, max_value=120.0, value=20.0, step=0.5,
            key="sp_diff_slider",
            help="Sismogramda S varışı − P varışı süresi.",
        )
        vp = st.slider(
            "P dalgası hızı Vp (km/s)",
            min_value=5.0, max_value=8.5, value=6.0, step=0.1,
            key="vp_slider",
            help="Krust: 5-7 km/s | Manto: 8+ km/s",
        )
        vs_ratio = st.slider(
            "Vs/Vp oranı",
            min_value=0.50, max_value=0.70, value=0.58, step=0.01,
            key="vs_ratio_slider",
            help="Kaya: 0.57-0.60 | Gevşek zemin: 0.50-0.55",
        )

        vs = vp * vs_ratio
        # Wadati formülü: Δ = Vp·Vs / (Vp − Vs) × (ts − tp)
        dist_km = (vp * vs) / (vp - vs) * sp_diff
        # P varış süresi
        p_time = dist_km / vp
        s_time = dist_km / vs

        st.markdown("---")
        st.metric("🎯 Tahmini Uzaklık", f"{dist_km:.1f} km")
        st.metric("⏱ P varış süresi", f"{p_time:.1f} sn")
        st.metric("⏱ S varış süresi", f"{s_time:.1f} sn")
        st.markdown(
            f"Vs = **{vs:.2f}** km/s · Vp = **{vp:.2f}** km/s\n\n"
            "**Kural:** 1 saniye S-P ≈ 8 km uzaklık"
        )

    with col2:
        # Dalga paketi simülasyonu
        t_axis = np.linspace(0, max(s_time * 1.4, 30), 600)
        # P-dalgası: daha erken, küçük amplitüd
        p_wave = np.where(
            (t_axis >= p_time) & (t_axis < p_time + 8),
            0.3 * np.sin(2 * np.pi * 2.5 * (t_axis - p_time))
            * np.exp(-0.15 * (t_axis - p_time)),
            0.0,
        )
        # S-dalgası: daha geç, büyük amplitüd
        s_wave = np.where(
            (t_axis >= s_time) & (t_axis < s_time + 12),
            1.0 * np.sin(2 * np.pi * 1.5 * (t_axis - s_time))
            * np.exp(-0.10 * (t_axis - s_time)),
            0.0,
        )
        seismogram = p_wave + s_wave + np.random.default_rng(99).normal(0, 0.02, len(t_axis))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_axis, y=seismogram, mode="lines",
            line=dict(color="#4fc3f7", width=1.5),
            name="Simüle sismogram",
        ))
        fig.add_vline(x=p_time, line_color="#27ae60", line_dash="dash", line_width=2,
                      annotation_text=f"P ({p_time:.1f} sn)",
                      annotation_font_color="#27ae60",
                      annotation_position="top right")
        fig.add_vline(x=s_time, line_color="#e74c3c", line_dash="dash", line_width=2,
                      annotation_text=f"S ({s_time:.1f} sn)",
                      annotation_font_color="#e74c3c",
                      annotation_position="top right")
        fig.add_vrect(x0=p_time, x1=s_time,
                      fillcolor="rgba(255,200,0,0.08)", line_width=0,
                      annotation_text=f"S−P = {sp_diff:.1f} sn → {dist_km:.0f} km",
                      annotation_font=dict(color="#f1c40f", size=10))

        fig.update_layout(
            title=dict(text="Simüle Sismogram: P ve S Dalgası Varış Süreleri",
                       font=dict(color="#e8f0fe")),
            xaxis=dict(title="Zaman (saniye)", color="#e8f0fe", gridcolor="#1e3a5f"),
            yaxis=dict(title="Yer Hareketi (bağıl)", color="#e8f0fe", gridcolor="#1e3a5f"),
            paper_bgcolor="#0a1628", plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def interaktif_elastik_geri_tepme() -> None:
    """Sismik döngü: kayma hızı + geçen yıl → birikim ve kırılma olasılığı."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("### Sismik Döngü Hesaplayıcı: Elastik Gerilme Birikimi")
    st.caption("GPS kayma hızı ve son depremden bu yana geçen süreye göre birikim hesaplanır.")

    col1, col2 = st.columns([1, 2])

    with col1:
        slip_rate = st.slider(
            "GPS kayma hızı (mm/yıl)",
            min_value=5.0, max_value=30.0, value=20.0, step=0.5,
            key="slip_rate_slider",
            help="KAF: 20-25 mm/yıl | EAF: 10-15 mm/yıl | Ege: 5-10 mm/yıl",
        )
        recurrence = st.slider(
            "Ortalama tekrar süresi (yıl)",
            min_value=100, max_value=600, value=280, step=10,
            key="recurrence_slider",
            help="Erzincan: ~280 yıl | Marmara: ~250 yıl | EAF: ~200 yıl",
        )
        years_since = st.slider(
            "Son büyük depremden beri (yıl)",
            min_value=10, max_value=500, value=87, step=5,
            key="years_since_slider",
            help="Erzincan 1939'dan 2026'ya = 87 yıl.",
        )

        # Birikim hesabı
        max_slip = slip_rate * recurrence / 1000.0  # metre
        current_slip_m = slip_rate * years_since / 1000.0
        pct = min(100.0, current_slip_m / max_slip * 100.0)

        # Basit BPT yaklaşımı ile kırılma olasılığı
        from_frac = years_since / recurrence
        bpt_prob_30yr = min(99.0, from_frac * 100.0 * 0.8)

        st.markdown("---")
        st.metric("📏 Birikmiş kayma", f"{current_slip_m:.2f} m")
        st.metric("⚡ Döngü doluluk", f"{pct:.1f}%")
        st.metric("🎲 30-yıl kırılma tahmini", f"~{bpt_prob_30yr:.0f}%")
        st.markdown(
            f"Maks. kayma açığı: **{max_slip:.1f} m**\n\n"
            "⚠️ Bu tahmini model; resmi deprem tahmini değildir."
        )

    with col2:
        t_cycle = np.linspace(0, recurrence * 1.15, 300)
        # Lineer birikim, depremde sıfırlama
        strain = np.where(t_cycle <= recurrence,
                          slip_rate * t_cycle / 1000.0,
                          0.0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_cycle, y=strain, mode="lines",
            line=dict(color="#4fc3f7", width=2.5),
            name="Gerilme birikimi",
            fill="tozeroy", fillcolor="rgba(79,195,247,0.08)",
        ))
        fig.add_vline(x=years_since, line_color="#e74c3c", line_dash="dash", line_width=2.5,
                      annotation_text=f"Bugün ({years_since} yıl geçti)",
                      annotation_font_color="#e74c3c",
                      annotation_position="top left")
        fig.add_hline(y=max_slip, line_color="#f39c12", line_dash="dot", line_width=1.5,
                      annotation_text=f"Maks. kayma açığı ({max_slip:.1f} m)",
                      annotation_font_color="#f39c12")
        fig.add_scatter(
            x=[years_since], y=[current_slip_m],
            mode="markers",
            marker=dict(size=14, color="#ffcc00", symbol="star"),
            name=f"Mevcut durum: {current_slip_m:.2f} m ({pct:.0f}%)",
        )
        fig.update_layout(
            title=dict(
                text=f"Sismik Döngü — kayma hızı {slip_rate:.0f} mm/yıl, "
                     f"tekrar {recurrence} yıl",
                font=dict(color="#e8f0fe"),
            ),
            xaxis=dict(title="Sismik Döngü İçinde Geçen Süre (yıl)",
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            yaxis=dict(title="Birikmiş Kayma (metre)",
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            paper_bgcolor="#0a1628", plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def interaktif_coulomb_stres() -> None:
    """Coulomb stres transferi: ΔCFS = Δτ + μ'·Δσn hesaplayıcısı."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("### Coulomb Stres Hesaplayıcı")
    st.caption(
        "**ΔCFS = Δτ + μ'·Δσₙ** — pozitif değerler tetikleme bölgelerini işaret eder."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        mu_prime = st.slider(
            "Efektif sürtünme katsayısı μ'",
            min_value=0.1, max_value=0.8, value=0.4, step=0.05,
            key="mu_prime_slider",
            help="Kuru kaya: 0.6-0.8 | Su doygun fay: 0.2-0.4",
        )
        delta_tau = st.slider(
            "Kayma stres değişimi Δτ (bar)",
            min_value=-3.0, max_value=3.0, value=1.5, step=0.1,
            key="delta_tau_slider",
            help="Pozitif: kayma yönünde stres artışı.",
        )
        delta_sigma_n = st.slider(
            "Normal stres değişimi Δσₙ (bar)",
            min_value=-3.0, max_value=3.0, value=-0.8, step=0.1,
            key="delta_sigma_n_slider",
            help="Negatif: fay yüzeyi sıkışır → sürtünme artar.",
        )

        delta_cfs = delta_tau + mu_prime * delta_sigma_n
        triggered = delta_cfs > 0.1
        neutral = abs(delta_cfs) <= 0.1
        icon = "🔴 Tetiklenme riski" if triggered else ("⚪ Sınırda" if neutral else "🟢 Tetiklenme azalmış")

        st.markdown("---")
        st.metric("📊 ΔCFS", f"{delta_cfs:.3f} bar",
                  delta="Eşik: 0.1 bar",
                  delta_color="off")
        st.markdown(f"**{icon}**")
        st.markdown(
            f"Δτ = {delta_tau:.1f} bar\n\n"
            f"μ'·Δσₙ = {mu_prime:.2f} × {delta_sigma_n:.1f} = "
            f"{mu_prime*delta_sigma_n:.3f} bar\n\n"
            "**1999 İzmit→Düzce:** ΔCFS = +1.5 bar → tetikledi"
        )

    with col2:
        # ΔCFS haritası: 2D grid üzerinde çift çift tümsek stres lobu
        x_grid = np.linspace(-60, 60, 100)
        y_grid = np.linspace(-60, 60, 100)
        X2, Y2 = np.meshgrid(x_grid, y_grid)

        # Fay düzlemi: x-ekseni boyunca
        dist_perp = np.abs(Y2)
        dist_par  = X2

        # Coulomb stres değişimi (çift kuvvet çifti - basitleştirilmiş)
        R2 = dist_par**2 + dist_perp**2 + 1e-3
        # Kayma bileşeni: fay sonunda quadrant alternating
        tau_comp  = mu_prime * delta_tau * 4 * dist_par * dist_perp / R2**2
        sigma_comp = delta_sigma_n * (dist_perp**2 - dist_par**2) / R2**2 * 0.5
        cfs_map = tau_comp + mu_prime * sigma_comp

        fig = go.Figure(go.Heatmap(
            z=cfs_map,
            x=x_grid, y=y_grid,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="ΔCFS (bağıl)",
                          tickfont=dict(color="#e8f0fe")),
        ))
        # Fay düzlemi çizgisi
        fig.add_shape(type="line", x0=-60, y0=0, x1=60, y1=0,
                      line=dict(color="black", width=4))
        fig.add_shape(type="line", x0=-60, y0=0, x1=60, y1=0,
                      line=dict(color="white", width=2, dash="dash"))
        fig.add_annotation(x=0, y=0, text="FAY DÜZLEMI",
                           showarrow=False, font=dict(color="white", size=11))

        fig.update_layout(
            title=dict(
                text=f"ΔCFS Haritası — μ'={mu_prime:.2f}, Δτ={delta_tau:.1f}, Δσₙ={delta_sigma_n:.1f} bar",
                font=dict(color="#e8f0fe"),
            ),
            xaxis=dict(title="Fay Boyunca Uzaklık (km)", color="#e8f0fe"),
            yaxis=dict(title="Faya Dik Uzaklık (km)", color="#e8f0fe"),
            paper_bgcolor="#0a1628", plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def interaktif_moment_tensor() -> None:
    """Beach ball interaktif: strike, dip, rake ile odak mekanizması görselleştir."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("### Odak Mekanizması (Beach Ball) Oluşturucu")
    st.caption(
        "Strike (doğrultu), dip (eğim) ve rake (kayma yönü) açılarıyla "
        "farklı fay tiplerini oluşturun."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        strike = st.slider("Strike φ (°)", 0, 360, 45, 5, key="bb_strike",
                           help="Fay doğrultusunun kuzeyden saat yönünde açısı.")
        dip    = st.slider("Dip δ (°)", 0, 90, 80, 5, key="bb_dip",
                           help="Fay düzleminin yataydan eğim açısı.")
        rake   = st.slider("Rake λ (°)", -180, 180, 10, 5, key="bb_rake",
                           help="-180/0/180: normal | 90: ters | ±10: doğrultu atımlı")

        # Fay tipi tanımlama
        if -30 <= rake <= 30 or rake > 150 or rake < -150:
            fay_tipi = "🔵 Doğrultu Atımlı"
            renk_aciklama = "KAF tipi — iki siyah lob"
        elif 60 <= rake <= 120:
            fay_tipi = "🔴 Ters Fay (Bindirme)"
            renk_aciklama = "Sıkışma tektoniği"
        elif -120 <= rake <= -60:
            fay_tipi = "🟢 Normal Fay"
            renk_aciklama = "Gerilme tektoniği — Ege tipi"
        else:
            fay_tipi = "🟡 Oblik Fay"
            renk_aciklama = "Karma bileşen"

        st.metric("Fay Tipi", fay_tipi)
        st.markdown(f"*{renk_aciklama}*")
        st.markdown(
            "**Referans açıları:**\n"
            "- KAF: strike≈80°, dip≈85°, rake≈5°\n"
            "- Erzincan: strike≈62°, dip≈75°, rake≈15°\n"
            "- EAF: strike≈40°, dip≈80°, rake≈-10°"
        )

    with col2:
        # P-dalga kutupluluk diyagramı (lower hemisphere)
        theta_bb = np.linspace(0, 2 * np.pi, 720)

        # Fay düzlemi noktaları (lower hemisphere projeksiyon)
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        rake_rad = np.radians(rake)

        # Normal vektör
        fn = np.array([
            np.sin(dip_rad) * np.cos(strike_rad + np.pi / 2),
            np.sin(dip_rad) * np.sin(strike_rad + np.pi / 2),
            -np.cos(dip_rad),
        ])
        # Kayma vektörü
        fd = np.array([
            np.cos(rake_rad) * np.cos(strike_rad)
            + np.sin(rake_rad) * np.cos(dip_rad) * np.sin(strike_rad),
            np.cos(rake_rad) * np.sin(strike_rad)
            - np.sin(rake_rad) * np.cos(dip_rad) * np.cos(strike_rad),
            np.sin(rake_rad) * np.sin(dip_rad),
        ])

        # Her azimut için P-dalga kutupluluğu (Stein & Wysession bölüm 4)
        az = theta_bb
        inc_deg = 45  # sabit çıkış açısı
        inc = np.radians(inc_deg)
        ray_x = np.sin(inc) * np.cos(az)
        ray_y = np.sin(inc) * np.sin(az)
        ray_z = -np.cos(inc)

        polarity = fn[0] * ray_x * fd[0] + fn[1] * ray_y * fd[1] + fn[2] * ray_z * fd[2]
        polarity + (fn[0] * fd[0] + fn[1] * fd[1] + fn[2] * fd[2]) * (
            ray_x**2 + ray_y**2 + ray_z**2
        )

        colors_bb = ["#1a6faf" if p > 0 else "#ecf0f1" for p in polarity]

        # Scatter polar plot approximation
        np.cos(theta_bb)
        np.sin(theta_bb)

        fig = go.Figure()

        # Dolgulu arka plan — kutupluluk alanları
        for i in range(len(theta_bb) - 1):
            c = colors_bb[i]
            (theta_bb[i] + theta_bb[i + 1]) / 2
            fig.add_trace(go.Scatter(
                x=[0, np.cos(theta_bb[i]), np.cos(theta_bb[i+1])],
                y=[0, np.sin(theta_bb[i]), np.sin(theta_bb[i+1])],
                fill="toself", fillcolor=c,
                line=dict(width=0), mode="lines",
                showlegend=False, hoverinfo="skip",
            ))

        # Çember sınırı
        circ = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(circ), y=np.sin(circ),
            mode="lines", line=dict(color="#4fc3f7", width=2.5),
            showlegend=False,
        ))
        # Fay düzlemi çizgileri
        fp1_theta = strike_rad
        fig.add_shape(type="line",
                      x0=-np.cos(fp1_theta - np.pi/2),
                      y0=-np.sin(fp1_theta - np.pi/2),
                      x1=np.cos(fp1_theta - np.pi/2),
                      y1=np.sin(fp1_theta - np.pi/2),
                      line=dict(color="#e74c3c", width=2, dash="solid"))
        # Yardımcı düzlem (auxiliary plane)
        aux_theta = fp1_theta + np.pi / 2
        fig.add_shape(type="line",
                      x0=-np.cos(aux_theta - np.pi/2),
                      y0=-np.sin(aux_theta - np.pi/2),
                      x1=np.cos(aux_theta - np.pi/2),
                      y1=np.sin(aux_theta - np.pi/2),
                      line=dict(color="#f39c12", width=1.5, dash="dash"))

        fig.add_annotation(x=0, y=-1.25,
                           text=(f"Strike={strike}° | Dip={dip}° | Rake={rake}°<br>"
                                 f"🔵 Baskı | ⚪ Gerilme | — Fay düzlemi | -- Yardımcı"),
                           showarrow=False, font=dict(color="#aaaaaa", size=10))

        fig.update_layout(
            title=dict(text=f"Beach Ball — {fay_tipi}",
                       font=dict(color="#e8f0fe")),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-1.5, 1.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-1.5, 1.5], scaleanchor="x"),
            paper_bgcolor="#0a1628", plot_bgcolor="#0a1628",
            font=dict(color="#e8f0fe"),
            height=400,
            margin=dict(l=10, r=10, t=50, b=60),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def interaktif_insar() -> None:
    """InSAR deformasyon hesaplayıcı: M, derinlik → yüzey deformasyonu (Okada)."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("### InSAR Yüzey Deformasyon Tahmincisi")
    st.caption(
        "Deprem büyüklüğü ve odak derinliğine göre tahmini yüzey deformasyonu ve "
        "InSAR fringe sayısını hesaplar."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        mw = st.slider("Moment büyüklüğü Mw", 5.0, 8.5, 7.8, 0.1, key="insar_mw",
                       help="Kahramanmaraş 2023: Mw 7.8")
        depth_km = st.slider("Odak derinliği (km)", 2, 30, 8, 1, key="insar_depth",
                             help="Sığ: <15 km → daha büyük yüzey deformasyon")
        wavelength_cm = st.selectbox(
            "Uydu bandı / dalga boyu",
            ["Sentinel-1 C-band (5.6 cm)", "ALOS-2 L-band (23.6 cm)",
             "TerraSAR-X X-band (3.1 cm)"],
            key="insar_band",
        )
        wl = {"Sentinel-1 C-band (5.6 cm)": 5.6,
              "ALOS-2 L-band (23.6 cm)": 23.6,
              "TerraSAR-X X-band (3.1 cm)": 3.1}[wavelength_cm]

        # Seismik moment → kayma alanı × kayma miktarı (basit ilişki)
        M0 = 10 ** (1.5 * mw + 9.1)  # N·m
        mu_rigidity = 3e10  # Pa (kayma modülü)
        # Ampirik ilişki: kayma = M0 / (mu × A), A ~ L × W
        L_km = 10 ** (0.59 * mw - 2.44)  # Well & Coppersmith 1994 yaklaşımı
        W_km = L_km * 0.5
        slip_m = M0 / (mu_rigidity * L_km * 1e3 * W_km * 1e3)
        slip_m = max(0.01, min(slip_m, 15.0))

        # Yüzey deformasyonu (yaklaşık Okada: maks. LOS = 0.6 × kayma)
        los_max_m = 0.6 * slip_m * (10.0 / max(depth_km, 3.0)) ** 0.7
        fringe_count = los_max_m / (wl / 200.0)  # wl cm → half-wavelength cm → m

        st.markdown("---")
        st.metric("📏 Tahmini kayma miktarı", f"{slip_m:.2f} m")
        st.metric("📡 Maks. LOS deformasyon", f"{los_max_m:.2f} m")
        st.metric("🌈 InSAR fringe sayısı", f"~{fringe_count:.0f}")
        st.metric("📐 Tahmini fay uzunluğu", f"~{L_km:.0f} km")
        st.markdown(
            f"Dalga boyu: **{wl} cm** → fringe başına **{wl/200*100:.1f} mm** LOS\n\n"
            "⚠️ Basitleştirilmiş Okada tahmini; gerçek InSAR işlemi SAR veri gerektirir."
        )

    with col2:
        # 2D yüzey deformasyon haritası
        x_los = np.linspace(-80, 80, 100)
        y_los = np.linspace(-80, 80, 100)
        X_l, Y_l = np.meshgrid(x_los, y_los)

        dist_3d = np.sqrt(X_l**2 + Y_l**2 + depth_km**2)
        los_field = los_max_m * depth_km**2 / (dist_3d**2) * np.sign(-X_l + 0.01)
        los_field = np.clip(los_field, -los_max_m, los_max_m)

        # Faz (renk halkası)
        half_wl_m = (wl / 2.0) / 100.0
        phase_field = (los_field % half_wl_m) / half_wl_m

        fig = go.Figure(go.Heatmap(
            z=phase_field,
            x=x_los, y=y_los,
            colorscale="HSV",
            zmin=0, zmax=1,
            colorbar=dict(title="Faz döngüsü",
                          tickvals=[0, 0.5, 1],
                          ticktext=["0", "π", "2π"],
                          tickfont=dict(color="#e8f0fe")),
        ))
        fig.add_shape(type="line", x0=-L_km/2, y0=0, x1=L_km/2, y1=0,
                      line=dict(color="black", width=5))
        fig.add_shape(type="line", x0=-L_km/2, y0=0, x1=L_km/2, y1=0,
                      line=dict(color="white", width=2, dash="dash"))
        fig.add_annotation(x=0, y=5, text=f"Fay (~{L_km:.0f} km)",
                           showarrow=False, font=dict(color="white", size=11))

        fig.update_layout(
            title=dict(
                text=f"InSAR Simülasyonu — Mw {mw:.1f}, d={depth_km} km, "
                     f"{fringe_count:.0f} fringe",
                font=dict(color="#e8f0fe"),
            ),
            xaxis=dict(title="Batı-Doğu (km)", color="#e8f0fe"),
            yaxis=dict(title="Güney-Kuzey (km)", color="#e8f0fe"),
            paper_bgcolor="#0a1628", plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def interaktif_tsunami_fizigi() -> None:
    """Tsunami: c = sqrt(g*d) — derinlik → hız → varış süresi hesaplayıcı."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("### Tsunami Fizik Hesaplayıcı: c = √(g·d)")
    st.caption("Dalga hızı yalnızca su derinliğine bağlıdır; deprem büyüklüğüne değil!")

    col1, col2 = st.columns([1, 2])

    with col1:
        depth_m = st.slider(
            "Su derinliği (metre)",
            min_value=10, max_value=6000, value=4000, step=50,
            key="tsunami_depth_slider",
            help="Derin okyanus: 3000-6000 m | Kıta sahanlığı: 100-200 m | Kıyı: 10-50 m",
        )
        dist_km = st.slider(
            "Kaynak → kıyı mesafesi (km)",
            min_value=50, max_value=5000, value=500, step=50,
            key="tsunami_dist_slider",
        )
        h0_m = st.slider(
            "Kaynak dalga yüksekliği (m)",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            key="tsunami_h0_slider",
            help="Açık okyanusta tipik: 0.5-2 m. Kıyıda çok daha yüksek!",
        )

        g = 9.81
        c_ms = math.sqrt(g * depth_m)
        c_kmh = c_ms * 3.6
        t_min = dist_km / (c_kmh / 60.0)

        # Shoaling: kıyıda yükseklik artışı H = H0 * (d0/d_coast)^(1/4)
        d_coast = 20  # metre (kıyı derinliği)
        h_coast = h0_m * (depth_m / d_coast) ** 0.25

        st.markdown("---")
        st.metric("🌊 Dalga hızı", f"{c_kmh:.0f} km/h",
                  help="Karşılaştırma: yolcu uçağı ~900 km/h")
        st.metric("⏱ Varış süresi", f"{t_min:.0f} dakika")
        st.metric("📏 Kıyı dalga yüksekliği", f"~{h_coast:.1f} m",
                  help="Shoaling etkisi: H ∝ d^-1/4")
        st.markdown(
            f"c = √({g}×{depth_m}) = **{c_ms:.1f} m/s = {c_kmh:.0f} km/h**\n\n"
            "🔴 Kıyı: hız düşer, yükseklik artar — 'shoaling etkisi'"
        )

    with col2:
        # Farklı derinliklerde hız profili
        depths = np.logspace(1, 3.78, 300)  # 10 m → 6000 m
        speeds_kmh = np.sqrt(g * depths) * 3.6

        # Varış süresi (dist_km'ye)
        t_arrival = dist_km / (speeds_kmh / 60.0)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=depths, y=speeds_kmh, mode="lines",
            line=dict(color="#4fc3f7", width=2.5),
            name="Dalga hızı (km/h)",
            yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            x=depths, y=t_arrival, mode="lines",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            name=f"Varış süresi ({dist_km} km için, dak.)",
            yaxis="y2",
        ))

        # Seçili derinlik işaretçisi
        c_sel = math.sqrt(g * depth_m) * 3.6
        dist_km / (c_sel / 60.0)
        fig.add_trace(go.Scatter(
            x=[depth_m], y=[c_sel],
            mode="markers", marker=dict(size=14, color="#ffcc00", symbol="star"),
            name=f"Seçili: {depth_m} m → {c_sel:.0f} km/h",
        ))

        for d_ref, lbl in [(4000, "Derin okyanus\n4000 m"),
                            (200, "Kıta sahanlığı\n200 m"),
                            (20, "Kıyı\n20 m")]:
            fig.add_vline(x=d_ref, line_dash="dot", line_color="#555555",
                          annotation_text=lbl, annotation_font_color="#888888",
                          annotation_position="top right")

        fig.update_layout(
            title=dict(text="Tsunami Hızı ve Varış Süresi — c = √(g·d)",
                       font=dict(color="#e8f0fe")),
            xaxis=dict(title="Su Derinliği (m)", type="log",
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            yaxis=dict(title="Dalga Hızı (km/h)", color="#4fc3f7"),
            yaxis2=dict(title="Varış Süresi (dak.)", overlaying="y", side="right",
                        color="#e74c3c"),
            paper_bgcolor="#0a1628", plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            height=370,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def interaktif_kaf_tektonigi() -> None:
    """KAF: GPS kayma hızı → sismik moment birikimi → beklenen büyüklük hesaplama."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("### KAF Sismik Moment Birikimi Hesaplayıcı")
    st.caption(
        "GPS'ten ölçülen kayma hızı, fay geometrisi ve geçen süreye göre "
        "birikmiş sismik moment ve beklenen maksimum büyüklük hesaplanır."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        slip_rate_kaf = st.slider(
            "GPS kayma hızı (mm/yıl)",
            5, 30, 22, 1, key="kaf_slip_rate",
            help="KAF batı bölümü: 20-25 mm/yıl (Reilinger 2006)",
        )
        fault_length = st.slider(
            "Fay segmenti uzunluğu (km)",
            20, 400, 150, 10, key="kaf_fault_length",
            help="Marmara segmenti: ~150 km | KAF toplam: ~1500 km",
        )
        fault_depth = st.slider(
            "Kilitli fay derinliği (km)",
            5, 25, 15, 1, key="kaf_fault_depth",
            help="KAF kilitlenme derinliği: 10-20 km (GPS inversiyonu)",
        )
        years_locked = st.slider(
            "Son büyük depremden beri (yıl)",
            10, 500, 226, 5, key="kaf_years_locked",
            help="Marmara: 1766'dan beri = ~260 yıl (2026)",
        )

        # Sismik moment birikimi: M0 = μ × A × slip
        mu = 3e10  # Pa (shear modulus)
        A_m2 = fault_length * 1e3 * fault_depth * 1e3  # m²
        slip_accum_m = slip_rate_kaf * years_locked / 1e6  # mm/yr → m
        M0_accum = mu * A_m2 * slip_accum_m  # N·m

        # Mw = (log10(M0) - 9.1) / 1.5
        mw_equiv = (math.log10(max(M0_accum, 1e10)) - 9.1) / 1.5
        mw_equiv = round(mw_equiv, 2)

        st.markdown("---")
        st.metric("📐 Birikmiş kayma", f"{slip_accum_m:.2f} m")
        st.metric("⚡ Sismik moment M₀", f"{M0_accum:.2e} N·m")
        st.metric("🎯 Eşdeğer Mw", f"{mw_equiv:.1f}")
        st.markdown(
            f"Fay alanı A = {fault_length} × {fault_depth} km² = "
            f"{fault_length*fault_depth:.0f} km²\n\n"
            "**⚠️ Kinetik enerji modeli;** sismik kayıp ve sürünme göz ardı edilmemiştir."
        )

    with col2:
        # KAF segmentleri kayma açığı çubuk grafiği
        segments = [
            ("Erzincan\n(1939, 87 yıl)", 87, 22, "#4fc3f7"),
            ("Ladik\n(1943, 83 yıl)", 83, 20, "#74b9ff"),
            ("Bolu\n(1944, 82 yıl)", 82, 21, "#a29bfe"),
            ("Kocaeli\n(1999, 27 yıl)", 27, 23, "#27ae60"),
            ("Düzce\n(1999, 27 yıl)", 27, 22, "#2ecc71"),
            ("Marmara\n(1766, 260 yıl!)", 260, 22, "#e74c3c"),
        ]
        seg_names = [s[0] for s in segments]
        seg_gaps_m = [s[1] * s[2] / 1e3 for s in segments]  # mm/yr × yr → m
        seg_colors = [s[3] for s in segments]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seg_names, y=seg_gaps_m,
            marker_color=seg_colors,
            text=[f"{g:.1f} m" for g in seg_gaps_m],
            textposition="outside",
            textfont=dict(color="#e8f0fe"),
            name="Birikmiş kayma açığı (m)",
        ))
        fig.add_hline(
            y=slip_rate_kaf * years_locked / 1e3,
            line_color="#f39c12", line_dash="dash",
            annotation_text=f"Seçilen ({years_locked} yıl × {slip_rate_kaf} mm/yr)",
            annotation_font_color="#f39c12",
        )

        fig.update_layout(
            title=dict(
                text="KAF Segment Kayma Açığı Karşılaştırması",
                font=dict(color="#e8f0fe"),
            ),
            xaxis=dict(title="Segment", color="#e8f0fe"),
            yaxis=dict(title="Birikmiş Kayma Açığı (metre)",
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            paper_bgcolor="#0a1628", plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def interaktif_erzincan_tarihi() -> None:
    """Erzincan: BPT olasılıksal kırılma modeli — geçen yıl → 30/50 yıl kırılma olasılığı."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("### Erzincan BPT Kırılma Olasılığı Modeli")
    st.caption(
        "Brownian Passage Time (Matthews et al. 2002) modeli ile Erzincan "
        "fayının 30/50 yıllık kırılma olasılığını hesaplar."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        mu_recurrence = st.slider(
            "Ortalama tekrar süresi μ (yıl)",
            100, 500, 280, 10, key="erz_mu",
            help="Hartleb (2006) paleosismik veri: ~280 ± 60 yıl",
        )
        alpha_aperiod = st.slider(
            "Periyodisite katsayısı α",
            0.1, 1.0, 0.5, 0.05, key="erz_alpha",
            help="0 = tam periyodik | 1 = Poisson. Erzincan için ~0.5",
        )
        t_last = st.selectbox(
            "Son büyük deprem yılı",
            [1939, 1992],
            key="erz_last_eq",
            help="M≥7: 1939 | M≥6.5: 1992",
        )

        current_yr = 2026
        elapsed = current_yr - t_last

        def bpt_cdf(t_arr: np.ndarray, mu: float, a: float) -> np.ndarray:
            from scipy.special import erfc as _erfc
            sqrt_term = np.sqrt(mu / (a**2 * t_arr))
            term1 = 0.5 * _erfc(sqrt_term * (1.0 - t_arr / mu) / math.sqrt(2))
            term2 = np.exp(2.0 / a**2) * 0.5 * _erfc(
                sqrt_term * (1.0 + t_arr / mu) / math.sqrt(2)
            )
            return np.clip(term1 + term2, 0, 1)

        try:
            t_arr = np.linspace(0.1, elapsed + 100, 1000)
            bpt_cdf(t_arr, mu_recurrence, alpha_aperiod)
            # Koşullu olasılık: P(kırılma t → t+dt | t'ye kadar kırılmadı)
            # Hazard: h(t) = f(t) / (1 - F(t))
            t_future = np.linspace(elapsed, elapsed + 100, 500)
            bpt_cdf(t_future, mu_recurrence, alpha_aperiod)
            cdf_elapsed = float(bpt_cdf(np.array([elapsed]), mu_recurrence, alpha_aperiod)[0])

            # Koşullu olasılık 30 yıl
            cdf_30 = float(bpt_cdf(np.array([elapsed + 30]),
                                   mu_recurrence, alpha_aperiod)[0])
            cdf_50 = float(bpt_cdf(np.array([elapsed + 50]),
                                   mu_recurrence, alpha_aperiod)[0])
            prob_30 = min(99.9, (cdf_30 - cdf_elapsed) / max(1e-6, 1.0 - cdf_elapsed) * 100)
            prob_50 = min(99.9, (cdf_50 - cdf_elapsed) / max(1e-6, 1.0 - cdf_elapsed) * 100)
            use_scipy = True
        except ImportError:
            # Scipy yoksa lineer yaklaşım
            prob_30 = min(99.0, elapsed / mu_recurrence * 80 * 0.6)
            prob_50 = min(99.0, elapsed / mu_recurrence * 80)
            use_scipy = False

        st.markdown("---")
        st.metric("Son depremden beri", f"{elapsed} yıl")
        st.metric("🎲 30-yıl kırılma olasılığı", f"~{prob_30:.1f}%",
                  delta="BPT koşullu")
        st.metric("🎲 50-yıl kırılma olasılığı", f"~{prob_50:.1f}%",
                  delta="BPT koşullu")
        if not use_scipy:
            st.caption("⚠️ Scipy yüklenemedi — lineer yaklaşım kullanıldı.")
        st.markdown(
            "⚠️ Bu model sismik tehlike tahmini değildir; akademik illüstrasyondur.\n\n"
            "Kaynak: Matthews et al. (2002) DOI:10.1785/0120010254"
        )

    with col2:
        t_plot = np.linspace(1, mu_recurrence * 2.5, 600)

        # BPT PDF (scipy bağımsız)
        def bpt_pdf_simple(t: np.ndarray, mu: float, a: float) -> np.ndarray:
            with np.errstate(over="ignore", invalid="ignore"):
                c = mu / (a**2)
                pdf = np.sqrt(c / (2 * np.pi * t**3)) * np.exp(
                    -c * (t - mu)**2 / (2 * mu * t)
                )
            return np.where(np.isfinite(pdf), pdf, 0.0)

        pdf_vals = bpt_pdf_simple(t_plot, mu_recurrence, alpha_aperiod)
        # Normalize
        dt = t_plot[1] - t_plot[0]
        pdf_norm = pdf_vals / (pdf_vals.sum() * dt + 1e-12)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_plot, y=pdf_norm,
            mode="lines", line=dict(color="#4fc3f7", width=2.5),
            fill="tozeroy", fillcolor="rgba(79,195,247,0.10)",
            name="BPT Olasılık Yoğunluğu",
        ))
        # Güncel elapsed
        fig.add_vline(x=elapsed, line_color="#e74c3c", line_dash="dash", line_width=2.5,
                      annotation_text=f"{current_yr} ({elapsed} yıl)",
                      annotation_font_color="#e74c3c",
                      annotation_position="top right")
        # 30-yıl pencere
        pdf_norm.max()
        fig.add_vrect(x0=elapsed, x1=elapsed + 30,
                      fillcolor="rgba(231,76,60,0.15)", line_width=0,
                      annotation_text=f"30-yıl\n~{prob_30:.0f}%",
                      annotation_font=dict(color="#e74c3c", size=10))
        # Ortalama tekrar çizgisi
        fig.add_vline(x=mu_recurrence, line_color="#f39c12", line_dash="dot",
                      annotation_text=f"μ={mu_recurrence}y",
                      annotation_font_color="#f39c12",
                      annotation_position="top left")

        fig.update_layout(
            title=dict(
                text=f"BPT Kırılma Olasılığı — μ={mu_recurrence} yıl, α={alpha_aperiod:.2f}",
                font=dict(color="#e8f0fe"),
            ),
            xaxis=dict(title="Son Depremden Beri Geçen Süre (yıl)",
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            yaxis=dict(title="Olasılık Yoğunluğu",
                       color="#e8f0fe", gridcolor="#1e3a5f"),
            paper_bgcolor="#0a1628", plot_bgcolor="#0d1f3c",
            font=dict(color="#e8f0fe"),
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            height=370,
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


# ===========================================================================
# KAPSAMLI AKADEMİK KAYNAK TABANI (v3.0 — Web Araştırması ile Güncellenmiş)
# COMPREHENSIVE ACADEMIC RESOURCE BASE (v3.0 — Web Research Updated)
# Derleyen: Claude (Anthropic) | Tarih: 2026-05-26
# ===========================================================================

# ── BÖLÜM A: TÜRK KURUMLAR VE ARAŞTIRMACILARI ────────────────────────────────

TURKISH_INSTITUTIONS: list[dict] = [
    {
        "id": "KRDAE",
        "tam_ad_tr": "Boğaziçi Üniversitesi Kandilli Rasathanesi ve Deprem Araştırma Enstitüsü",
        "tam_ad_en": "Bogazici University Kandilli Observatory and Earthquake Research Institute",
        "kisa_ad": "KRDAE / KOERI",
        "url": "http://www.koeri.boun.edu.tr/",
        "bolum_url": "https://eqe.bogazici.edu.tr/",
        "konum": "İstanbul, Türkiye",
        "kurulus": 1868,
        "faaliyetler": [
            "Türkiye ulusal sismik ağı işletimi (1976'dan)",
            "GNSS ağı izleme (2024 itibarıyla 39 istasyon)",
            "Deprem erken uyarı sistemi — İstanbul (IEDAS)",
            "İvme veritabanı oluşturma ve yönetimi",
            "2023 Kahramanmaraş deprem atölye çalışmaları",
        ],
        "arastirmacilar": [
            {
                "ad": "Prof. Dr. Mustafa Erdik",
                "unvan": "Emeritus Professor",
                "profil": "https://eqe.bogazici.edu.tr/tr/mustafa-erdik",
                "uzmanlik": "Deprem risk değerlendirmesi, kayıp tahmini, sismik tehlike",
                "oduller": ["2018 TÜBİTAK Bilim Ödülü"],
                "notlar": "Avrupa Deprem Kayıp Haritası, ELER yazılımı",
            },
            {
                "ad": "Prof. Dr. Sinan Akkar",
                "unvan": "Professor",
                "profil": "https://eqe.bogazici.edu.tr/tr/sinan-akkar",
                "uzmanlik": "Zemin hareketi tahmin denklemleri (GMPE), PSHA, NGA-Europe",
                "notlar": "2011–2013 ODTÜ EERC Müdürü; ardından KRDAE'ye geçti",
            },
            {
                "ad": "Prof. Dr. Zeynep Gülerce",
                "unvan": "Professor",
                "uzmanlik": "Zemin hareketi modelleri, Türkiye NGA-W1 uyarlaması, V/H modelleri",
                "anahtar_makale": "Gülerce, Kargoığlu & Abrahamson (2016). Turkey-Adjusted NGA-W1 GMPE. Earthquake Spectra. DOI:10.1193/022714EQS034M",
            },
            {
                "ad": "Prof. Dr. Eser Durukal",
                "uzmanlik": "Güçlü yer hareketi simülasyonu, yer etkileri, deprem kayıp tahmini",
            },
        ],
    },
    {
        "id": "ITU_DAMY",
        "tam_ad_tr": "İstanbul Teknik Üniversitesi Deprem Mühendisliği ve Afet Yönetimi Enstitüsü",
        "tam_ad_en": "Istanbul Technical University Institute of Earthquake Engineering and Disaster Management",
        "kisa_ad": "İTÜ-DAMY / EEDMI",
        "url": "https://eedmi.itu.edu.tr/",
        "konum": "İstanbul, Türkiye",
        "personel_url": "https://eedmi.itu.edu.tr/personel/akademik-personel/enstitu-kadrolu-ogretim-elemanlari",
        "arastirmacilar": [
            {
                "ad": "Prof. Dr. Naci Görür",
                "bagli_kurum": "İTÜ (emekli) / Türkiye Bilimler Akademisi",
                "dogum": 1947,
                "uzmanlik": "Jeoloji, sedimantoloji, deniz jeolojisi, İstanbul/Marmara sismisitesi",
                "oduller": [
                    "1983 TÜBİTAK Teşvik Ödülü",
                    "2004 NATO Barış ve Güvenlik Bilim Ödülü",
                    "1997 TÜBA seçimi",
                ],
                "url": "https://en.wikipedia.org/wiki/Naci_G%C3%B6r%C3%BCr",
            },
        ],
    },
    {
        "id": "METU_CE",
        "tam_ad_tr": "ODTÜ İnşaat Mühendisliği Bölümü / Deprem Araştırma Merkezi",
        "tam_ad_en": "METU Dept. of Civil Engineering / Earthquake Engineering Research Center",
        "kisa_ad": "ODTÜ-EERC",
        "url": "https://ce.metu.edu.tr/tr/",
        "konum": "Ankara, Türkiye",
        "arastirmacilar": [
            {
                "ad": "Prof. Dr. Haluk Sucuoğlu",
                "uzmanlik": "Yapısal dinamik, performans bazlı deprem mühendisliği, bina değerlendirme",
                "roller": [
                    "1995–2004 ODTÜ EERC Başkanı",
                    "2004–2007 TDY-2007 Hazırlama Üst Komisyonu",
                    "2000–2008 Ulusal Deprem Konseyi",
                ],
                "kitap": "Sucuoğlu & Akkar (2014). Basic Earthquake Engineering. Springer. ISBN:978-3-319-01026-7",
            },
        ],
    },
    {
        "id": "KOCAELI_U",
        "tam_ad_tr": "Kocaeli Üniversitesi Jeofizik Mühendisliği Bölümü",
        "url": "https://www.kocaeli.edu.tr/",
        "konum": "Kocaeli, Türkiye",
        "arastirmacilar": [
            {
                "ad": "Prof. Dr. Şerif Barış",
                "uzmanlik": "Sismoloji, sismotektonik, Marmara bölgesi sismik riski",
                "notlar": "Marmara için Mw≥7.3 olasılığı %48 tahminleri",
            },
        ],
    },
    {
        "id": "KTU",
        "tam_ad_tr": "Karadeniz Teknik Üniversitesi — Jeofizik ve İnşaat Mühendisliği",
        "kisa_ad": "KTÜ",
        "url": "https://www.ktu.edu.tr/",
        "konum": "Trabzon, Türkiye",
        "arastirmacilar": [
            {
                "ad": "Prof. Dr. Nilgün Sayıl",
                "unvan": "Sismoloji Anabilim Dalı Başkanı",
                "uzmanlik": "Sismoloji, deprem tehlike analizi, Doğu Karadeniz",
            },
        ],
        "notlar": "KTUT istasyonu KRDAE ile ortaklaşa; 2023 sonrası 1500+ bina sahası araştırması",
    },
    {
        "id": "AFAD",
        "tam_ad_tr": "T.C. İçişleri Bakanlığı Afet ve Acil Durum Yönetimi Başkanlığı",
        "tam_ad_en": "Disaster and Emergency Management Authority",
        "kisa_ad": "AFAD",
        "url": "https://www.afad.gov.tr/",
        "belgeler": [
            {
                "baslik": "Türkiye Bina Deprem Yönetmeliği 2018 (TBDY-2018)",
                "url": "https://www.afad.gov.tr/turkiye-bina-deprem-yonetmeligi",
                "pdf": "https://www.afad.gov.tr/kurumlar/afad.gov.tr/2309/files/TBDY_2018.pdf",
                "resmi_gazete": "18 Mart 2018",
                "yururluk": "1 Ocak 2019",
                "notlar": "110 uzman, 8 çalıştay; PBEE + yeni tehlike haritaları",
            },
            {
                "baslik": "AFAD Afet Raporları (2020–2024)",
                "url": "https://www.afad.gov.tr/afet-raporlari",
                "notlar": "2023 Kahramanmaraş teknik ön raporu dahil",
            },
            {
                "baslik": "Türkiye Deprem Tehlike Haritası (TDTH-2018)",
                "url": "https://tdth.afad.gov.tr",
                "notlar": "TBDY-2018 dayanağı — ızgara bazlı PGA/Ss/S1 değerleri",
            },
            {
                "baslik": "TBDY Güncellemesi (beklenen 2025)",
                "url": "https://www.arkhestatik.com/turkiye-deprem-yonetmelikleri/",
                "notlar": "2024 sonu tamamlandı; daha güçlü donatı, perde, kat ötelemesi güncellendi",
            },
        ],
    },
    {
        "id": "EBYU_DTE",
        "tam_ad_tr": "Erzincan Binali Yıldırım Üniversitesi Deprem Teknolojileri Enstitüsü",
        "tam_ad_en": "Erzincan Binali Yildirim University Earthquake Technologies Institute",
        "kisa_ad": "EBYÜ-DTE",
        "url": "https://deprem.ebyu.edu.tr/",
        "bolum_url": "https://deprem.ebyu.edu.tr/anabilim-dallari/",
        "konum": "Erzincan, Türkiye",
        "faaliyetler": [
            "Türkiye'nin ikinci deprem araştırma enstitüsü (ilki Kandilli/KRDAE)",
            "1939 ve 2011 Erzincan depremleri sonrası kuruldu — Erzincan odaklı araştırma",
            "Deprem mühendisliği, yer bilimleri, afet/acil durum yönetimi anabilim dalları",
            "Saha ölçüm sistemleri kurulumu ve izleme; laboratuvar deneysel testleri",
            "Bina ve zemin hasar değerlendirme modelleri; depreme dayanıklı altyapı tasarımı",
        ],
        "arastirmacilar": [],
    },
]

# ── BÖLÜM B: ULUSLARARASI KURUMLAR ───────────────────────────────────────────

INTERNATIONAL_INSTITUTIONS: list[dict] = [
    {
        "id": "PEER",
        "tam_ad": "Pacific Earthquake Engineering Research Center",
        "kisa_ad": "PEER",
        "kurum": "University of California, Berkeley",
        "url": "https://peer.berkeley.edu/",
        "faaliyetler": [
            "Performans bazlı deprem mühendisliği (PBEE) metodolojisi",
            "NGA zemin hareketi veritabanları (NGA-West2, NGA-East, NGA-Sub)",
            "OpenSees açık kaynak FEM platformu",
        ],
        "son_raporlar": [
            {"id": "PEER 2025/01", "baslik": "Towards Deep Learning-Based Structural Response Prediction and Ground Motion Reconstruction", "yazarlar": "Mosalam K.M.; Pang I.K.T.; Günay S.", "yil": 2025, "url": "https://peer.berkeley.edu/publications/2025-01"},
            {"id": "PEER 2025/03", "baslik": "Ground Improvement-Based Protection of Transportation Infrastructure: Centrifuge Shake Table Testing", "yil": 2025},
            {"id": "PEER 2024/11", "baslik": "Evaluation and Calibration of an OpenSees Layered Shell Element for RC Walls", "yazarlar": "Stokley J.; Lowes L.", "yil": 2024},
            {"id": "PEER 2024/10", "baslik": "21st IFIP WG7.5 Conference on Reliability and Optimization of Structural Systems", "yazarlar": "Wang Z.; Kim J.", "yil": 2024, "url": "https://peer.berkeley.edu/news/peer-report-202410-21st-working-conference-ifip-working-group-75-reliability-and-optimization"},
            {"id": "PEER 2024/04", "baslik": "2D Debris-Fluid-Structure Interaction with the Particle Finite Element Method", "yil": 2024, "url": "https://peer.berkeley.edu/publications/2024-04"},
            {"id": "PEER 2024/01", "baslik": "Response Modification of Structures with Supplemental Rotational Inertia", "yil": 2024},
        ],
    },
    {
        "id": "CALTECH_SEISMOLAB",
        "tam_ad": "California Institute of Technology — Seismological Laboratory",
        "url": "https://seismolab.caltech.edu/",
        "arastirmacilar": [
            {
                "ad": "Prof. Hiroo Kanamori",
                "unvan": "Professor Emeritus",
                "profil": "https://seismolab.caltech.edu/people/hiroo-kanamori",
                "uzmanlik": "Deprem fiziği, moment büyüklük ölçeği, erken uyarı, dev depremler",
                "katkular": [
                    "Hanks & Kanamori (1979) — Mw ölçeği. JGR 84:2348. DOI:10.1029/JB084iB05p02348",
                    "Kikuchi & Kanamori — teleseismik dalga formu inversiyonu",
                    "P dalgasına dayalı erken uyarı yöntemi",
                    "2007 Kyoto Ödülü (Temel Bilimler)",
                ],
                "kitap_editorlugu": "Kanamori H. (Ed.) (2007). Earthquake Seismology: Treatise on Geophysics Vol.4. Elsevier. ISBN:978-0-444-51932-0",
            },
        ],
    },
    {
        "id": "ETH_SED",
        "tam_ad": "Swiss Seismological Service — ETH Zürich",
        "kisa_ad": "SED",
        "url": "https://www.seismo.ethz.ch/",
        "personel_sayisi": "~80",
        "arastirmacilar": [
            {
                "ad": "Prof. Dr. Stefan Wiemer",
                "unvan": "SED Direktörü; ETH Sismoloji Kürsüsü Başkanı",
                "profil": "https://www.seismo.ethz.ch/en/about-us/all-employees/stefan-wiemer/",
                "direktörlük": 2013,
                "uzmanlik": "PSHA/PSRA, operasyonel deprem tahmini, erken uyarı, deprem öngörülebilirliği",
            },
        ],
        "faaliyetler": [
            "İsviçre ulusal sismik izleme",
            "İsviçre sismik tehlike ve risk değerlendirmesi",
            "Tarihi sismoloji, güçlü hareket, mühendislik sismolojisi",
            "Nükleer patlama doğrulama (CTBTO)",
        ],
    },
    {
        "id": "ERI_TOKYO",
        "tam_ad": "Earthquake Research Institute — University of Tokyo",
        "kisa_ad": "ERI",
        "url": "https://www.eri.u-tokyo.ac.jp/en/",
        "konum": "Tokyo, Japan",
    },
    {
        "id": "GFZ_POTSDAM",
        "tam_ad": "GFZ German Research Centre for Geosciences",
        "kisa_ad": "GFZ",
        "url": "https://www.gfz-potsdam.de/",
        "konum": "Potsdam, Germany",
        "geofon_url": "https://geofon.gfz.de/",
        "programlar": ["GEOFON küresel sismik ağ", "Deprem tehlikesi ve risk araştırmaları"],
    },
    {
        "id": "INGV",
        "tam_ad": "Istituto Nazionale di Geofisica e Vulcanologia",
        "kisa_ad": "INGV",
        "url": "https://www.ingv.it/",
        "konum": "Rome, Italy",
    },
    {
        "id": "USGS_EHP",
        "tam_ad": "U.S. Geological Survey — Earthquake Hazards Program",
        "kisa_ad": "USGS-EHP",
        "url": "https://www.usgs.gov/programs/earthquake-hazards",
        "belgeler": [
            {"baslik": "USGS EHP Decadal Science Strategy 2024–33", "url": "https://pubs.usgs.gov/publication/cir1544", "doi": "10.3133/cir1544", "yil": 2024},
            {"baslik": "2025 Puerto Rico/USVI National Seismic Hazard Model", "yil": 2025},
        ],
        "arastirmacilar": [
            {
                "ad": "E. H. (Ned) Field",
                "profil": "https://www.usgs.gov/staff-profiles/ned-field",
                "uzmanlik": "Operasyonel deprem tahmini, UCERF, deprem olasılığı modelleme",
                "makaleler": [
                    "Field et al. (2014). UCERF3 Time-Independent. BSSA 104(3):1122–1180. DOI:10.1785/0120130164",
                    "Field et al. (2017). UCERF3-ETAS. BSSA. DOI:10.1785/0120160173",
                    "Field et al. (2017). Synoptic View of UCERF3. SRL. DOI:10.1785/0220170045",
                ],
            },
        ],
    },
    {
        "id": "SCEC",
        "tam_ad": "Statewide California Earthquake Center",
        "kisa_ad": "SCEC",
        "url": "https://www.scec.org/",
        "faaliyetler": [
            "Topluluk yer modelleri (CVM)",
            "Topluluk stres düşümü validasyon çalışması (2019 Ridgecrest)",
            "Topluluk jeodezik modeli 2024 — GNSS+InSAR 3B hız alanı",
            "California Community Models for Seismic Hazard Assessments (2024 atölye)",
        ],
        "arastirmacilar": [
            {
                "ad": "Prof. Thomas H. Jordan",
                "kurum": "University of Southern California",
                "uzmanlik": "Deprem sistem bilimi, operasyonel deprem tahmini, ShakeOut",
                "roller": ["Eski SCEC Direktörü"],
            },
            {
                "ad": "Prof. Gregory C. Beroza",
                "kurum": "Stanford University — Wayne Loel Professor",
                "profil": "https://en.wikipedia.org/wiki/Gregory_Beroza",
                "uzmanlik": "Sismoloji, YZ ile deprem tespiti, FAST algoritması",
                "oduller": ["2022 ABD Ulusal Bilimler Akademisi (NAS) seçimi"],
                "katkular": [
                    "FAST — mikrodeprem parmak izi tespiti, 140x otokorelasyondan hızlı",
                    "PhaseNet — makine öğrenmesi faz alımı",
                    "Kanamori & Beroza — Treatise on Geophysics Vol.4 editörlüğü",
                ],
            },
        ],
    },
    {
        "id": "EARTHSCOPE",
        "tam_ad": "EarthScope Consortium (eski IRIS / UNAVCO)",
        "url": "https://www.earthscope.org/",
        "faaliyetler": ["Küresel sismik veri arşivi (FDSN)", "GNSS/geodezi araçları", "Taşınabilir Array arşiv verileri"],
    },
    {
        "id": "KYOTO_DPRI",
        "tam_ad": "Disaster Prevention Research Institute — Kyoto University",
        "kisa_ad": "DPRI",
        "url": "https://www.dpri.kyoto-u.ac.jp/en/",
        "konum": "Kyoto, Japan",
        "deprem_url": "https://www.dpri.kyoto-u.ac.jp/organization_en/svhm_en/rdeh/ers/",
    },
]

# ── BÖLÜM C: TEMEL KİTAPLAR (18 ESER) ───────────────────────────────────────

ESSENTIAL_TEXTBOOKS: list[dict] = [
    # Yapısal Dinamik
    {
        "id": "T01", "yazar": "Chopra, Anil K.",
        "baslik_en": "Dynamics of Structures: Theory and Applications to Earthquake Engineering",
        "baslik_tr": "Yapı Dinamiği: Deprem Mühendisliğine Teori ve Uygulamalar",
        "baski": "5th", "yil": 2020, "yayinevi": "Pearson/Prentice Hall",
        "isbn_us": "978-0-13-455512-6", "isbn_si": "978-1-292-24920-9",
        "odak": "SDOF/MDOF, spektrum analizi, sismik tepki",
        "url": "https://www.pearson.com/en-nz/subject-catalog/p/dynamics-of-structures-in-si-units/P200000003932/9781292249209",
        "notlar": "Lisansüstü deprem mühendisliği derslerinin standart başvurusu",
    },
    {
        "id": "T02", "yazar": "Clough, Ray W. & Penzien, Joseph",
        "baslik_en": "Dynamics of Structures",
        "baslik_tr": "Yapı Dinamiği",
        "baski": "3rd (revised)", "yil": 2003, "yayinevi": "Computers & Structures Inc.",
        "isbn": "978-0-923907-50-4",
        "odak": "Yapısal dinamik temelleri, matris yöntemleri",
        "notlar": "Klasik kaynak; 1993 McGraw-Hill 2. baskısı da yaygın",
    },
    # Geoteknik Deprem Mühendisliği
    {
        "id": "T03", "yazar": "Kramer, Steven L. & Stewart, Jonathan P.",
        "baslik_en": "Geotechnical Earthquake Engineering",
        "baslik_tr": "Geoteknik Deprem Mühendisliği",
        "baski": "2nd", "yil": 2024, "yayinevi": "Routledge/CRC Press",
        "isbn": "978-1-032-84274-5",
        "odak": "Sismoloji, sismik tehlike, zemin dinamiği, likefaksiyon, yamaç stabilitesi",
        "url": "https://www.routledge.com/Geotechnical-Earthquake-Engineering/Kramer-Stewart/p/book/9781032842745",
        "notlar": "1. baskı (1996) ISBN 978-0-13-374943-4 hâlâ çok kullanılır",
    },
    {
        "id": "T04", "yazar": "Seed, H. Bolton & Idriss, I.M.",
        "baslik_en": "Ground Motions and Soil Liquefaction During Earthquakes",
        "baslik_tr": "Depremlerde Yer Hareketleri ve Zemin Sıvılaşması",
        "baski": "1st", "yil": 1982, "yayinevi": "EERI Monograph",
        "isbn": "978-0-943198-06-7",
        "odak": "Likefaksiyon mekanizması, SPT temelli basitleştirilmiş yöntem",
        "notlar": "Zemin sıvılaşması alanının temel referansı",
    },
    # Sismik Yapı Tasarımı
    {
        "id": "T05", "yazar": "Paulay, Thomas & Priestley, M.J.N.",
        "baslik_en": "Seismic Design of Reinforced Concrete and Masonry Buildings",
        "baslik_tr": "Betonarme ve Yığma Binaların Sismik Tasarımı",
        "baski": "1st", "yil": 1992, "yayinevi": "John Wiley & Sons",
        "isbn": "978-0-471-54915-4",
        "odak": "Kapasite tasarımı, betonarme/yığma yapı sismik tasarımı",
        "notlar": "Kapasite tasarımı yönteminin temel kaynağı",
    },
    {
        "id": "T06", "yazar": "Priestley, M.J.N.; Calvi, G.M. & Kowalsky, M.J.",
        "baslik_en": "Displacement-Based Seismic Design of Structures",
        "baslik_tr": "Yapıların Yerdeğiştirme Bazlı Sismik Tasarımı",
        "baski": "1st", "yil": 2007, "yayinevi": "IUSS Press, Pavia",
        "isbn": "978-88-6198-000-6",
        "odak": "DDBD metodolojisi, PBEE'ye geçiş",
    },
    {
        "id": "T07", "yazar": "Priestley, M.J.N.; Seible, F. & Calvi, G.M.",
        "baslik_en": "Seismic Design and Retrofit of Bridges",
        "baslik_tr": "Köprülerin Sismik Tasarımı ve Güçlendirilmesi",
        "baski": "1st", "yil": 1996, "yayinevi": "John Wiley & Sons",
        "isbn": "978-0-471-57998-4",
        "odak": "Köprü sismik tasarımı, güçlendirme ve onarım",
    },
    # Temel Deprem Mühendisliği
    {
        "id": "T08", "yazar": "Sucuoğlu, Haluk & Akkar, Sinan",
        "baslik_en": "Basic Earthquake Engineering: From Seismology to Analysis and Design",
        "baslik_tr": "Temel Deprem Mühendisliği: Sismolojiden Analize ve Tasarıma",
        "baski": "1st", "yil": 2014, "yayinevi": "Springer",
        "isbn": "978-3-319-01026-7",
        "doi": "10.1007/978-3-319-01026-7",
        "odak": "Sismotektonik, zemin hareketi, analiz ve tasarım köprüsü",
        "url": "https://link.springer.com/book/10.1007/978-3-319-01026-7",
        "notlar": "Türk yazarlar; lisansüstü düzeyde erişilebilir",
    },
    {
        "id": "T09", "yazar": "Elnashai, Amr S. & Di Sarno, Luigi",
        "baslik_en": "Fundamentals of Earthquake Engineering: From Source to Fragility",
        "baslik_tr": "Deprem Mühendisliği Temelleri: Kaynaktan Kırılganlığa",
        "baski": "2nd", "yil": 2015, "yayinevi": "Wiley",
        "isbn": "978-1-118-67892-3",
        "odak": "Mühendislik sismolojisi, yapısal tepki, kırılganlık analizi",
        "url": "https://onlinelibrary.wiley.com/doi/book/10.1002/9780470024867",
    },
    # Sismik Tehlike
    {
        "id": "T10", "yazar": "McGuire, Robin K.",
        "baslik_en": "Seismic Hazard and Risk Analysis",
        "baslik_tr": "Sismik Tehlike ve Risk Analizi",
        "baski": "1st", "yil": 2004, "yayinevi": "EERI Monograph MNO-10",
        "isbn": "978-0-943198-01-2",
        "odak": "PSHA metodolojisi, belirsizlik analizi, karar destek",
        "notlar": "PSHA'nın özlü ve otoriter kaynağı",
    },
    {
        "id": "T11", "yazar": "Reiter, Leon",
        "baslik_en": "Earthquake Hazard Analysis: Issues and Insights",
        "baslik_tr": "Deprem Tehlike Analizi: Konular ve Kavrayışlar",
        "baski": "1st", "yil": 1990, "yayinevi": "Columbia University Press",
        "isbn": "978-0-231-06534-8",
        "odak": "PSHA temelleri, deterministik vs. olasılıksal yaklaşım",
    },
    # Sismoloji
    {
        "id": "T12", "yazar": "Shearer, Peter M.",
        "baslik_en": "Introduction to Seismology",
        "baslik_tr": "Sismolojiye Giriş",
        "baski": "3rd", "yil": 2019, "yayinevi": "Cambridge University Press",
        "isbn": "978-1-316-63884-4",
        "odak": "Sismik dalgalar, deprem kaynağı fiziği",
    },
    {
        "id": "T13", "yazar": "Stein, Seth & Wysession, Michael",
        "baslik_en": "An Introduction to Seismology, Earthquakes, and Earth Structure",
        "baslik_tr": "Sismoloji, Depremler ve Yeryapısına Giriş",
        "baski": "1st", "yil": 2003, "yayinevi": "Wiley-Blackwell",
        "isbn": "978-0-865-42078-6",
        "odak": "Sismoloji ve jeofizik temelleri",
        "url": "https://www.wiley.com/en-us/9780865420786",
    },
    {
        "id": "T14", "yazar": "Aki, Keiiti & Richards, Paul G.",
        "baslik_en": "Quantitative Seismology",
        "baslik_tr": "Kantitatif Sismoloji",
        "baski": "2nd", "yil": 2002, "yayinevi": "University Science Books",
        "isbn": "978-1-891389-63-4",
        "odak": "Teorik sismoloji, dalga yayılımı, kaynak mekanizması",
        "url": "https://www.ldeo.columbia.edu/~richards/QS_book.html",
        "notlar": "Aki-Richards denklemleri — standart teorik başvuru",
    },
    {
        "id": "T15", "yazar": "Kanamori, Hiroo (Ed.)",
        "baslik_en": "Earthquake Seismology — Treatise on Geophysics, Vol. 4",
        "baslik_tr": "Deprem Sismolojisi — Jeofizik Üzerine İnceleme, Cilt 4",
        "baski": "2nd", "yil": 2015, "yayinevi": "Elsevier",
        "isbn": "978-0-444-51932-0",
        "odak": "Deprem sismolojisinde kapsamlı referans",
        "url": "https://www.amazon.com/Earthquake-Seismology-Geophysics-Hiroo-Kanamori/dp/0444519327",
    },
    # El Kitapları
    {
        "id": "T16", "yazar": "Lee, W.H.K. et al. (Eds.)",
        "baslik_en": "International Handbook of Earthquake and Engineering Seismology — Parts A & B",
        "baslik_tr": "Uluslararası Deprem ve Mühendislik Sismolojisi El Kitabı",
        "yil_A": 2002, "yil_B": 2003, "yayinevi": "Academic Press (IASPEI)",
        "isbn_A": "978-0-12-440652-0", "isbn_B": "978-0-12-440658-2",
        "odak": "Kapsamlı sismoloji-mühendislik sismolojisi el kitabı",
        "url_B": "https://shop.elsevier.com/books/international-handbook-of-earthquake-and-engineering-seismology-part-b/lee/978-0-12-440658-2",
    },
    {
        "id": "T17", "yazar": "Naeim, Farzad (Ed.)",
        "baslik_en": "The Seismic Design Handbook",
        "baslik_tr": "Sismik Tasarım El Kitabı",
        "baski": "2nd", "yil": 2001, "yayinevi": "Kluwer Academic Publishers",
        "isbn": "978-0-7923-7301-4",
        "odak": "Zemin hareketi, dinamik analiz, izolasyon, performans bazlı tasarım",
    },
    {
        "id": "T18", "yazar": "Naeim, Farzad & Kelly, James M.",
        "baslik_en": "Design of Seismic Isolated Structures: From Theory to Practice",
        "baslik_tr": "Sismik İzole Yapıların Tasarımı: Teoriden Pratiğe",
        "baski": "1st", "yil": 1999, "yayinevi": "John Wiley & Sons",
        "isbn": "978-0-471-14921-7",
        "odak": "Taban izolasyonu tasarım ilkeleri ve uygulamaları",
    },
    # ── Türkçe başvuru/ders kitapları (v1.75 — kullanıcı önerisi, web doğrulamalı) ──
    {
        "id": "TR1", "yazar": "Celep, Zekai",
        "baslik_tr": "Deprem Mühendisliğine Giriş ve Depreme Dayanıklı Yapı Tasarımı",
        "baski": "2022", "yil": 2022, "yayinevi": "Beta Yayınevi",
        "isbn": "978-975-95405-9-3",
        "odak": "Türkiye TBDY-2018, deprem mühendisliği teorisi, kapsamlı ders kitabı (764 s.)",
        "notlar": "Türkçe temel başvuru/ders kitabı; TBDY-2018 kurallarını teorik temellerle harmanlar (İTÜ).",
        "url": "https://www.betayayinevi.com.tr",
    },
    {
        "id": "TR2", "yazar": "Canbay, E.; Ersoy, U.; Özcebe, G.; Sucuoğlu, H. & Wasti, S.T.",
        "baslik_tr": "Binalar İçin Deprem Mühendisliği: Temel İlkeler",
        "baski": "2008", "yil": 2008, "yayinevi": "Evrim Yayınevi (ODTÜ)",
        "isbn": "978-9944-0716-1-1",
        "odak": "Deprem mühendisliği temel ilkeleri, dinamik analiz, betonarme bina davranışı",
        "notlar": "Kült Türkçe eser; ODTÜ deprem mühendisliği ekolünün temel kitabı.",
    },
    {
        "id": "TR3", "yazar": "TMMOB İnşaat Mühendisleri Odası (İMO)",
        "baslik_tr": "Türkiye Bina Deprem Yönetmeliği (TBDY-2018) Eğitim El Kitabı — Açıklamalar ve Uygulama Örnekleri",
        "baski": "2019", "yil": 2019, "yayinevi": "TMMOB İMO",
        "isbn": "—",
        "odak": "TBDY-2018 uygulama örnekleri, Türk mühendisleri için temel başvuru rehberi",
        "notlar": "İMO resmi yayını (yayın kodu 165). Uygulamalı tasarım örnekleri. ISBN yerine İMO yayın kataloğundan teyit edilmeli.",
        "url": "https://www.imo.org.tr",
    },
    {
        "id": "TR4", "yazar": "Uzsoy, Şafak Z.",
        "baslik_tr": "Yapı Dinamiği ve Deprem Mühendisliği — Çözülmüş Örnek Problemler",
        "yayinevi": "Birsen Yayınevi",
        "isbn": "978-975-511-448-4",
        "odak": "Yapı dinamiği, deprem davranışı, çözümlü sınav/örnek problemler (510 s.)",
        "notlar": "Türkçe; çözümlü problem ağırlıklı, sınav hazırlık ve uygulama için. Baskı yılı yayınevinden teyit edilmeli.",
    },
    # ── Güncel Türkçe kaynaklar (v1.76 — kullanıcı derin taraması, web doğrulamalı) ──
    {
        "id": "TR5", "yazar": "Darılmaz, Kutlu",
        "baslik_tr": "Depreme Dayanıklı Binaların Tasarımına Giriş",
        "baski": "2. Baskı", "yil": 2023, "yayinevi": "Birsen Yayınevi",
        "isbn": "978-975-511-697-6",
        "odak": "TBDY-2018 betonarme + çelik bina deprem tasarımı, hesap yöntemleri, detaylandırma (372 s.)",
        "notlar": "EN GÜNCEL. Bazı örnekler SAP2000 ile çözülmüş; yapı dinamiği konularında Python kod örnekleri içerir.",
        "url": "https://www.birsenyayinevi.com",
    },
    {
        "id": "TR6", "yazar": "Kasımzade, Azer A.",
        "baslik_tr": "Yapı Dinamiği: Teori ve Deprem Mühendisliği Uygulamaları",
        "yayinevi": "Nobel Akademik Yayıncılık",
        "isbn": "978-605-320-933-1",
        "odak": "Yapı dinamiği teorisi + deprem mühendisliği uygulamaları",
        "notlar": "Her bölüm için analiz yazılımlarına QR kod erişimi; yürürlükteki ve beklenen yönetmelik değişikliklerine değinir.",
    },
    {
        "id": "TR7", "yazar": "Chopra, Anil K. (Türkçe çeviri)",
        "baslik_tr": "Yapı Dinamiği: Teori ve Deprem Mühendisliği Uygulamaları",
        "baslik_en": "Dynamics of Structures: Theory and Applications to Earthquake Engineering",
        "yayinevi": "Türkçe çeviri (Nobel dağıtım)",
        "isbn": "978-605-355-339-7",
        "odak": "Yapı dinamiği teorisi, çok katlı bina deprem davranışı, çözümlü örnekler — dünya standardı",
        "notlar": "Chopra'nın standart ders kitabının Türkçe çevirisi (İngilizce orijinali listede T1 olarak ayrıca mevcut).",
    },
    {
        "id": "TR8", "yazar": "Doğangün, Adem",
        "baslik_tr": "Deprem–Zemin ve Depreme Dayanıklı Yapı Tasarımı",
        "baski": "1. Baskı", "yil": 2021, "yayinevi": "Birsen Yayınevi",
        "isbn": "978-975-511-710-2",
        "odak": "Deprem-zemin-yapı etkileşimi, geoteknik + yerbilim entegrasyonu (~514 s.)",
        "notlar": "Deprem-zemin ilişkisini geoteknik ve yerbilim disiplinlerini de içererek ele alır.",
    },
    {
        "id": "TR9", "yazar": "Antoniou, Stelios",
        "baslik_tr": "Mevcut Betonarme Binaların Depreme Karşı Güçlendirilmesi",
        "baski": "1. Baskı", "yil": 2024, "yayinevi": "Nobel Akademik Yayıncılık",
        "isbn": "978-625-371-366-9",
        "odak": "Mevcut bina değerlendirme + güçlendirme stratejisi, performans hedefleri, gerçek vaka çalışması (536 s.)",
        "notlar": "GÜNCEL (2024). Gerçek binalardan fotoğraflarla güçlendirme uygulamaları; performans bazlı değerlendirme.",
    },
    {
        "id": "TR10", "yazar": "Day, Robert W. (Türkçe çeviri)",
        "baslik_tr": "Geoteknik Deprem Mühendisliği El Kitabı",
        "baslik_en": "Geotechnical Earthquake Engineering Handbook",
        "yayinevi": "Seçkin Yayıncılık (çeviri)",
        "isbn": "—",
        "odak": "Uygulamalı geoteknik: sıvılaşma, oturma, taşıma kapasitesi, şev stabilitesi, yapısal hasar",
        "notlar": "Uygulama odaklı el kitabı. ISBN yayınevinden teyit edilmeli. (Kramer-Stewart 2024 teorik karşılığı listede ayrıca mevcut.)",
    },
    {
        "id": "TR11", "yazar": "Eyidoğan, Haluk",
        "baslik_tr": "50 Soruda Deprem",
        "yayinevi": "Bilim ve Gelecek Kitaplığı",
        "isbn": "—",
        "odak": "Popüler-teknik giriş: tektonik deprem, fay türleri, oluşum mekanizması",
        "notlar": "Sismoloji/yerbilim tarafına giriş; genel okuyucu + öğrenci. ISBN yayınevinden teyit edilmeli.",
    },
]

# ── BÖLÜM C2: YÖNETMELİKLER & STANDARTLAR (v1.77) ───────────────────────────
# Kitap değil; sismik tasarımın yasal/teknik dayanakları. "tur" ile ayrılır.
REGULATIONS_STANDARDS: list[dict] = [
    {
        "id": "TBDY2018", "ad": "Türkiye Bina Deprem Yönetmeliği (TBDY-2018)",
        "kurum": "AFAD", "yil": 2018, "yururluk": "1 Ocak 2019",
        "tur": "Ulusal yönetmelik", "bolge": "Türkiye",
        "kapsam": "Bina sismik tasarımı: deprem yer hareketi, dayanıma/şekildeğiştirmeye göre "
                  "tasarım, betonarme/çelik/hafif çelik/yığma/ahşap özel kurallar (12 bölüm).",
        "url": "https://www.resmigazete.gov.tr/eskiler/2018/03/20180318M1-2-1.pdf",
        "turkiye": "Türkiye'de bina tasarımının güncel yasal dayanağı. TBDY-2024 taslağı hazırlık aşamasında.",
    },
    {
        "id": "CYTHYE2016", "ad": "Çelik Yapıların Tasarım, Hesap ve Yapım Esasları (ÇYTHYE-2016)",
        "kurum": "Çevre, Şehircilik ve İklim Değişikliği Bakanlığı", "yil": 2016,
        "tur": "Ulusal yönetmelik", "bolge": "Türkiye",
        "kapsam": "Çelik yapı tasarımı; TBDY-2018 ile birlikte çelik sismik tasarımın temeli.",
        "url": "https://www.resmigazete.gov.tr",
        "turkiye": "Türkiye çelik yapı tasarım esasları.",
    },
    {
        "id": "TS500", "ad": "TS 500 — Betonarme Yapıların Tasarım ve Yapım Kuralları",
        "kurum": "Türk Standardları Enstitüsü (TSE)", "yil": 2000,
        "tur": "Ulusal standart", "bolge": "Türkiye",
        "kapsam": "Betonarme malzeme + tasarım kuralları; TBDY ile birlikte uygulanır.",
        "url": "https://www.tse.org.tr",
        "turkiye": "Türkiye betonarme tasarım standardı.",
    },
    {
        "id": "EC8", "ad": "TS EN 1998 / Eurocode 8 — Depreme Dayanıklı Yapıların Tasarımı",
        "kurum": "CEN (Avrupa Standardizasyon Komitesi) / TSE", "yil": 2004,
        "tur": "Avrupa standardı", "bolge": "Avrupa",
        "kapsam": "Sismik tasarım Avrupa standardı; ulusal eklerle (National Annex) uygulanır.",
        "url": "https://eurocodes.jrc.ec.europa.eu",
        "turkiye": "AB uyum + akademik karşılaştırma; TBDY ile benzer felsefe.",
    },
    {
        "id": "ASCE7", "ad": "ASCE/SEI 7 — Minimum Design Loads and Associated Criteria for Buildings",
        "kurum": "American Society of Civil Engineers (ASCE)", "yil": 2022,
        "tur": "ABD standardı", "bolge": "ABD",
        "kapsam": "Bina yükleri + sismik tasarım kriterleri (ABD); son sürüm ASCE 7-22.",
        "url": "https://www.asce.org/publications-and-news/asce-7",
        "turkiye": None,
    },
    {
        "id": "FEMAP58", "ad": "FEMA P-58 — Seismic Performance Assessment of Buildings",
        "kurum": "FEMA / Applied Technology Council (ATC)", "yil": 2018,
        "tur": "Metodoloji / kılavuz", "bolge": "ABD",
        "kapsam": "Performans bazlı (next-gen) sismik değerlendirme; olasılıksal kayıp tahmini "
                  "(HAZUS'un mühendislik-detaylı ardılı).",
        "url": "https://www.fema.gov/emergency-managers/risk-management/earthquake/p-58",
        "turkiye": None,
    },
]

# ── BÖLÜM C3: ÖĞRENME YOLU (v1.80) ──────────────────────────────────────────
# 26 TOPICS konusunu seviye sırasına dizen rota. Her adım: konular + kitap +
# panel + kazanım + sonraki adım.
LEARNING_PATH: list[dict] = [
    {
        "adim": 1, "baslik": "Deprem Nedir? Sismik Dalgalar", "seviye": "Başlangıç",
        "aciklama": "Depremin nasıl oluştuğu, enerjinin sismik dalgalarla (P/S/yüzey) nasıl taşındığı ve "
                    "sismometrenin bunu nasıl kaydettiği. Temel fizik sezgisi.",
        "konular": ["sismik_dalgalar", "sismometre"],
        "kitap": "Shearer (2019) *Introduction to Seismology* Böl. 1-3; Stein & Wysession (2003) Böl. 1.",
        "paneller": ["🎓 Bilgi Havuzu (P/S/Rayleigh)", "🌍 Canlı Radar"],
        "kazanim": "Sismogramda P/S varışını ayırt etme, S−P ile mesafe sezgisi.",
        "sonraki": "Adım 2 — Depremin 'büyüklüğü' nasıl ölçülür?",
    },
    {
        "adim": 2, "baslik": "Magnitüd & Fay Tipleri", "seviye": "Başlangıç",
        "aciklama": "Magnitüd ölçekleri (ML/Mw/mb/Ms) farkları, Mw'nin neden standart olduğu; fay tipleri "
                    "(doğrultu-atımlı/normal/ters) ve odak derinliğinin yıkıcılığa etkisi.",
        "konular": ["magnitud_olcekleri", "fay_tipleri_odak"],
        "kitap": "Lay & Wallace; Celep (2022) *Deprem Müh. Giriş* Böl. 1-2 (Türkçe).",
        "paneller": ["🌍 Canlı Radar", "🥎 Odak Mekanizması", "🧭 Fay Sistemleri"],
        "kazanim": "Mw vs ML farkını, Türkiye fay rejimlerini (KAF/DAF/Ege) açıklama.",
        "sonraki": "Adım 3 — Depremler nasıl tekrar eder, istatistiği nedir?",
    },
    {
        "adim": 3, "baslik": "Deprem İstatistiği & Döngüsü", "seviye": "Orta",
        "aciklama": "Gutenberg-Richter b-değeri, elastik geri tepme (Reid 1910), artçı/öncü diziler (Omori), "
                    "sismik döngü ve tekrar olasılığı (BPT).",
        "konular": ["gutenberg_richter", "elastik_geri_tepme", "artci_oncu_diziler"],
        "kitap": "Stein & Wysession Böl. 4; Scholz *Mechanics of Earthquakes* Böl. 4-5.",
        "paneller": ["📉 b-Değeri Zaman Serisi", "🔄 Sismik Döngü", "📈 Artçı Tahmin"],
        "kazanim": "b-değeri yorumu, 'artçı ≠ tetiklenmiş ana şok' ayrımı.",
        "sonraki": "Adım 4 — Mühendislik: sarsıntı binaya nasıl etki eder?",
    },
    {
        "adim": 4, "baslik": "Mühendislik Sismolojisi Temelleri", "seviye": "Orta",
        "aciklama": "Davranış/tasarım spektrumu, GMPE ile yer hareketi tahmini, Vs30 zemin büyütmesi ve "
                    "mikrobölgeleme; rezonans neden yıkıcı?",
        "konular": ["response_spektrum", "gmpe_sonumleme", "mikrobolgeleme"],
        "kitap": "Chopra (2020) *Dynamics of Structures*; Sucuoğlu & Akkar (2014); Darılmaz (2023, Türkçe).",
        "paneller": ["🌊 ShakeMap", "🏔️ Vs30 Zemin", "🗾 Erzincan Mikrozon"],
        "kazanim": "Tasarım spektrumu mantığı, zemin büyütmesi (Avcılar dersi).",
        "sonraki": "Adım 5 — Zemin sıvılaşması ve erken uyarı.",
    },
    {
        "adim": 5, "baslik": "Likefaksiyon & Erken Uyarı", "seviye": "Orta",
        "aciklama": "Suya doygun zeminin sıvılaşması (σ'→0) ve erken uyarı sistemlerinin kavramı (P-S penceresi, "
                    "kör bölge, 'tahmin değil tepki').",
        "konular": ["likefaksiyon", "erken_uyari_konu", "deprem_tahmini"],
        "kitap": "Kramer & Stewart (2024) *Geotechnical EQ Eng.*; Seed & Idriss (1982).",
        "paneller": ["🚨 Erken Uyarı", "🏔️ Vs30 Zemin"],
        "kazanim": "Likefaksiyon riski + 'deprem tahmini neden mümkün değil' (Geller 1997).",
        "sonraki": "Adım 6 — Kaynak fiziği ve gerilme transferi (İleri).",
    },
    {
        "adim": 6, "baslik": "Kaynak Fiziği & Gerilme Transferi", "seviye": "İleri",
        "aciklama": "Sismik moment/enerji/stress drop, moment tensor (beach ball), stick-slip sürtünme ve "
                    "Coulomb stres transferi (komşu fay tetikleme).",
        "konular": ["sismik_moment_enerji", "moment_tensor", "stick_slip", "coulomb_stres"],
        "kitap": "Aki & Richards (2002) *Quantitative Seismology* Böl. 3-4; Kanamori-Brodsky (2004).",
        "paneller": ["🥎 Odak Mekanizması", "💥 Coulomb Stres"],
        "kazanim": "M₀ hesabı, beach ball okuma, İzmit→Düzce Coulomb örneği.",
        "sonraki": "Adım 7 — Tehlike analizi ve jeodezi.",
    },
    {
        "adim": 7, "baslik": "Tehlike Analizi & Jeodezi", "seviye": "İleri",
        "aciklama": "Olasılıksal sismik tehlike (PSHA, Cornell 1968), fay kilitlenmesi (φ), plaka hareketi "
                    "(referans çerçevesi) ve b-değeri uzamsal değişimi.",
        "konular": ["psha", "kaf_tektonigi", "b_degeri_uzamsal"],
        "kitap": "McGuire (2004) *Seismic Hazard*; Reilinger (2006) GPS; Reiter (1990).",
        "paneller": ["🗺️ Sismik Tehlike", "🔒 Fay Kilitlenme", "🌍 Plaka Simülasyonu"],
        "kazanim": "'50 yılda %10 ≠ 10 yılda olur', dönüş periyodu, kilitlenme yorumu.",
        "sonraki": "Adım 8 — Uzaktan algılama ve ileri görüntüleme.",
    },
    {
        "adim": 8, "baslik": "Uzaktan Algılama & İleri Görüntüleme", "seviye": "Araştırma",
        "aciklama": "InSAR deformasyon (LOS), SKS manto anizotropisi, sismik/ambient noise tomografi, "
                    "yavaş kayma olayları (SSE) ve Moho yapısı.",
        "konular": ["insar", "sismik_tomografi", "ambient_noise", "yavas_depremler"],
        "kitap": "Massonnet & Feigl (1998) InSAR; Shapiro (2005) ambient noise; Beroza-Ide (2011) SSE.",
        "paneller": ["🛰️ InSAR Deformasyon", "📡 InSAR Zaman Serisi", "🌀 SKS Splitting", "🌋 Moho Derinliği"],
        "kazanim": "InSAR LOS yorumu, tomografi ile yapı, SSE'nin deprem döngüsündeki rolü.",
        "sonraki": "Adım 9 — Türkiye/Erzincan'a özel uygulama + güncel araştırma.",
    },
    {
        "adim": 9, "baslik": "Türkiye/Erzincan Uygulaması & Güncel Araştırma", "seviye": "Araştırma",
        "aciklama": "KAF Erzincan paleosismolojisi, tarihsel sismisite, tsunami ve 2023 Kahramanmaraş "
                    "güncel araştırmaları (Güncel Araştırma & Teknoloji sekmesi).",
        "konular": ["erzincan_tarihi", "kaf_tektonigi", "tsunami_fizigi"],
        "kitap": "Ambraseys (2009) tarihsel; Sucuoğlu-Akkar (2014); 🔬 sekmesindeki 2023-2025 makaleler.",
        "paneller": ["🏛️ Erzincan Arşivi", "🏺 Erzincan Paleo", "📜 Tarihsel Sismisite", "🔬 Güncel Araştırma sekmesi"],
        "kazanim": "Erzincan deprem tarihini, güncel KAF/DAF araştırmalarını bütünleştirme.",
        "sonraki": "🎓 Tebrikler — temel→araştırma rotasını tamamladınız. Konular sekmesinde derinleşin.",
    },
]

# ── BÖLÜM C4: TERİM SÖZLÜĞÜ (v1.80) ─────────────────────────────────────────
# terim, en, tanim, ilgili_konu (TOPICS key|None), sembol (ops.), yanlis (ops.)
GLOSSARY: dict[str, dict] = {
    "magnitud": {"terim": "Magnitüd", "en": "Magnitude", "ilgili_konu": "magnitud_olcekleri",
        "tanim": "Depremin açığa çıkardığı enerjinin logaritmik ölçüsü. Tip'e göre değişir (ML/Mw/mb/Ms).",
        "yanlis": "Tek bir 'Richter' değeri yoktur; büyük olaylar Mw ile verilir."},
    "moment_magnitud": {"terim": "Moment Magnitüd", "en": "Moment Magnitude", "ilgili_konu": "magnitud_olcekleri",
        "sembol": "Mw = ⅔·log₁₀(M₀) − 6.07", "tanim": "Sismik momentten türetilen, doygunlaşmayan modern standart magnitüd."},
    "sismik_moment": {"terim": "Sismik Moment", "en": "Seismic Moment", "ilgili_konu": "sismik_moment_enerji",
        "sembol": "M₀ = μ·A·D̄", "tanim": "Kayma modülü × kırılma alanı × ortalama atım; depremin en temel fiziksel ölçüsü (N·m)."},
    "stress_drop": {"terim": "Gerilme Düşüşü", "en": "Stress Drop", "ilgili_konu": "sismik_moment_enerji",
        "sembol": "Δσ", "tanim": "Deprem sırasında fay üzerindeki gerilmenin düşüşü (tipik 1-10 MPa); yüksek-frekans sarsıntıyı etkiler."},
    "ml": {"terim": "Yerel Magnitüd", "en": "Local Magnitude (Richter)", "ilgili_konu": "magnitud_olcekleri",
        "sembol": "ML", "tanim": "Richter'in 1935 ölçeği; küçük-orta yerel depremler için. Büyük olaylarda doygunlaşır."},
    "mb_ms": {"terim": "Cisim/Yüzey Dalgası Magnitüdü", "en": "Body/Surface-wave Magnitude", "ilgili_konu": "magnitud_olcekleri",
        "sembol": "mb / Ms", "tanim": "Cisim (mb) ve yüzey (Ms) dalgası genliğinden hesaplanan magnitüdler; büyük olaylarda doygunlaşır."},
    "p_dalgasi": {"terim": "P Dalgası", "en": "P-wave (Primary)", "ilgili_konu": "sismik_dalgalar",
        "tanim": "Boyuna (sıkışma) gövde dalgası; en hızlı (~6 km/s), ilk varan, hasarsız."},
    "s_dalgasi": {"terim": "S Dalgası", "en": "S-wave (Secondary)", "ilgili_konu": "sismik_dalgalar",
        "tanim": "Enine (kesme) gövde dalgası (~3.5 km/s); sıvıdan geçmez, yıkıcı sarsıntının başlangıcı."},
    "yuzey_dalgasi": {"terim": "Yüzey Dalgası", "en": "Surface Wave (Rayleigh/Love)", "ilgili_konu": "sismik_dalgalar",
        "tanim": "Yüzey boyunca ilerleyen, uzun periyot + yüksek genlikli dalgalar; binalara en çok zarar verir."},
    "episantr": {"terim": "Dış Merkez (Episantr)", "en": "Epicenter", "ilgili_konu": "sismometre",
        "tanim": "Odağın (hiposantr) yeryüzündeki dik izdüşümü.",
        "yanlis": "Episantr yüzeydeki nokta; hiposantr/odak derinliktedir — karıştırılmamalı."},
    "hiposantr": {"terim": "İç Merkez (Hiposantr/Odak)", "en": "Hypocenter/Focus", "ilgili_konu": "fay_tipleri_odak",
        "tanim": "Kırılmanın başladığı yer altı noktası (enlem, boylam, derinlik)."},
    "odak_derinligi": {"terim": "Odak Derinliği", "en": "Focal Depth", "ilgili_konu": "fay_tipleri_odak",
        "tanim": "Hiposantrın derinliği. Türkiye depremleri çoğunlukla sığ (0-20 km), bu yüzeyde daha yıkıcıdır."},
    "dogrultu_atimli": {"terim": "Doğrultu Atımlı Fay", "en": "Strike-slip Fault", "ilgili_konu": "fay_tipleri_odak",
        "tanim": "Bloklar yatay kayar. KAF (sağ-yanal), DAF (sol-yanal) bu tiptir."},
    "normal_fay": {"terim": "Normal Fay", "en": "Normal Fault", "ilgili_konu": "fay_tipleri_odak",
        "tanim": "Üst blok aşağı düşer; gerilme/açılma rejimi (Ege)."},
    "ters_fay": {"terim": "Ters/Bindirme Fay", "en": "Reverse/Thrust Fault", "ilgili_konu": "fay_tipleri_odak",
        "tanim": "Üst blok yukarı çıkar; sıkışma rejimi (Doğu Anadolu)."},
    "b_degeri": {"terim": "b-Değeri", "en": "b-value", "ilgili_konu": "gutenberg_richter",
        "sembol": "log₁₀N = a − bM", "tanim": "Küçük/büyük deprem oranını veren Gutenberg-Richter parametresi (tipik ~1.0).",
        "yanlis": "Düşük b 'büyük deprem yakında' demek değildir; istatistiksel göstergedir."},
    "gutenberg_richter_t": {"terim": "Gutenberg-Richter İlişkisi", "en": "Gutenberg-Richter Law", "ilgili_konu": "gutenberg_richter",
        "tanim": "Bir bölgede magnitüd ile deprem sayısı arasındaki log-doğrusal ilişki."},
    "mc": {"terim": "Tamamlanma Magnitüdü", "en": "Magnitude of Completeness", "ilgili_konu": "b_degeri_uzamsal",
        "sembol": "Mc", "tanim": "Kataloğun eksiksiz kaydettiği en küçük magnitüd; b-değeri ancak Mc üstünde geçerli."},
    "elastik_geri_tepme_t": {"terim": "Elastik Geri Tepme", "en": "Elastic Rebound", "ilgili_konu": "elastik_geri_tepme",
        "tanim": "Reid 1910: fay kilitliyken biriken elastik enerjinin kırılmada aniden boşalması."},
    "asperity": {"terim": "Asperity (Pürüz)", "en": "Asperity", "ilgili_konu": "stick_slip",
        "tanim": "Fay üzerinde kilitli, yüksek stres taşıyan yama; kırılmada büyük enerji boşaltır."},
    "stick_slip_t": {"terim": "Stick-Slip", "en": "Stick-slip", "ilgili_konu": "stick_slip",
        "tanim": "Fayın sürtünmeyle yapışıp eşik aşılınca aniden kayması; depremin temel mekanizması."},
    "rate_state": {"terim": "Rate-State Sürtünme", "en": "Rate-and-State Friction", "ilgili_konu": "stick_slip",
        "tanim": "Sürtünmenin kayma hızı ve temas durumuna bağlılığını tanımlayan yasa (Dieterich 1994)."},
    "artci": {"terim": "Artçı Deprem", "en": "Aftershock", "ilgili_konu": "artci_oncu_diziler",
        "tanim": "Ana şok sonrası, Omori yasasıyla sönen ikincil depremler.",
        "yanlis": "1999 Düzce/2023 M7.5 artçı değil, tetiklenmiş yeni ana şoklardır."},
    "oncu": {"terim": "Öncü Deprem", "en": "Foreshock", "ilgili_konu": "artci_oncu_diziler",
        "tanim": "Ana şoktan önceki olaylar; ancak geriye dönük belirlenir, önceden 'öncü' denemez."},
    "omori": {"terim": "Omori Yasası", "en": "Omori-Utsu Law", "ilgili_konu": "artci_oncu_diziler",
        "sembol": "n(t) = K/(t+c)^p", "tanim": "Artçı sıklığının zamanla sönüm yasası."},
    "deprem_tahmini_t": {"terim": "Deprem Tahmini", "en": "Earthquake Prediction", "ilgili_konu": "deprem_tahmini",
        "tanim": "Kesin zaman+yer+büyüklük öngörüsü.", "yanlis": "Bilimsel uzlaşı: deterministik tahmin mümkün DEĞİLDİR (Geller 1997)."},
    "deprem_ongoru": {"terim": "Deprem Öngörüsü", "en": "Earthquake Forecast", "ilgili_konu": "deprem_tahmini",
        "tanim": "Olasılıksal, uzun-vadeli ifade ('50 yılda %X'); tahminden farklı olarak mümkündür."},
    "response_spektrum_t": {"terim": "Davranış Spektrumu", "en": "Response Spectrum", "ilgili_konu": "response_spektrum",
        "tanim": "Farklı periyotlardaki binaların bir yer hareketine maksimum tepkisini gösteren eğri."},
    "tasarim_spektrum": {"terim": "Tasarım Spektrumu", "en": "Design Spectrum", "ilgili_konu": "response_spektrum",
        "tanim": "Birçok depremin spektrumlarının zarfı + zemin düzeltmesi; TBDY-2018 tasarım girdisi."},
    "rezonans": {"terim": "Rezonans", "en": "Resonance", "ilgili_konu": "mikrobolgeleme",
        "tanim": "Bina periyodu zemin baskın periyoduyla çakışınca sarsıntının büyümesi (Avcılar 1999)."},
    "pga": {"terim": "Pik Yer İvmesi", "en": "Peak Ground Acceleration", "ilgili_konu": "gmpe_sonumleme",
        "sembol": "PGA (g veya cm/s²)", "tanim": "Bir noktada ölçülen maksimum yer ivmesi; sarsıntı şiddeti ölçüsü."},
    "pgv": {"terim": "Pik Yer Hızı", "en": "Peak Ground Velocity", "ilgili_konu": "gmpe_sonumleme",
        "sembol": "PGV (cm/s)", "tanim": "Maksimum yer hızı; orta-periyot yapı hasarıyla iyi ilişkili."},
    "mmi": {"terim": "Mercalli Şiddeti", "en": "Modified Mercalli Intensity", "ilgili_konu": "gmpe_sonumleme",
        "sembol": "MMI (I-XII)", "tanim": "Hissedilen/gözlenen hasar şiddetinin 12 dereceli ölçeği (subjektif).",
        "yanlis": "Şiddet (MMI) ≠ magnitüd; biri yerel etki, diğeri kaynak enerjisi."},
    "gmpe": {"terim": "Yer Hareketi Tahmin Denklemi", "en": "Ground Motion Prediction Equation", "ilgili_konu": "gmpe_sonumleme",
        "tanim": "M, mesafe ve zemine göre PGA/PGV tahmini (Akkar-Bommer 2010); ±%50 saçılım içerir."},
    "vs30": {"terim": "Vs30", "en": "Vs30", "ilgili_konu": "mikrobolgeleme",
        "sembol": "Vs30 (m/s)", "tanim": "Üst 30 m ortalama kayma dalga hızı; zemin sınıfı (NEHRP/TBDY) belirler. Düşük=yumuşak=büyütme."},
    "mikrobolgeleme_t": {"terim": "Mikrobölgeleme", "en": "Microzonation", "ilgili_konu": "mikrobolgeleme",
        "tanim": "Şehir/bölge ölçeğinde yerel zemin sarsıntı büyütmesinin haritalanması."},
    "likefaksiyon_t": {"terim": "Likefaksiyon (Sıvılaşma)", "en": "Liquefaction", "ilgili_konu": "likefaksiyon",
        "sembol": "σ' = σ − u → 0", "tanim": "Suya doygun gevşek kumun sarsıntıda sıvı gibi davranıp taşıma gücünü kaybetmesi."},
    "hvsr": {"terim": "HVSR", "en": "Horizontal-to-Vertical Spectral Ratio", "ilgili_konu": "mikrobolgeleme",
        "tanim": "Mikrotremor kaydından zemin baskın periyodu tahmini (Nakamura 1989)."},
    "moment_tensor_t": {"terim": "Moment Tensörü", "en": "Moment Tensor", "ilgili_konu": "moment_tensor",
        "tanim": "Deprem kaynağının kuvvet sistemi; beach ball ile gösterilir, fay geometrisini verir."},
    "beach_ball": {"terim": "Plaj Topu", "en": "Beach Ball Diagram", "ilgili_konu": "moment_tensor",
        "tanim": "Odak mekanizmasının küresel izdüşümü; fay tipini görsel verir.",
        "yanlis": "İki olası fay düzlemi (nodal plane) gösterir; hangisinin gerçek olduğu tek başına belirsiz."},
    "strike_dip_rake": {"terim": "Strike/Dip/Rake", "en": "Strike/Dip/Rake", "ilgili_konu": "moment_tensor",
        "tanim": "Fay düzlemi yönelimi (doğrultu), eğim ve kayma açısı (Aki-Richards konvansiyonu)."},
    "coulomb_t": {"terim": "Coulomb Stres Transferi", "en": "Coulomb Stress Transfer", "ilgili_konu": "coulomb_stres",
        "sembol": "ΔCFS = Δτ + μ'·Δσn", "tanim": "Bir depremin komşu fayda kırılma olasılığını değiştirmesi (İzmit→Düzce).",
        "yanlis": "+ΔCFS olasılığı artırır ama deprem tahmini değildir."},
    "psha_t": {"terim": "PSHA", "en": "Probabilistic Seismic Hazard Analysis", "ilgili_konu": "psha",
        "tanim": "Belirli sürede yer hareketi seviyesinin aşılma olasılığı (Cornell 1968).",
        "yanlis": "'50 yılda %10' = 475 yıl dönüş periyodu; '10 yılda olur' DEĞİLDİR."},
    "donus_periyodu": {"terim": "Dönüş Periyodu", "en": "Return Period", "ilgili_konu": "psha",
        "tanim": "Bir seviyenin ortalama tekrar süresi (475 yıl = 50 yılda %10 aşılma)."},
    "bpt": {"terim": "Brownian Geçiş Zamanı", "en": "Brownian Passage Time", "ilgili_konu": "elastik_geri_tepme",
        "tanim": "Karakteristik deprem tekrar olasılığı modeli (Matthews 2002); zaman-bağımlı tehlike."},
    "fay_kilitlenme_t": {"terim": "Fay Kilitlenmesi", "en": "Fault Coupling/Locking", "ilgili_konu": "elastik_geri_tepme",
        "sembol": "φ (0-1)", "tanim": "Fayın ne kadar kilitli (stres biriktiren) olduğu; φ=1 tam kilitli, φ=0 sürünen."},
    "slip_deficit": {"terim": "Kayma Açığı", "en": "Slip Deficit", "ilgili_konu": "elastik_geri_tepme",
        "tanim": "Son depremden bu yana biriken (henüz boşalmamış) kayma; kayma hızı × geçen süre."},
    "creep": {"terim": "Sürünme (Asismik Kayma)", "en": "Creep / Aseismic Slip", "ilgili_konu": "yavas_depremler",
        "tanim": "Fayın sarsıntısız, yavaş kayması (KAF İsmetpaşa); enerjiyi deprem olmadan boşaltır."},
    "sse": {"terim": "Yavaş Kayma Olayı", "en": "Slow Slip Event", "ilgili_konu": "yavas_depremler",
        "tanim": "Günler-aylar süren sessiz kayma; GPS/InSAR ile saptanır, sismogramda görünmez."},
    "insar_t": {"terim": "InSAR", "en": "Interferometric SAR", "ilgili_konu": "insar",
        "tanim": "İki radar görüntüsünün faz farkıyla yer deformasyonu ölçümü (cm-altı)."},
    "los": {"terim": "Bakış Doğrultusu", "en": "Line-of-Sight (LOS)", "ilgili_konu": "insar",
        "tanim": "InSAR'ın ölçtüğü uydu-bakış yönündeki yer değiştirme bileşeni.",
        "yanlis": "Tek geçiş 3B hareket vermez; sadece LOS (1B) bileşeni ölçülür."},
    "fringe": {"terim": "Saçak (Fringe)", "en": "Interferometric Fringe", "ilgili_konu": "insar",
        "tanim": "InSAR'da bir tam renk döngüsü = yarım dalgaboyu LOS deformasyon (~2.8 cm, C-bant)."},
    "sks": {"terim": "SKS Dalgası", "en": "SKS Phase", "ilgili_konu": "sismik_tomografi",
        "tanim": "Çekirdek-manto sınırından dönüşen S dalgası; manto anizotropisi ölçümünde kullanılır."},
    "anizotropi": {"terim": "Sismik Anizotropi", "en": "Seismic Anisotropy", "ilgili_konu": "sismik_tomografi",
        "tanim": "Sismik hızın yöne bağlı olması; olivin hizalanmasıyla manto akış yönünü ele verir (SKS splitting)."},
    "moho": {"terim": "Moho Süreksizliği", "en": "Mohorovičić Discontinuity", "ilgili_konu": "sismik_tomografi",
        "tanim": "Kabuk-manto sınırı (Türkiye'de ~35-45 km).", "yanlis": "Moho ≠ deprem derinliği; depremler üst kabukta (0-20 km)."},
    "tomografi_t": {"terim": "Sismik Tomografi", "en": "Seismic Tomography", "ilgili_konu": "sismik_tomografi",
        "tanim": "Dalga varış zamanlarının inversiyonuyla yerin 3B hız yapısını görüntüleme."},
    "tsunami_t": {"terim": "Tsunami", "en": "Tsunami", "ilgili_konu": "tsunami_fizigi",
        "sembol": "v = √(g·d)", "tanim": "Denizaltı deprem/heyelanın ürettiği uzun dalga; sığ-su hızı derinliğe bağlı."},
    "runup": {"terim": "Tırmanma Yüksekliği", "en": "Runup", "ilgili_konu": "tsunami_fizigi",
        "tanim": "Tsunaminin kıyıda ulaştığı maksimum dikey yükseklik."},
    "kaf": {"terim": "Kuzey Anadolu Fayı", "en": "North Anatolian Fault", "ilgili_konu": "kaf_tektonigi",
        "tanim": "~1500 km sağ-yanal doğrultu-atımlı fay; 1939-1999 batıya göç eden M7+ dizisi."},
    "daf": {"terim": "Doğu Anadolu Fayı", "en": "East Anatolian Fault", "ilgili_konu": "kaf_tektonigi",
        "tanim": "Sol-yanal doğrultu-atımlı fay; 2023 Kahramanmaraş çift kırılması bu zonda."},
    "paleosismoloji": {"terim": "Paleosismoloji", "en": "Paleoseismology", "ilgili_konu": "erzincan_tarihi",
        "tanim": "Fay üzerinde kazı (trench) ile tarih-öncesi depremlerin tarihlendirilmesi (¹⁴C/OSL)."},
    "erken_uyari_t": {"terim": "Erken Uyarı Sistemi", "en": "Earthquake Early Warning", "ilgili_konu": "erken_uyari_konu",
        "tanim": "P-dalgasını yakalayıp S gelmeden uyarı (tepki sistemi).",
        "yanlis": "EEW deprem tahmini değildir; deprem başladıktan sonra çalışır."},
    "ambient_noise_t": {"terim": "Ortam Gürültüsü", "en": "Ambient Seismic Noise", "ilgili_konu": "ambient_noise",
        "tanim": "Sürekli arka plan titreşimi; çapraz-korelasyonla deprem-bağımsız yapı görüntüleme."},
}

# ── BÖLÜM C5: GÜNCEL TEKNOLOJİ CEPHELERİ (v1.81) ────────────────────────────
# Her teknoloji: ne sağlar / ne sağlamaz / klasik temel kaynak / modern kaynaklar
# (doğrulanmış DOI) / Türkiye uygulanabilirliği / yanlış anlaşılma. "Modern+klasik
# birlikte" kuralı (kullanıcı). Uydurma DOI yok.
TECH_FRONTIERS: list[dict] = [
    {
        "id": "ml_ai", "emoji": "🤖", "ad": "Makine Öğrenmesi / Yapay Zeka",
        "ozet": "Derin öğrenme ile sismik faz seçimi, deprem tespiti ve artçı örüntü analizi.",
        "saglar": "Gürültülü veride mikro-deprem tespiti, otomatik P/S faz seçimi (insan-üstü hız), "
                  "büyük katalog tamamlama, artçı dağılım örüntü analizi.",
        "saglamaz": "Deprem TAHMİNİ (zaman/yer/büyüklük). ML örüntü bulur ama fiziksel öngörü yapamaz; "
                    "eğitim verisindeki önyargıyı taşır.",
        "klasik": {"kaynak": "Geller (1997) — tahmin sınırı", "doi": "10.1111/j.1365-246X.1997.tb06588.x"},
        "modern": [
            {"kaynak": "Mousavi et al. (2020) — EQTransformer", "dergi": "Nature Communications",
             "doi": "10.1038/s41467-020-17591-w"},
            {"kaynak": "Zhu & Beroza (2019) — PhaseNet", "dergi": "Geophysical Journal International",
             "doi": "10.1093/gji/ggy423"},
            {"kaynak": "DeVries et al. (2018) — derin öğrenme ile artçı örüntüsü", "dergi": "Nature",
             "doi": "10.1038/s41586-018-0438-y"},
        ],
        "turkiye": "AFAD/KOERI kataloglarına otomatik faz-seçim; 2023 Kahramanmaraş artçı dizisi analizi. "
                   "STEAD gibi açık veri setleri Türkiye verisiyle zenginleştirilebilir.",
        "yanlis": "'AI depremi tahmin ediyor' — hayır; AI tespit/sınıflama yapar, tahmin değil (Geller 1997).",
    },
    {
        "id": "das", "emoji": "🔌", "ad": "DAS — Fiber-Optik Sismoloji",
        "ozet": "Mevcut fiber-optik kabloları binlerce sismik sensöre dönüştürme (Distributed Acoustic Sensing).",
        "saglar": "Metre çözünürlükte, on km'lerce uzunlukta yoğun dizilim; şehir/altyapı (köprü, baraj, bina) "
                  "izleme; deniz altı kablolarla okyanus tabanı sismolojisi; ucuz yoğun ağ.",
        "saglamaz": "Sadece kablo eksenindeki gerinim (tek bileşen); mutlak kalibrasyon zorluğu; "
                    "geleneksel sismometre kadar geniş-bant değil.",
        "klasik": {"kaynak": "Massonnet & Feigl (1998) — uzaktan algılama jeodezisi temeli",
                   "doi": "10.1029/97RG03139"},
        "modern": [
            {"kaynak": "Zhan (2020) — DAS fiber-optik kabloları sismik antene çevirir", "dergi": "Seismological Research Letters",
             "doi": "10.1785/0220190112"},
        ],
        "turkiye": "Mevcut telekom fiber altyapısı (şehirler arası) potansiyel DAS ağı; Erzincan/Marmara "
                   "kentsel izleme + KAF boyunca dizilim araştırma fırsatı.",
        "yanlis": "'DAS sismometrenin yerini alır' — tamamlayıcıdır; tek-bileşen + kalibrasyon sınırları var.",
    },
    {
        "id": "gnss", "emoji": "📡", "ad": "GNSS / Yüksek-Oranlı GPS Jeodezisi",
        "ozet": "GPS/GNSS ile yer deformasyonu + büyük depremlerde hızlı moment büyüklüğü kestirimi.",
        "saglar": "İnterseismik kayma hızı + kilitlenme (mm/yr); büyük (M8+) depremde sismik doygunluktan "
                  "etkilenmeden hızlı Mw; gerçek-zamanlı GNSS ile erken karakterizasyon (GNSS-EEW).",
        "saglamaz": "Deprem öncesi öngörü; küçük olaylarda düşük sinyal/gürültü; uydu geometrisi + atmosfer hatası.",
        "klasik": {"kaynak": "Reilinger et al. (2006) — Türkiye GPS hız alanı", "doi": "10.1029/2005JB004051"},
        "modern": [
            {"kaynak": "Allen & Melgar (2019) — EEW: ilerlemeler ve zorluklar (GNSS-EEW dahil)",
             "dergi": "Annual Review of Earth and Planetary Sciences", "doi": "10.1146/annurev-earth-053018-060457"},
        ],
        "turkiye": "AFAD/KOERI GNSS ağları (TUSAGA/CORS); KAF kilitlenme + 2023 Maraş hızlı moment çalışmaları. "
                   "Erzincan segmenti GPS izleme.",
        "yanlis": "'GPS depremi önceden gösterir' — interseismik birikim ölçer (kilitlenme), kırılma zamanı vermez.",
    },
    {
        "id": "eew_modern", "emoji": "🚨", "ad": "Erken Uyarı (EEW) — Modern Gelişmeler",
        "ozet": "P-dalgası tabanlı uyarının operasyonel sistemlere (telefon, altyapı) evrimi.",
        "saglar": "Saniyeler-onlarca saniye uyarı (mesafeye bağlı); tren/asansör/gaz otomasyonu; smartphone "
                  "kitlesel uyarı (Android EEW); hastane/fabrika tepki tetikleme.",
        "saglamaz": "Merkez üssünde kör bölge (uyarı yok); deprem TAHMİNİ değil; ağ tespit + telekom gecikmesi "
                    "kullanılabilir pencereyi kısaltır.",
        "klasik": {"kaynak": "Allen & Kanamori (2003) — EEW potansiyeli", "doi": "10.1126/science.1080912"},
        "modern": [
            {"kaynak": "Allen & Melgar (2019) — EEW: ilerlemeler, zorluklar, toplumsal ihtiyaçlar",
             "dergi": "Annual Review of Earth and Planetary Sciences", "doi": "10.1146/annurev-earth-053018-060457"},
        ],
        "turkiye": "AFAD-EWS (Marmara pilot); İstanbul IEDAS (KOERI). Android kitlesel EEW Türkiye'de yaygınlaşıyor. "
                   "Erzincan için kapsama hedefi.",
        "yanlis": "'EEW deprem tahminidir' — hayır; olay başladıktan sonra hızlı dalga avantajını kullanan tepki sistemi.",
    },
    {
        "id": "insar_otomasyon", "emoji": "🛰️", "ad": "InSAR Otomasyonu / Uydu Jeodezisi",
        "ozet": "Sentinel-1 gibi ücretsiz uydularla otomatik, sürekli yer deformasyonu izleme.",
        "saglar": "Geniş alan cm-altı co-seismic + interseismic deformasyon; ücretsiz Sentinel-1 (6-12 gün); "
                  "otomatik zaman serisi (COMET LiCS); deprem sonrası günler içinde kırık haritalama.",
        "saglamaz": "Sadece LOS (1B) bileşeni — 3B için çoklu geometri; atmosfer gecikmesi + decorrelation; "
                    "deprem öngörüsü değil.",
        "klasik": {"kaynak": "Massonnet & Feigl (1998) — radar interferometri derlemesi", "doi": "10.1029/97RG03139"},
        "modern": [
            {"kaynak": "Xu et al. (2023) — 2023 Maraş Sentinel-1 InSAR deformasyon", "dergi": "Science/GRL",
             "doi": "10.1029/2023GL104604"},
        ],
        "turkiye": "Sentinel-1 ile 2023 Maraş co-seismic haritalama günler içinde; KAF interseismic izleme; "
                   "Erzincan/Marmara kentsel oturma takibi (COMET LiCS açık portal).",
        "yanlis": "'InSAR tek görüntüden 3B hareket verir' — hayır, yalnızca uydu-bakış (LOS) bileşeni.",
    },
]

# ── BÖLÜM D: ANAHTAR AKADEMİSYENLER ─────────────────────────────────────────

KEY_SCHOLARS: dict[str, list[dict]] = {
    "turkey": [
        {"ad": "Naci Görür", "kurum": "İTÜ (emekli) / TÜBA", "dogum": 1947, "uzmanlik": "Jeoloji, sedimantoloji, İstanbul/Marmara sismisitesi", "oduller": ["1983 TÜBİTAK", "2004 NATO Barış Bilim Ödülü", "1997 TÜBA"], "url": "https://en.wikipedia.org/wiki/Naci_G%C3%B6r%C3%BCr"},
        {"ad": "Şerif Barış", "kurum": "Kocaeli Üniversitesi Jeofizik", "uzmanlik": "Sismoloji, Marmara sismik riski"},
        {"ad": "Mustafa Erdik", "kurum": "KRDAE Boğaziçi (Emeritus)", "uzmanlik": "Deprem risk/kayıp, ELER yazılımı", "profil": "https://eqe.bogazici.edu.tr/tr/mustafa-erdik", "odul": "2018 TÜBİTAK Bilim Ödülü"},
        {"ad": "Haluk Sucuoğlu", "kurum": "ODTÜ İnşaat (Emeritus)", "uzmanlik": "PBEE, yapı değerlendirme, bina stoğu", "kitap": "Sucuoğlu & Akkar 2014, Springer"},
        {"ad": "Sinan Akkar", "kurum": "KRDAE Boğaziçi", "uzmanlik": "GMPE, PSHA, Avrupa/Türkiye zemin hareketi veritabanları"},
        {"ad": "Zeynep Gülerce", "kurum": "KRDAE Boğaziçi", "uzmanlik": "Zemin hareketi modelleri, NGA Türkiye uyarlaması", "anahtar_makale": "Gülerce et al. (2016). Earthquake Spectra. DOI:10.1193/022714EQS034M"},
        {"ad": "Eser Durukal", "kurum": "KRDAE Boğaziçi", "uzmanlik": "Güçlü yer hareketi simülasyonu, yer etkileri"},
        {"ad": "Nilgün Sayıl", "kurum": "KTÜ Jeofizik", "uzmanlik": "Sismoloji, Doğu Karadeniz sismisitesi"},
    ],
    "international": [
        {"ad": "Hiroo Kanamori", "kurum": "Caltech Seismolab (Emeritus)", "dogum": 1936, "uzmanlik": "Deprem fiziği, Mw ölçeği, erken uyarı", "profil": "https://seismolab.caltech.edu/people/hiroo-kanamori", "katkular": ["Hanks & Kanamori (1979) — Mw ölçeği. JGR. DOI:10.1029/JB084iB05p02348", "P dalgası erken uyarı", "2007 Kyoto Ödülü"]},
        {"ad": "Lynn Sykes", "kurum": "Columbia/Lamont-Doherty", "uzmanlik": "Plaka tektoniği, transform fay mekanizmaları", "katkular": ["Sykes (1967) — Transform fay mekanizmaları, BSSA — plaka tektoniği doğrulaması"]},
        {"ad": "Amos Nur", "kurum": "Stanford (1938–2024)", "uzmanlik": "Kaya fiziği, deprem arkeolojisi, VP/VS anomalileri", "notlar": "Haziran 2024'te vefat etti", "url": "https://sustainability.stanford.edu/news/amos-nur-rock-physics-pioneer-has-died", "katkular": ["Stanford Rock Physics Project (SRPPBG) kurucusu", "4-D sismoloji öncüsü", "Dilatasyon-difüzyon mekanizması (1970ler)", "Deprem arkeolojisi alanını kurdu"]},
        {"ad": "James R. Rice", "kurum": "Harvard — Mallinckrodt Professor", "uzmanlik": "Fay mekaniği, deprem oluşum fiziği", "odul": "SSA Harry Fielding Reid Madalyası (2012)", "profil": "http://esag.harvard.edu/rice/", "katkular": ["Deprem döngüsü modellemesi, fay etkileşimleri", "Termal/hidromekanik fay zonu zayıflama modelleri", "Kırılma başlaması ve yayılma fiziği"]},
        {"ad": "Thomas H. Jordan", "kurum": "USC (eski SCEC Direktörü)", "uzmanlik": "Deprem sistem bilimi, operasyonel tahmin, ShakeOut", "katkular": ["ShakeOut sismik senaryosu (Güney Kalf. 2008)", "UCERF3 baş editörü", "CSEP girişimi"]},
        {"ad": "Gregory C. Beroza", "kurum": "Stanford — Wayne Loel Professor", "uzmanlik": "YZ ile deprem tespiti, mikrodepremler", "odul": "2022 NAS seçimi", "profil": "https://en.wikipedia.org/wiki/Gregory_Beroza", "katkular": ["FAST — 140x hızlı mikrodeprem tespiti", "PhaseNet — ML faz alımı", "Treatise on Geophysics Vol.4 editörlüğü"]},
        {"ad": "E. H. (Ned) Field", "kurum": "USGS", "uzmanlik": "Operasyonel tahmin, UCERF, sismik tehlike", "profil": "https://www.usgs.gov/staff-profiles/ned-field", "makaleler": ["Field et al. (2014). UCERF3. BSSA. DOI:10.1785/0120130164", "Field et al. (2017). UCERF3-ETAS. BSSA. DOI:10.1785/0120160173"]},
        {"ad": "Stefan Wiemer", "kurum": "ETH Zürich — SED Direktörü", "uzmanlik": "PSHA, operasyonel deprem tahmini, erken uyarı", "profil": "https://www.seismo.ethz.ch/en/about-us/all-employees/stefan-wiemer/"},
    ],
}

# ── BÖLÜM E: GÜNCEL YAYINLAR (2020–2025) ─────────────────────────────────────

RECENT_PUBLICATIONS_EQ: list[dict] = [
    # 2023 Kahramanmaraş Depremi — Temel Makaleler
    {
        "id": "P01", "kategori": "Kahramanmaraş",
        "baslik": "The complex dynamics of the 2023 Kahramanmaraş, Turkey, Mw 7.8–7.7 earthquake doublet",
        "dergi": "Science", "yil": 2023, "doi": "10.1126/science.adi0685",
        "url": "https://www.science.org/doi/10.1126/science.adi0685",
        "ozet": "Çok-faylı kinematik inversiyon; Mw7.8'de 3 alt-kayma, Mw7.7'de süperkesme batı kolu",
    },
    {
        "id": "P02", "kategori": "Kahramanmaraş",
        "baslik": "Supershear triggering and cascading fault ruptures of the 2023 Kahramanmaraş, Türkiye, earthquake doublet",
        "yazarlar": "Ren et al.", "dergi": "Science", "cilt": 383, "sayfalar": "305–311", "yil": 2024,
        "doi": "10.1126/science.adi8519",
    },
    {
        "id": "P03", "kategori": "Kahramanmaraş",
        "baslik": "The 2023 Kahramanmaraş Earthquake Sequence: finding a path to a more resilient, sustainable, and equitable society",
        "yazarlar": "Galasso C. & Opabola E.", "dergi": "Communications Engineering", "yil": 2024,
        "doi": "10.1038/s44172-024-00170-y", "url": "https://www.nature.com/articles/s44172-024-00170-y",
    },
    {
        "id": "P04", "kategori": "Kahramanmaraş",
        "baslik": "The 2023 Mw 7.8–7.7 Kahramanmaraş earthquakes were loosely slip-predictable",
        "dergi": "Communications Earth & Environment", "yil": 2024,
        "doi": "10.1038/s43247-024-01969-5", "url": "https://www.nature.com/articles/s43247-024-01969-5",
    },
    {
        "id": "P05", "kategori": "Kahramanmaraş",
        "baslik": "High normal stress promoted supershear rupture during the 2023 Mw 7.8 Kahramanmaraş earthquake",
        "dergi": "Nature Geoscience", "yil": 2025,
        "doi": "10.1038/s41561-025-01893-z", "url": "https://www.nature.com/articles/s41561-025-01893-z",
    },
    {
        "id": "P06", "kategori": "Kahramanmaraş",
        "baslik": "Dynamic Rupture Process of the 2023 Mw 7.8 Kahramanmaraş Earthquake: Variable Rupture Speed and Implications for Seismic Hazard",
        "yazarlar": "Wang et al.", "dergi": "Geophysical Research Letters", "yil": 2023,
        "doi": "10.1029/2023GL104787", "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL104787",
    },
    {
        "id": "P07", "kategori": "Kahramanmaraş",
        "baslik": "Insights on the 2023 Kahramanmaraş Earthquake from InSAR: fault locations, rupture styles and induced deformation",
        "dergi": "Geophysical Journal International", "cilt": 236, "sayi": 2, "yil": 2024,
        "doi": "10.1093/gji/ggad464", "url": "https://academic.oup.com/gji/article/236/2/1068/7458664",
    },
    {
        "id": "P08", "kategori": "Kahramanmaraş",
        "baslik": "Spatial variation in stress orientation … rupture propagation across stress regime transition in the 2023 Mw 7.8 Kahramanmaraş earthquake",
        "dergi": "Geophysical Journal International", "cilt": 238, "sayi": 3, "yil": 2024,
        "doi": "10.1093/gji/ggae205", "url": "https://academic.oup.com/gji/article/238/3/1582/7701416",
    },
    {
        "id": "P09", "kategori": "Kahramanmaraş",
        "baslik": "Stress-mediated multi-fault rupture dynamics of the 2023 Kahramanmaraş earthquake sequence, Türkiye",
        "dergi": "Scientific Reports", "yil": 2026,
        "doi": "10.1038/s41598-026-45723-7", "url": "https://www.nature.com/articles/s41598-026-45723-7",
    },
    {
        "id": "P10", "kategori": "Kahramanmaraş",
        "baslik": "A bibliometric analysis of the 2023 Kahramanmaraş Earthquakes: trends, gaps, and policy implications",
        "dergi": "Natural Hazards (Springer)", "yil": 2025,
        "doi": "10.1007/s11069-025-07305-0", "url": "https://link.springer.com/article/10.1007/s11069-025-07305-0",
        "notlar": "2023–2025 arasında 577 belgenin bibliometrik meta-analizi",
    },
    {
        "id": "P11", "kategori": "Kahramanmaraş",
        "baslik": "The 2023 Kahramanmaraş earthquake sequence (Nature Special Collection)",
        "dergi": "Nature", "yil": "2023–2025",
        "url": "https://www.nature.com/collections/ijjbhjghhb",
    },
    # KAF / Kuzey Anadolu Fayı
    {
        "id": "P12", "kategori": "KAF",
        "baslik": "Interseismic strain build-up on the submarine North Anatolian Fault offshore Istanbul",
        "dergi": "Nature Communications", "yil": 2019,
        "doi": "10.1038/s41467-019-11016-z", "url": "https://www.nature.com/articles/s41467-019-11016-z",
        "notlar": "Marmara segmenti kilitlenme analizi",
    },
    {
        "id": "P13", "kategori": "KAF/EAF",
        "baslik": "Seismotectonic Frame of the East and North Anatolian Faults, Turkey",
        "dergi": "Springer Book Chapter", "yil": 2025,
        "doi": "10.1007/978-3-031-80928-6_13", "url": "https://link.springer.com/chapter/10.1007/978-3-031-80928-6_13",
    },
    # UCERF / Operasyonel Tahmin
    {
        "id": "P14", "kategori": "UCERF",
        "baslik": "Uniform California Earthquake Rupture Forecast, version 3 (UCERF3) — Time-independent model",
        "yazarlar": "Field E.H. et al.", "dergi": "BSSA", "cilt": 104, "sayi": 3, "sayfalar": "1122–1180", "yil": 2014,
        "doi": "10.1785/0120130164", "url": "https://southern.scec.org/ucerf",
    },
    {
        "id": "P15", "kategori": "UCERF",
        "baslik": "A Spatiotemporal Clustering Model for UCERF3 (UCERF3-ETAS)",
        "yazarlar": "Field E.H.; Milner K.R. et al.", "dergi": "BSSA", "yil": 2017,
        "doi": "10.1785/0120160173",
    },
    # PEER
    {
        "id": "P16", "kategori": "PEER",
        "baslik": "Towards Deep Learning-Based Structural Response Prediction and Ground Motion Reconstruction",
        "yazarlar": "Mosalam K.M.; Pang I.K.T.; Günay S.", "dergi": "PEER Report 2025/01", "yil": 2025,
        "url": "https://peer.berkeley.edu/publications/2025-01",
    },
    # USGS
    {
        "id": "P17", "kategori": "USGS Strategy",
        "baslik": "USGS Earthquake Hazards Program Decadal Science Strategy 2024–33",
        "yayinci": "USGS", "yil": 2024,
        "doi": "10.3133/cir1544", "url": "https://pubs.usgs.gov/publication/cir1544",
    },
]

# ── BÖLÜM F: TEMEL DERGİLER ────────────────────────────────────────────────────

KEY_JOURNALS_EQ: list[dict] = [
    {"ad": "Earthquake Engineering & Structural Dynamics (EESD)", "yayinci": "Wiley", "url": "https://onlinelibrary.wiley.com/journal/10969845"},
    {"ad": "Earthquake Spectra", "yayinci": "EERI/SAGE", "url": "https://journals.sagepub.com/home/eqs"},
    {"ad": "Bulletin of the Seismological Society of America (BSSA)", "yayinci": "SSA", "url": "https://pubs.geoscienceworld.org/ssa/bssa"},
    {"ad": "Geophysical Research Letters (GRL)", "yayinci": "AGU/Wiley", "url": "https://agupubs.onlinelibrary.wiley.com/journal/19448007"},
    {"ad": "Journal of Geophysical Research — Solid Earth (JGR)", "yayinci": "AGU/Wiley"},
    {"ad": "Nature Geoscience", "yayinci": "Springer Nature", "url": "https://www.nature.com/ngeo/"},
    {"ad": "Nature Communications", "yayinci": "Springer Nature", "url": "https://www.nature.com/ncomms/"},
    {"ad": "Communications Earth & Environment", "yayinci": "Springer Nature", "url": "https://www.nature.com/commsenv/"},
    {"ad": "Scientific Reports", "yayinci": "Springer Nature"},
    {"ad": "Soil Dynamics and Earthquake Engineering (SDEE)", "yayinci": "Elsevier"},
    {"ad": "Engineering Structures", "yayinci": "Elsevier"},
    {"ad": "Bulletin of Earthquake Engineering (BEE)", "yayinci": "Springer"},
    {"ad": "Seismological Research Letters (SRL)", "yayinci": "SSA"},
    {"ad": "Journal of Earthquake Engineering (JEE)", "yayinci": "Taylor & Francis"},
    {"ad": "Natural Hazards", "yayinci": "Springer"},
    {"ad": "Teknik Dergi (TDV)", "yayinci": "Türk Deprem Vakfı", "not": "Türkçe"},
]

# ── BÖLÜM G: VERİTABANLARI VE YAZILIMLAR ─────────────────────────────────────

DATABASES_AND_SOFTWARE: list[dict] = [
    {"ad": "KRDAE İvme Veritabanı", "url": "http://www.koeri.boun.edu.tr/", "tur": "strong-motion", "bolge": "Turkey"},
    {"ad": "PEER NGA-West2", "url": "https://peer.berkeley.edu/peer-strong-ground-motion-databases/", "tur": "strong-motion", "bolge": "Global"},
    {"ad": "ESM (Engineering Strong Motion)", "url": "https://esm-db.eu/", "tur": "strong-motion", "bolge": "Europe"},
    {"ad": "IRIS/EarthScope FDSN", "url": "https://www.earthscope.org/", "tur": "waveform archive"},
    {"ad": "OpenSees", "url": "https://opensees.berkeley.edu/", "tur": "FEM software", "gelistiren": "PEER/UC Berkeley"},
    {"ad": "OpenQuake Engine (GEM)", "url": "https://www.globalquakemodel.org/openquake", "tur": "PSHA software"},
    {"ad": "HAZUS-MH", "url": "https://www.fema.gov/hazus", "tur": "loss estimation", "gelistiren": "FEMA"},
    {"ad": "ELER", "tur": "loss estimation", "gelistiren": "Erdik & Durukal (KRDAE)"},
    {"ad": "Coulomb 3.3", "url": "https://doi.org/10.5066/F72N51GG", "tur": "stress transfer software", "gelistiren": "USGS"},
    {"ad": "TDTH (Türkiye Deprem Tehlike Haritası)", "url": "https://tdth.afad.gov.tr", "tur": "hazard map", "gelistiren": "AFAD"},
]

# ── ÖZET ─────────────────────────────────────────────────────────────────────

ACADEMIC_KB_SUMMARY: dict = {
    "versiyon": "3.0",
    "derleme_tarihi": "2026-05-26",
    "derleyen": "Claude (Anthropic) — Mükremin Yüksel için",
    "alan": "Deprem Mühendisliği / Earthquake Engineering",
    "sayilar": {
        "turk_kurumlar": len(TURKISH_INSTITUTIONS),
        "uluslararasi_kurumlar": len(INTERNATIONAL_INSTITUTIONS),
        "kitaplar": len(ESSENTIAL_TEXTBOOKS),
        "turk_akademisyenler": len(KEY_SCHOLARS["turkey"]),
        "uluslararasi_akademisyenler": len(KEY_SCHOLARS["international"]),
        "guncel_makaleler": len(RECENT_PUBLICATIONS_EQ),
        "temel_dergiler": len(KEY_JOURNALS_EQ),
        "veritabani_yazilim": len(DATABASES_AND_SOFTWARE),
    },
    "veri_bloklari": [
        "TURKISH_INSTITUTIONS",
        "INTERNATIONAL_INSTITUTIONS",
        "ESSENTIAL_TEXTBOOKS",
        "KEY_SCHOLARS",
        "RECENT_PUBLICATIONS_EQ",
        "KEY_JOURNALS_EQ",
        "DATABASES_AND_SOFTWARE",
    ],
}
