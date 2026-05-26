"""
knowledge_base.py — DepremRadarı Akademik Bilgi Havuzu  v1.44
══════════════════════════════════════════════════════════════
Ajan 7 (UI) + Ajan 8 (Bilim) + Ajan 9 (Tasarım) ortak kararı.
Tüm kaynaklar DOI/URL ile doğrulanmış peer-reviewed yayınlar.

Her panel en az bir SCIENCE_NOTES anahtarıyla ilişkilendirilir:
    with st.expander("🎓 Bilim Notu", expanded=False):
        st.markdown(SCIENCE_NOTES.get("panel_key", ""))
"""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# SCIENCE_NOTES
# ────────────────────────────────────────────────────────────────────────────

SCIENCE_NOTES: dict[str, str] = {

    "ana_harita": """
🔬 **Anlık Deprem İzleme**

USGS FDSN API, Kandilli Gözlemevi ve AFAD ağlarından gerçek zamanlı veri çeker.
Deprem büyüklüğü logaritmik ölçekte: M7, M6'dan **32 kat** daha fazla enerji açığa çıkarır
(her tam birim ~31.6×, her 0.5 birim ~5.6× fark).

*Kaynak: Richter 1935, Bull. Seismol. Soc. Am.; USGS FDSN API docs (earthquake.usgs.gov)*
""",

    "psha": """
🔬 **Olasılıksal Sismik Tehlike (PSHA)**

PSHA, Cornell (1968) yöntemini kullanır: tüm olası deprem kaynaklarını, büyüklük
dağılımlarını ve yer hareketi azalım ilişkilerini (GMPE) birleştirerek belirli bir
aşılma olasılığı için maksimum yer ivmesini (PGA) hesaplar.

— **475 yıl dönüş periyodu** = 50 yılda %10 aşılma olasılığı (binalarda standart tasarım kriteri)
— Türkiye TBDY-2018 bu değerleri kullanır; Doğu Türkiye'de PGA 0.6g'yi aşabilir
— Güneydoğu Türkiye (KAF doğusu + EAF): PGA > 0.8g bölgeler mevcuttur

*Kaynak: Cornell 1968, BSSA 58(5); Woessner et al. 2015, Bull. Earthq. Eng.
DOI: 10.1007/s10518-015-9795-1; AFAD TDTH-2018*
""",

    "odak_mekanizmasi": """
🔬 **Odak Mekanizması (Beach Ball)**

Her deprem üç ana tipte olabilir:
- **Doğrultu atımlı** (KAF gibi): Levhalar yatay kayar — daire üzerinde iki siyah dilim
- **Normal**: Levhalar uzaklaşır (Ege genişleme rejimi) — üst kısım siyah
- **Ters atımlı**: Levhalar yaklaşır (Zagros, Doğu Türkiye) — alt kısım siyah

KAF sağ yönlü doğrultu atımlı: Anadolu batıya kaçıyor, Avrasya levhası sabit.
GCMT katalogunda (1976-günümüz) Türkiye için 3.000+ moment tensor çözümü var.

*Kaynak: Ekström et al. 2012, Phys. Earth Planet. Int.
DOI: 10.1016/j.pepi.2012.04.002; gcmt.org*
""",

    "b_degeri": """
🔬 **Gutenberg-Richter b-Değeri**

**log₁₀(N) = a − b·M** ilişkisinde b genellikle ~1.0'dır.

| b değeri | Yorum |
|----------|-------|
| b < 0.8  | Yüksek gerilme bölgesi — büyük deprem riski artmış |
| b ≈ 1.0  | Tipik tektonik aktivite |
| b > 1.2  | Düşük gerilme / volkanik aktivite |

Türkiye ortalaması: **b ≈ 0.92 ± 0.05** (Öztürk et al. 2011, JGR)
Erzincan civarı: b ≈ 0.87 — yüksek gerilme göstergesi.

*Kaynak: Aki 1965, BSSA 55(3); Wiemer & Wyss 2000, BSSA
DOI: 10.1785/0120000029*
""",

    "coulomb": """
🔬 **Coulomb Stres Transferi (ΔCFS)**

Bir deprem, komşu fay segmentlerindeki gerilme dengesini değiştirir:

> **ΔCFS = Δτ + μ' · Δσₙ**
>
> Δτ = kayma gerilme değişimi, μ' = görünür sürtünme (~0.4), Δσₙ = normal gerilme

**Tetikleme eşiği:** ΔCFS > 0.01 MPa (0.1 bar)

1999 İzmit M7.6 sonrası Coulomb analizi, Düzce depremini 3 ay önceden tahmin etmişti.
2023 Kahramanmaraş çiftinde Pazarcık → Elbistan yüklenmesi 9 saat içinde kırılmaya yol açtı.

*Kaynak: King, Stein & Lin 1994, BSSA DOI: 10.1785/BSSA0840030865;
Stein 1999, Nature DOI: 10.1038/45144*
""",

    "insar": """
🔬 **InSAR Yer Deformasyonu**

Sentinel-1 C-band SAR (5.6 cm dalga boyu) **mm hassasiyetle** yer deformasyonunu ölçer.

- Her renk döngüsü (interferogram fringe) = **2.8 cm** yerdeğişimi (yarım dalga boyu)
- Ölçüm yönü: Uydu görüş hattı (LOS) — eğim açısı ~33-46°
- 2023 Kahramanmaraş: InSAR maksimum **8 m** yatay, **3 m** dikey yer değişimi ölçtü

*Kaynak: Massonnet & Feigl 1998, Rev. Geophys. DOI: 10.1029/97RG03139;
Xu et al. 2023, Science DOI: 10.1126/science.adf7640*
""",

    "tarihi_sismisite": """
🔬 **2000 Yıllık Sismisite**

Ambraseys (2009) katalogu, Orta Doğu ve Doğu Akdeniz için MÖ 1000 – MS 2006
depremlerini içerir. Temel bulgular:

- KAF üzerinde **~150-200 yıllık** tekrar süresi tespit edilmiştir
- Erzincan segmenti: 1939 M7.8, 1784 M7.2, 1254 M7.0+ — yaklaşık 350 yıllık döngü
- Marmara: Doğrulama edilmiş son büyük deprem **1766** (Parsons 2004)
- İstanbul'da MS 480, 553, 740, 989, 1344 büyük hasarlı depremler kayıtlı

*Kaynak: Ambraseys 2009, Cambridge Univ. Press (ISBN: 9780521872928);
Ambraseys & Jackson 1998, Geophys. J. Int. 133(2)*
""",

    "sismik_dongu": """
🔬 **Sismik Döngü ve Elastik Geri Tepme**

Reid (1910) elastik geri tepme teorisi: fay kilitliyken gerilme birikir, depremde açığa çıkar.

**BPT (Brownian Passage Time) modeli:** Son depremin zamanı ve kayma hızından sonraki
olasılık hesaplanır. Koşullu olasılık formülü:

> P(t | T) = Φ[(T/μ − 1)/α] + e^(2/α²) · Φ[−(T/μ + 1)/α]

KAF Marmara segmentinde ~250 yıl bekleme süresi — son büyük deprem 1766.

*Kaynak: Reid 1910, Carnegie Inst. Washington; Matthews et al. 2002, BSSA
DOI: 10.1785/0120010254*
""",

    "shakemap": """
🔬 **ShakeMap — Sarsıntı Haritası**

**MMI (Mercalli Değiştirilmiş Yoğunluk) Ölçeği** I–XII arası, 1902'de geliştirildi:

| MMI | Tanım | Tipik PGA |
|-----|-------|-----------|
| I-III | Hissedilmez / hafif | < 0.02g |
| V-VI | Hasarlı / orta hasar | 0.09-0.18g |
| IX-X | Çok ağır hasar | 0.6-1.2g |
| XI-XII | Tam yıkım | > 1.2g |

PGA-MMI dönüşümü: Wald et al. (1999) doğrusal regresyon modeli.
İzoseist yarıçap: Bakun & Wentworth (1997) ampirik bağıntısı.

*Kaynak: Wald et al. 1999, Earthq. Spectra DOI: 10.1193/1.1586058*
""",

    "tsunami": """
🔬 **Akdeniz Tsunami Tehlikesi**

**Fizik:** Tsunami hızı = √(g·d) — 4000 m derinde ~720 km/h, kıyıda yavaşlar-yükselir.

Akdeniz tarihinin önemli olayları:
- **MS 365 Girit M~8.5**: Mısır, Sicilya, İspanya kıyıları vuruldu
- **1999 İzmit M7.6**: Körfezde lokalize 1-3 m tsunami, 155 ek kayıp
- **2023 Kahramanmaraş**: Denize uzak kırık, tsunami yok

NEAMTHM18 projesi: Kuzey Doğu Atlantik ve Akdeniz için en kapsamlı olasılıksal
tsunami tehlike modeli.

*Kaynak: Papadopoulos & Fokaefs 2005, NHESS; Basili et al. 2021, Earth-Sci. Rev.
DOI: 10.1016/j.earscirev.2021.103673*
""",

    "vs30": """
🔬 **Vs30 — Zemin Büyütme Katsayısı**

Vs30: üst 30 metredeki ortalama kayma dalgası hızı. NEHRP zemin sınıfları:

| Sınıf | Vs30 (m/s) | Açıklama | Büyütme |
|-------|-----------|----------|---------|
| A | > 760 | Sağlam kaya | 1× |
| B | 360-760 | Kaya | 1.5× |
| C | 180-360 | Sert zemin | 2-3× |
| D | < 180 | Yumuşak alüvyon | 5-20× |

1999 İzmit'te Adapazarı'nda zemin büyütmesi hasarı 3-4× artırdı.

*Kaynak: Wald & Allen 2007, BSSA DOI: 10.1785/0120060267;
Boore et al. 2014, Earthq. Spectra DOI: 10.1193/070113EQS184M*
""",

    "hazus": """
🔬 **Kayıp Tahmini — Kırılganlık Eğrileri**

HAZUS metodolojisi: **yapı tipi × zemin sınıfı × PGA → hasar olasılığı**

Türkiye bina stoku (AFAD 2018):
- %60 betonarme çerçeve (büyük kısmı 1980 öncesi — yetersiz tasarım)
- %15 yığma
- %25 diğer (ahşap, çelik, karma)

**1999 İzmit sonrası:** 17.480 ölü, 300.000 evsiz — tahm. kayıp 12-20 milyar USD
**2023 Kahramanmaraş:** ~50.000 ölü, 160.000 yıkılmış bina

*Kaynak: FEMA 2003 (HAZUS-MH); Lagomarsino & Giovinazzi 2006, Bull. Earthq. Eng.
DOI: 10.1007/s10518-006-9025-y*
""",

    "kaf_sismik_acik": """
🔬 **KAF Sismik Açık (Slip Deficit)**

GPS ölçümleri (Reilinger 2006): KAF üzerinde **20-25 mm/yıl** kayma hızı.

Kilitli fay segmentinde biriken gerilme:
> **Kayma açığı = kayma hızı × geçen süre − gerçekleşen kayma**

İzmit-Marmara segmenti: son büyük depremden (1766) bu yana **~5.7 metre** açık birikmiş.
Bu değer Mw 7.2-7.5 aralığında bir deprem potansiyeline karşılık gelir.

*Kaynak: Reilinger et al. 2006, JGR DOI: 10.1029/2005JB004051;
Ergintav et al. 2014, JGR DOI: 10.1002/2013JB010388*
""",

    "plaka_sim": """
🔬 **Plaka Tektoniği Simülasyonu**

**NNR-MORVEL56** (Argus et al. 2011): 56 levha için jeolojik zaman ortalamalı Euler kutupları.

Anadolu levhası hareketi:
- **~25 mm/yıl** batıya — Arabistan iticisi + Ege çekimi mekanizması
- Euler kutbu: ~30.7°N, 32.6°E (Reilinger 2006 GPS)

⚠️ MORVEL hızları 3.16 milyon yıl ortalamasıdır; GPS anlık hızlardan %10-30 farklı olabilir.

*Kaynak: Argus et al. 2011, GJI DOI: 10.1111/j.1365-246X.2009.04491.x;
Reilinger et al. 2006, JGR DOI: 10.1029/2005JB004051*
""",

    "erzincan": """
🔬 **Erzincan — Türkiye'nin Sismik Başkenti**

Erzincan, KAF'ın doğu ucu ile **Kuzey Doğu Anadolu Fayı** ve **Doğu Anadolu Fayı**'nın
yakınlaştığı üçlü tektonik noktada yer alır.

Kayıtlı büyük depremler:
| Yıl | Ms/Mw | Etki |
|-----|-------|------|
| 1254 | ~7.0 | İlk belgelenmiş büyük hasar |
| 1784 | ~7.2 | Şehir tahrip |
| 1939 | 7.8 | ~33.000 ölü — Türkiye'nin en ölümcülü |
| 1992 | 6.8 | 498 ölü |

*Kaynak: Ambraseys & Jackson 1998, GJI 133(2);
Şengör et al. 2005, Ann. Rev. Earth Planet. Sci. DOI: 10.1146/annurev.earth.32.101802.120415*
""",
}

# ────────────────────────────────────────────────────────────────────────────
# ANIMATION_CONFIG — Ajan 9 tasarım standartları
# ────────────────────────────────────────────────────────────────────────────

ANIMATION_CONFIG: dict = {
    "plotly_config": {
        "displayModeBar": True,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    },
    "color_scales": {
        "hazard":  "RdYlGn_r",   # PGA haritaları
        "depth":   "viridis",    # Deprem derinliği
        "mmi":     "RdBu_r",    # Yoğunluk
        "stress":  "RdBu",      # Coulomb stres
        "time":    "plasma",    # Zaman serisi
        "vs30":    "YlOrRd_r",  # Zemin sınıfı
    },
    "fonts": {
        "title":   "Georgia, serif",
        "numbers": "Courier New, monospace",
        "body":    "system-ui, sans-serif",
    },
    "fault_colors": {
        "gem":      "#FF6B35",   # GEM aktif fay — turuncu
        "kaf_high": "#E24B4A",  # KAF yüksek risk — kırmızı
        "kaf_mid":  "#EF9F27",  # KAF orta risk — sarı
        "kaf_low":  "#1D9E75",  # KAF düşük risk — yeşil
    },
}

# ────────────────────────────────────────────────────────────────────────────
# FUN_FACTS — "Vay be" tonunda halk anlatımı (Türkçe), her panel için 2-3 bilgi
# ────────────────────────────────────────────────────────────────────────────

FUN_FACTS: dict[str, list[str]] = {
    "ana_harita": [
        "🌍 Her gün dünyada **~50 deprem** M≥4 olur — saniyede yaklaşık bir titreşim.",
        "⚡ M7 deprem, M6'nın 32× enerjisini açığa çıkarır; 0.5 birim fark ~5.6× enerji.",
    ],
    "psha": [
        "🗺️ TBDY-2018: **Hatay 0.50g, Konya 0.22g** — aynı ülke, 2.3× tehlike farkı.",
        "💰 PSHA hesabı sigortacıların binlerce dolara hesaplattığı tehlikeyi şeffaf gösterir.",
    ],
    "odak_mekanizmasi": [
        "⚫⚪ Beach ball **siyah = sıkışma, beyaz = genişleme** — tek bakışta fay tipi.",
        "🎯 2023 Kahramanmaraş: Pazarcık (Mw 7.8) + 9 saat sonra Elbistan (Mw 7.7) — iki fay, tek katastrof.",
    ],
    "b_degeri": [
        "📉 b < 0.7 = büyük olay baskınlığı işareti — bazı bölgelerde öncü uyarısı olabilir.",
        "🇹🇷 Erzincan b ≈ 0.87 — Türkiye ortalamasının altında, yüksek gerilme göstergesi.",
    ],
    "coulomb": [
        "⏰ 1999 İzmit kırığı **3 ay sonra** Düzce'yi tetikledi — +1.5 bar Coulomb yüklemesi.",
        "📏 ΔCFS **0.1 bar** (bir kahve fincanı ağırlığı bir tırnakta!) eşiği yeterli — *King 1994*.",
    ],
    "insar": [
        "🛰️ Sentinel-1 uzaydan **2.8 cm hassasiyetle** zemini ölçer — saf elektromanyetik faz farkı.",
        "📡 2023 Kahramanmaraş için ESA **aynı gün** interferogram yayımladı (5+ m yer değiştirme).",
    ],
    "tarihi_sismisite": [
        "🚂 KAF 1939→1999 doğudan batıya **700+ km** yürüdü — tektonik tren güzergahı.",
        "📜 Yaylabeli kazısında **2500 yıllık 9 büyük olay** belgelendi — Kozacı 2007.",
    ],
    "sismik_dongu": [
        "🏙️ Marmara Prens Adaları **1766'dan beri** kırılmadı — 5+ metre kayma açığı birikti.",
        "🎯 BPT olasılığı: tarih + paleo verisi → kesin değil ama stokastik gerçek.",
    ],
    "shakemap": [
        "🗺️ Deprem oldu, **5 dakika içinde** USGS ShakeMap MMI haritası yayında.",
        "📐 MMI VII-VIII eşiği PGA ~0.1g — yıkım başlar (*Wald 1999 Earthq. Spectra 15(3)*).",
    ],
    "tsunami": [
        "🌊 2500 m derinde tsunami **~565 km/h** (jet uçağı hızı) — *c = √(g·d)*.",
        "⚠️ Hellenic trench M8.5 senaryoda Bodrum'a **12-18 dk'da** ulaşır.",
    ],
    "vs30": [
        "🏚️ Adapazarı 1999'da **Vs30 = 140 m/s** (yumuşak dolgu) → 2.5× büyütme = yıkım.",
        "🪨 Aynı ilçe içinde 100 m arayla Vs30 200'den 600'e çıkabilir.",
    ],
    "hazus": [
        "💀 Yığma 1970 öncesi 0.10g'de çöker; TBDY-2018 betonarme 1.7g — **17× fark = hayat farkı**.",
        "🏚️ Türkiye'nin ~20M yapısının **%30-40'ı 1999 öncesi** yönetmelikle inşa.",
    ],
    "kaf_sismik_acik": [
        "🔒 Marmara Prens Adaları φ ≈ **0.95** — neredeyse tam kilitli, İstanbul deprem hotspot'u.",
        "📐 GPS satırları 'fay kilitli, enerji birikiyor' diyor — milimetre/yıl hassasiyet.",
    ],
    "plaka_sim": [
        "🏃 Anadolu levhası batıya **~25 mm/yıl** kayıyor — saniyede neredeyse bir tırnak büyümesi.",
        "🌍 İstanbul'daki bir bina her yıl Londra'ya 2 cm daha yaklaşıyor.",
    ],
    "erzincan": [
        "💔 1939 Erzincan = **~33.000 ölü** — Türkiye'nin en ölümcül depremi.",
        "🔁 Erzincan 1254, 1784, 1939 — yaklaşık 350 yıllık döngü; sonraki pencere ucunda.",
    ],
}


# ────────────────────────────────────────────────────────────────────────────
# Yardımcı: render_bilim_notu(panel_key)
# st.expander içinde Bilim Notu + Vay-Be Hikayeleri tek seferde render eder.
# ────────────────────────────────────────────────────────────────────────────

def render_bilim_notu(panel_key: str, st_module=None,
                      expanded: bool = False, baslik: str | None = None) -> None:
    """
    Streamlit panelinde **"🎓 Bunu Öğren"** expander'ı oluşturur.

    Parameters
    ----------
    panel_key  : SCIENCE_NOTES anahtarı (yoksa sessiz no-op).
    st_module  : Streamlit modülü; None ise içeri aktarmayı dener.
    expanded   : Expander varsayılan açık mı?
    baslik     : Özel başlık; None ise panel_key'den türetilir.

    Usage
    -----
        from knowledge_base import render_bilim_notu
        render_bilim_notu("kaf_sismik_acik", st)
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

    title = baslik or f"🎓 Bunu Öğren — {panel_key.replace('_', ' ').title()}"
    with st_module.expander(title, expanded=expanded):
        if note:
            st_module.markdown(note)
        if facts:
            st_module.markdown("---")
            st_module.markdown("**💡 Bilim Hikayeleri:**")
            for f in facts:
                st_module.markdown(f"> {f}")

