# Akademik Açıklama Standardı (Faz 0)

**Tarih:** 2026-05-26
**Hazırlayan:** Ajan koordinatörü (kullanıcı + Codex onaylı)
**Bağlayıcı:** Faz 1 Pilot ve sonraki tüm Faz 1-5 implementasyonları bu standarda uymak ZORUNDADIR.

---

## 1. 6 Bölümlü Açıklama Şablonu

Her grafik veya panel için aşağıdaki 6 bölüm DOLU olmak ZORUNDADIR. Eksik bölüm → `Eksik` sınıfı, akademik standart **karşılanmamış**.

### 🎯 Bölüm 1 — Ne Gösteriyor? (What)
**Amaç:** Grafiğin tek cümlelik özet anlatımı + bir paragraf bağlam.

**Yasak:**
- Yüzeysel/marketing tonu ("Bu harika grafik...")
- Belirsiz dil ("Bir şekilde gösterir")

**Örnek (kötü):**
> *"Bu grafik depremleri gösterir."*

**Örnek (iyi):**
> *"Her nokta seçili zaman penceresindeki bir deprem olayını temsil eder. Yatay eksen depremlerin gerçekleştiği tarih, dikey eksen Magnitude (Moment Magnitude Mw, USGS/AFAD raporundan), renk derinliği temsil eder. Amaç: Erzincan ve çevresindeki sismik aktivitenin zamansal-uzamsal dağılımını tek bakışta görmek."*

---

### 👁️ Bölüm 2 — Nasıl Okunur? (How)
**Amaç:** Eksen birimi, renk paleti, sembol anlamı, ölçek özelliği (lineer/log), interaktif elementler.

**İçermeli:**
- X-ekseni: birim + aralık
- Y-ekseni: birim + aralık + log/lineer
- Renk kodu: anlam + ölçek (kategorik/sıralı/diverging)
- Sembol/şekil: boyut/çap anlamı
- Hover/etiket: nasıl etkileşilir
- Yakınlaştırma/pan ipucu

**Örnek (iyi):**
> *"X: tarih (gün hassasiyet), Y: Magnitude (Mw 0-9 lineer), nokta boyutu Mw'ye **kuadratik** orantılı (M5 → 25× daha büyük nokta). Renk derinlik (km): yeşil 0-30, sarı 30-70, kırmızı 70+ km. Hover → konum + kaynak. Modebar'dan zoom-in/pan/reset yapabilirsiniz."*

---

### 🔬 Bölüm 3 — Formül / Teori (Science)
**Amaç:** Hangi bilimsel modele veya formüle dayanır. Açık denklem (LaTeX) veya algoritma açıklaması.

**İçermeli:**
- Temel denklem(ler) — `st.latex()` ile
- Parametrelerin tanımları
- Hangi teorinin uzantısı/uyarlamasıdır
- Hesap algoritmasının ana adımları

**Yasak:**
- Kaynaksız formül
- "Bilim diyor ki..." gibi belirsiz iddialar

**Örnek (iyi):**
> *Gutenberg-Richter ilişkisi (Gutenberg & Richter 1944):*
> $$\log_{10} N(M) = a - b \cdot M$$
> *Burada $N(M)$ verilen magnitude $M$ veya üstündeki olay sayısı, $a$ aktivite seviyesi, $b$ büyük/küçük deprem oranını gösteren $b$-değeri. Tipik global $b \approx 1.0$. Maximum-likelihood estimation (Aki 1965):*
> $$b = \frac{\log_{10}(e)}{\bar{M} - M_c}$$

---

### 💡 Bölüm 4 — Yorumlama Rehberi (Interpretation)
**Amaç:** Kullanıcı **bu grafiği bilimsel olarak doğru yorumlayabilsin**. Örnek değerler + ne anlam ifade ettiği.

**İçermeli:**
- Tipik aralıklar + ne anlama gelir
- "Yüksek değer ↔ Düşük değer" karşılaştırması
- Yaygın yanlış yorumlamalar (uyarı niteliği)
- Pratik kullanım örnekleri

**Örnek (iyi):**
> *b ≈ 1.0 → "normal" sismik rejim (global ortalama). b > 1.2 → küçük deprem ağırlıklı (düşük stres, asperity yok), b < 0.8 → büyük deprem ağırlıklı (yüksek stres, kilitli fay). NAFZ Erzincan segmenti tipik b ≈ 0.9-1.0 (Bayrak 2002). **Yanlış yorumlama:** b < 0.8 "büyük deprem yakında" değildir — sadece istatistiksel bir göstergedir.*

---

### ⚠️ Bölüm 5 — Sınırlamalar (Limitations)
**Amaç:** Modelin/yöntemin GEÇERLİLİK SINIRLARI. Hangi durumda yanlış sonuç verir, hangi varsayımlara dayanır.

**İçermeli:**
- Veri eksikliği/önyargısı (kataloğ kompletizlik)
- Model varsayımları (lineer/poisson/stationarity)
- Hangi rejimde geçersizdir
- Hangi profesyonel ürün/raporla KARIŞTIRMAMALI

**Yasak:**
- Sınırlama yok demek
- "Mükemmel" gibi mutlak iddia

**Örnek (iyi):**
> *Gutenberg-Richter, **Mc (magnitude of completeness)** üstünde geçerlidir. Kandilli kataloğu için Türkiye Mc ≈ 2.5-3.0 (Bayrak 2002). Bu eşiğin altındaki olaylar katalog eksikliğinden gözükmez ve b-tahminini bozar. Ayrıca lineer regresyon güçlü-olay outlier'larına duyarlıdır; MLE (Aki 1965) yöntemi tercih edilir. **Resmi tehlike değerlendirmesi değildir** — AFAD/PSHA haritaları kullanılmalıdır.*

---

### 📚 Bölüm 6 — Kaynaklar (References)
**Amaç:** Her bilimsel iddianın doğrulanabilir, peer-reviewed referansla bağlanması.

**İçermeli:**
- 3-5 peer-reviewed makale (DOI ZORUNLU)
- 1-2 kitap bölümü (Aki & Richards, Lay & Wallace, vb.)
- 1 Türkçe yerel kaynak (varsa: KOERI, AFAD, MTA)
- 1 popüler/eğitsel kaynak (varsa: IRIS Education)

**Yasak:**
- DOI olmadan makale atfı
- "Bilim adamları söyledi" gibi belirsiz atıf
- Wikipedia tek kaynak olarak

**Örnek (iyi):**
- Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a − bM. *Bull. Earthq. Res. Inst. Tokyo*, 43, 237-239.
- Gutenberg, B., & Richter, C. F. (1944). Frequency of earthquakes in California. *Bull. Seismol. Soc. Am.*, 34(4), 185-188.
- Bayrak, Y., et al. (2002). A quantitative appraisal of earthquake hazard parameters in Turkey. *Bull. Seismol. Soc. Am.*, 92(2), 537-547. [DOI:10.1785/0120000748](https://doi.org/10.1785/0120000748)
- Lay, T., & Wallace, T. C. (1995). *Modern Global Seismology*. Academic Press, Bölüm 12.4.

---

## 2. Kaynak Disiplini (Strict)

### 2.1 Kaynaksız İddia YOK
- Hiçbir bilimsel iddia kaynak olmadan yazılmaz.
- "Tipik aralık", "yaygın değer" gibi ifadeler de kaynakla beraber.

### 2.2 DOI Doğrulama
- Her DOI **gerçek** olmalı (örn. `10.1785/0120050039` formatı)
- Uydurma DOI **kesinlikle yasak**

### 2.3 Kitap Bölüm Numarası
- Aki & Richards 2002, Bölüm 4.2 gibi spesifik atıf
- "Aki kitabında" gibi belirsiz atıf yasak

### 2.4 Türkçe Yerel Kaynaklar
- Sucuoğlu & Akkar 2014 *Basic Earthquake Engineering* (Türkiye standardı)
- Aydan, Ö. (2017) *Rock Mechanics and Earthquake Engineering*
- Erdik, M. (örneğin Marmara EEW projesi)
- KOERI yayınları, AFAD raporları (resmi)

---

## 3. Disclaimer Zorunlu Yerler

### 3.1 Resmi Tehlike Haritası ile Karıştırılabilecek
**Zorunlu disclaimer:**
> ⚠️ **Bu panel resmi tehlike değerlendirmesi değildir.** Yapı tasarımı, sigorta veya afet planlaması için TBDY-2018 (Türkiye Bina Deprem Yönetmeliği) ve AFAD'ın resmi tehlike haritalarını kullanın.

**Hangi panellerde:** PSHA, ShakeMap, HAZUS, Vs30, Erzincan Mikrozon

### 3.2 Erken Uyarı Sistemi ile Karıştırılabilecek
**Zorunlu disclaimer:**
> ⚠️ **Bu simülasyon eğitim/kavram demosu — gerçek bir Erken Uyarı Sistemi (EEW) değildir.** Türkiye'de operasyonel EEW olarak AFAD-EQE/EWS çalışmaktadır.

**Hangi panellerde:** Erken Uyarı, Artçı Tahmin

### 3.3 Tahmin / Öngörü ile Karıştırılabilecek
**Zorunlu disclaimer:**
> ⚠️ **Deprem tahmini DEĞİLDİR.** Bu panel istatistiksel veya retrospektif analiz aracıdır; gelecekteki olayları öngörmez. Mevcut bilimsel uzlaşı: deterministik deprem tahmini henüz mümkün değildir (Geller et al. 1997 *Science*).

**Hangi panellerde:** η Kümeleme, RTL, AMR, ETAS, Sismik Açık, Foreshock, Astronomik

### 3.4 Hasar/Kayıp Tahmini ile Karıştırılabilecek
**Zorunlu disclaimer:**
> ⚠️ **Hasar/kayıp tahmini değildir.** HAZUS-MH (FEMA 2003) metodolojisi *bölgesel hızlı kestirim*tir; bina-bazlı yapısal hasar için detaylı mühendislik analizi gerekir.

**Hangi panellerde:** HAZUS Kayıp

### 3.5 Plaka Hareketi / Yer Değiştirme
**Zorunlu disclaimer:**
> ⚠️ **Seçilen referans çerçevesine göre göreceli hızdır.** Tek bir plaka hızı mutlak değil — referansa bağlı yorumlanmalıdır. Vektörler **TOPLANMAZ** (matematiksel olarak yanlış).

**Hangi panellerde:** Plaka Simülasyonu

---

## 4. Helper Fonksiyon Sözleşmesi

Aşağıdaki Python fonksiyon imzası `earthquake.py` içine eklenmiştir ve **TÜM** akademik açıklamalarda kullanılacaktır:

```python
def render_academic_explanation(
    title: str = "📖 Akademik Açıklama",
    what: str = None,           # Bölüm 1 — Ne gösteriyor?
    how: str = None,            # Bölüm 2 — Nasıl okunur?
    science: str = None,        # Bölüm 3 — Formül/teori (LaTeX destek)
    interpretation: str = None, # Bölüm 4 — Yorumlama rehberi
    limitations: str = None,    # Bölüm 5 — Sınırlamalar
    references: list = None,    # Bölüm 6 — Kaynak listesi (string)
    disclaimer: str = None,     # Opsiyonel: ek disclaimer
    expanded: bool = False,     # Default kapalı (kullanıcı tıklar)
) -> None:
    """Streamlit expander içinde 6-bölümlü akademik açıklama kartı render eder.
    Hiçbir parametre boş geçilmemeli (kaynak disiplini)."""
```

**Kullanım örneği:**

```python
render_academic_explanation(
    title="📖 Gutenberg-Richter b-Değeri — Akademik Açıklama",
    what="Her nokta seçili pencerede bir magnitude...",
    how="X-ekseni magnitude, Y-ekseni log10(N)...",
    science="""
$$\\log_{10} N(M) = a - b \\cdot M$$
Aki 1965 maximum likelihood estimation...
""",
    interpretation="b≈1.0 normal, b<0.8 yüksek stres asperity, b>1.2 düşük stres...",
    limitations="Mc altında geçersiz, lineer regresyon outlier'lara duyarlı...",
    references=[
        "Aki, K. (1965). *Bull. Earthq. Res. Inst. Tokyo* 43, 237-239.",
        "Gutenberg & Richter (1944). *BSSA* 34(4), 185-188.",
        "Bayrak et al. (2002). *BSSA* 92(2). [DOI:10.1785/0120000748](https://doi.org/10.1785/0120000748)",
        "Aki & Richards (2002). *Quantitative Seismology* Bölüm 12.4.",
    ],
    disclaimer=None,  # Bu panel için ek disclaimer gerekmez
)
```

---

## 4.5 Pilot Sonrası Eklenen Kurallar (v1.1 — Codex onayı)

Faz 1 (3 pilot) bitiminde içeriklerden çıkarılan örtük kurallar yazılı standarda alındı. **Sprint 2 ve sonrası için bağlayıcıdır.**

### 4.5.1 — Türkiye/Erzincan Örneği Zorunlu
Her panelde **en az 1 Türkiye/Erzincan/KAF/DAF örneği** bulunmak ZORUNDADIR. Genel global örnek tek başına yetmez; Türkiye karşılığı eklenmelidir.

**Tipik bağlam adayları:**
- 1939 Erzincan (M7.9, KAF doğu), 1999 İzmit (M7.4, KAF batı), 1999 Düzce (M7.2)
- 2023 Kahramanmaraş çift kırılma (Mw 7.8 + 7.5, DAFZ)
- KAF Erzincan/Yedisu/Tercan segmenti slip-rate, b-değeri, Vs30
- AFAD/KOERI kataloğu, MTA Diri Fay 2013, TBDY-2018

**Yasak:** Sadece San Andreas / Kobe / Tohoku örneğiyle bitirmek. Türkiye atfı yoksa panel 🟢 Akademik sayılmaz.

---

### 4.5.2 — Formül Birimi ve Tipik Kabuk Değeri Zorunlu
Her LaTeX denkleminin **hemen altında** Türkçe sade açıklama + her sembol için:

- **Birim** (SI birim sistemi tercih edilir: Pa, m, mm/yr, GPa, °)
- **Tipik kabuk/Türkiye değeri** (örn. *"$G \approx$ 30 GPa, kıtasal kabuk için"*)

**Örnek (iyi):**
> $\tau \geq \mu \cdot (\sigma_n - P_f) + C$
>
> burada $\tau$ = uygulanan kayma gerilmesi (**Pa**, kabukta tipik 10–100 MPa), $\mu$ = sürtünme katsayısı (**boyutsuz**, ≈ 0.6–0.85, Byerlee 1978), $\sigma_n$ = normal gerilme (**Pa**), $P_f$ = gözenek basıncı (**Pa**), $C$ = kohezyon (**Pa**, faylar için ihmal edilebilir).

**Yasak:** Çıplak denklem, sembol açıklaması olmadan veya birim belirtmeden.

---

### 4.5.3 — Geller 1997 + Disclaimer Genişletilmiş Kapsamı
Aşağıdaki paneller "tahmin/öngörü çağrıştıran" kategorisindedir ve **Geller 1997 atfı + açık disclaimer** ZORUNDADIR (önceki §3.3 listesini bu maddeyle genişletiyoruz):

- η Kümeleme, RTL, AMR, ETAS, Sismik Açık, Foreshock
- Astronomik korelasyon
- **b-Değeri / Gutenberg-Richter (özellikle b<0.8 yorumu)**
- **Elastik Geri Tepme / slip-deficit yorumlamaları**
- **Coulomb Stres transferi (komşu segment "tetiklenir" iddiası)**
- **PSHA olasılık çıktıları** (50 yılda %10 vs. "10 yıl içinde olacak" yanlış okuma)
- **EEW skor tahmini** (gerçek operasyonel sistem değil)

**Standart cümle (kopyala-yapıştır şablon):**
> ⚠️ **Bu panel/yorum bir deprem tahmin aracı DEĞİLDİR.** Mevcut bilimsel uzlaşı: deterministik deprem tahmini henüz mümkün değildir (Geller, R. J., 1997, *GJI* 131(3), 425–450, [DOI:10.1111/j.1365-246X.1997.tb06588.x](https://doi.org/10.1111/j.1365-246X.1997.tb06588.x)). Slip-deficit, b-değeri, η, RTL gibi göstergeler **olasılıksal tehlike girdisidir**, kesin zaman/yer/magnitüd değil.

---

## 4.6 LaTeX Yoğunluğu Rehberi (Codex tavsiyesi)

Katı limit YOK, ama **hedef profiller**:

| Panel tipi | LaTeX hedefi | Üst sınır |
|---|---:|---:|
| Canlı izleme (radar, ShakeMap) | 1–2 | 3 |
| İstatistik / Analiz (b-değeri, η, RTL) | 2–4 | 5 |
| Teorik eğitim (Reid, Coulomb, Okada) | 3–5 | 6 |
| Uydu / jeofizik (InSAR, SKS, Moho) | 2–3 | 4 |

**Kural:** Her denklemin hemen altında **sade Türkçe** açıklama (1–2 cümle). Üst üste 3 denklem peşi sıra konulmamalı — araya yorumlama paragrafı.

---

## 4.7 Referans Politikası (Codex tavsiyesi)

**Kalite > Sayı.**

| Kural | Değer |
|---|---|
| Panel başına minimum güvenilir kaynak | **4** |
| Görünür liste max referans | **8–10** |
| Fazlası | "Ek kaynaklar" alt-expander'da |
| Türkçe/yerel kaynak min | **1** (KAF/DAF/AFAD/KOERI/MTA/TBDY) |
| DOI veya peer-reviewed bağlantı | Her makale referansı için **zorunlu** |
| Wikipedia tek kaynak olarak | **Yasak** |

**Görünür kaynak listesinde ne olmalı:**
1. Orijinal/öncü kaynak (örn. Reid 1910, Gutenberg-Richter 1944)
2. Standart referans (Aki & Richards 2002, Lay & Wallace 1995)
3. Modern derleme (Annu. Rev. Earth Planet. Sci. veya Reports on Progress in Physics)
4. Türkiye/Erzincan özel çalışma (Bayrak 2002, Hubert-Ferrari 2002, Reilinger 2006)
5. (Varsa) son 5 yıl içinde Türkiye odaklı yayın (Melgar 2023, Kahramanmaraş 2023)

---

## 4.8 Panel Üstü Mini Okuma Rehberi (Codex tavsiyesi)

Akademik expander default `expanded=False`. Ama kullanıcı **expander'a tıklamadan da** grafiği yorumlayabilmeli. Bunun için her panelin **hemen üstünde** 1–2 cümlelik **mini rehber** bulunmak ZORUNDADIR.

**Şablon:**
> 📊 **Mini rehber:** [Grafiğin ne gösterdiği — tek cümle.] [Nasıl okunur — tek cümle.] *Detaylı akademik açıklama, formüller ve kaynaklar için 📖 Akademik Açıklama panelini açın.*

**Örnek (Canlı Radar için):**
> 📊 **Mini rehber:** Her halka son 7 günde gerçekleşmiş bir depremi temsil eder; halka boyutu magnitüd ile kuadratik, rengi derinlik ile kodlanır. *Detaylı açıklama, USGS magnitude ölçeği ve haversine formülü için 📖 Akademik Açıklama panelini açın.*

---

## 4.9 Expander İçinde Performans Kısıtları (Codex tavsiyesi)

`st.expander` **lazy-render değildir** — içindeki kod her rerun'da çalışır. Bu nedenle expander içinde:

✅ **İzin verilen:** `st.markdown`, LaTeX, `st.warning`, string-based referans listesi.
❌ **YASAK:** Plotly figürü, `pd.read_*`, network isteği, ağır hesap, büyük dataframe slice.

Ağır içerik gerekiyorsa **panel ana gövdesinde** render edilmeli, expander'a yalnızca **metin** girmeli.

---

## 5. İçerik Sınıflandırma Tablosu

Her panel/grafik için aşağıdaki 4 seviyeden biri atanır:

| Seviye | Tanım | Karar |
|---|---|---|
| 🔴 **Eksik** | 0-2 bölüm dolu | Akademik standardı KARŞILAMIYOR — yeniden yazılmalı |
| 🟠 **Temel** | 3-4 bölüm dolu, yüzeysel | Kabul edilebilir minimum — iyileştirme önerilir |
| 🟡 **Orta** | 5-6 bölüm dolu, peer-reviewed kaynaklı | Akademik kabul edilebilir |
| 🟢 **Akademik** | 6 bölüm + LaTeX + 3+ DOI + disclaimer | **Hedef seviye** |

`features/CONTENT_AUDIT.md` her panelin mevcut seviyesini ve hedef seviyesini listeler.

---

## 6. Versiyon Geçmişi

- **v1.0** (2026-05-26) — İlk standart belgesi. Faz 0 onayı: kullanıcı + Codex.
- **v1.1** (2026-05-26) — Faz 1 (3 pilot) sonrası Codex onaylı eklemeler: §4.5 (Türkiye örneği zorunlu / formül birimi / Geller disclaimer genişletilmiş kapsamı), §4.6 (LaTeX yoğunluğu rehberi), §4.7 (referans politikası — kalite>sayı, min 4 max 8–10), §4.8 (panel üstü mini rehber zorunlu), §4.9 (expander performans kısıtları).
