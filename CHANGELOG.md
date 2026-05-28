# Changelog

## v1.74–v1.81 - 2026-05-28 — 📚 FAZ B: AKADEMİK BİLGİ MERKEZİ (büyük sürüm)

**Amaç:** Kullanıcı vizyonu — "kitaplardan bilgiler, ileri düzey pratik kaynaklar, yeni teknolojiler, araştırmalar". Panel-açıklama standardından (Faz A) ayrı, eğitim/kaynak merkezi. Mevcut `📚 Akademik Kütüphane` → 4-sekmeli **Akademik Bilgi Merkezi**.

**Batch 0 (v1.74):** `features/BILGI_MERKEZI_SCHEMA.md` şeması + 4-sekme iskeleti. KEŞİF: knowledge_base.py'de zaten 18 kitap + 17 yayın + bilim insanları/dergiler/araçlar/kurumlar vardı ama UI'a bağlı değildi — bağlandı.

**Batch 1 (v1.75–1.77):** Kitap Rehberi 18→29 (11 Türkçe, web-doğrulamalı: Celep, Darılmaz 2023, Doğangün, Antoniou 2024, Uzsoy, Kasımzade, Chopra-TR, Day, Eyidoğan, TBDY El Kitabı). `features/KAYNAKCA.md` otomatik kaynakça + **6 yönetmelik/standart** (TBDY-2018, ÇYTHYE-2016, TS 500, Eurocode 8, ASCE 7, FEMA P-58).

**Batch 2 (v1.78–1.79):** Konular 10→26. 16 yeni konu (magnitüd, sismometre, fay tipleri, artçı/öncü, response spektrum, likefaksiyon, mikrobölgeleme, GMPE, tomografi, erken uyarı + ileri: moment/enerji, stick-slip, **deprem tahmini-neden zor**, ambient noise, b-değeri uzamsal, yavaş depremler). +5 REFERENCES (Geller, Dieterich, Kanamori-Brodsky, Shapiro, Beroza-Ide).

**Batch 4 (v1.80):** Öğrenme Yolu (9 adım: başlangıç→araştırma, her adım konu+kitap+panel+kazanım) + Sözlük (63 terim, aranabilir, Türkçe+EN+sembol+yanlış-anlama notu).

**Batch 3 (v1.81):** Güncel Araştırma 5 teknoloji cephesi (modern+klasik): ML/AI (EQTransformer, PhaseNet, DeVries), DAS fiber-optik (Zhan 2020), GNSS-EEW, EEW modern, InSAR otomasyonu. + **EBYU Deprem Teknolojileri Enstitüsü** (deprem.ebyu.edu.tr — Türkiye'nin 2. deprem enstitüsü).

**Kalite:** Sıfır uydurma DOI/ISBN (kritik ISBN/DOI'ler web-doğrulandı); kitaplardan tam metin yok (telif: kavram+bölüm+erişim); Türkiye/Erzincan odağı; CI yeşil (ruff+pytest); performans korundu (sekme içi ağır hesap yok).

---

## v1.54–v1.72 - 2026-05-26→28 — 🎓 AKADEMİK STANDART PROJESİ (büyük sürüm)

**Amaç:** Tüm uygulamayı (~35 panel / 39 açıklama bloğu) yüzeysel açıklamalardan peer-reviewed kaynaklı, formüllü, sınırlamaları açık **akademik standarda** yükseltmek. Codex "sıkılaştırılmış B" stratejisi: önce standart + 3 pilot, sonra sprint sprint.

**Faz 0 (v1.54):** `features/ACADEMIC_STANDARD.md` (6-bölüm şablon: Ne/Nasıl Okunur/Formül/Yorumlama/Sınırlamalar/Kaynaklar + kaynak disiplini + disclaimer zonları), `features/CONTENT_AUDIT.md`, `render_academic_explanation()` helper.

**Faz 1 Pilot (v1.55–1.57):** Canlı Radar · b-Değeri/Gutenberg-Richter · Elastik Geri Tepme (Reid 1910).

**Standart v1.1 (v1.58–1.59):** Türkiye örneği zorunlu, formül birimi, Geller 1997 disclaimer kapsamı, LaTeX/referans politikası, grafik-altı **mini-rehber** (kategori-renkli).

**Performans (v1.60–1.61):** Sismik Açık/Coulomb donma fix (çift GEM render → None-separated tek-trace, 730→17 trace). Fay Sistemleri 3.6M haversine cache'lendi.

**Sprint 2 (v1.62):** η Kümeleme · RTL · AMR.
**Sprint 3 (v1.63):** Erken Uyarı · ShakeMap.
**Sprint 4 (v1.64–1.65):** Plaka Simülasyonu · Coulomb Stres.
**Sprint 5 (v1.67):** Odak Mekanizması · InSAR Deformasyon · InSAR Zaman Serisi · Fay Kilitlenme · Moho · SKS Splitting.
**Sprint 6 (v1.66):** Erzincan Arşivi · Erzincan Paleo · Erzincan Mikrozon · Paleosismik Kazı · Tarihsel Sismisite.
**Sprint 7 (v1.69–1.71):** 18 panel — risk/olasılık (Sismik Açık, Sismik Döngü, Artçı Tahmin, PSHA, HAZUS) + fizik/ortam (Tsunami×3, Vs30, Fay Sistemleri, Astronomik, Dinamik Tetikleme) + eğitim/meta (Bilgi Havuzu P/S, Erzincan Sahnesi, Ambraseys, Akademik Kütüphane, Raporlar, Sistem).

**QA fix'leri:** v1.68 KaTeX mobil overflow CSS (`.katex-display{overflow-x:auto}`), v1.72 literal template sızıntısı (`v{sürüm}`, `{city}`).

**Kalite metrikleri:**
- 39 akademik açıklama bloğu + 39 mini-rehber + 39 disclaimer
- ~150 benzersiz peer-reviewed referans, **sıfır uydurma DOI**
- 198 dengeli `$$` LaTeX bloğu (KaTeX uyumlu, mobil overflow korumalı)
- Sıfır ağır-hesap ihlali (expander §4.9)
- Bilimsel namus: yöntem eleştirileri dahil (Hardebeck 2008, Kagan-Jackson 1991, Hough 2018), "deprem tahmini değildir" + "korelasyon≠nedensellik" tekrar

**Deploy fix (e23f8f8):** `earthquake_core.py`'de `filter_historical_events` + `normalize_historical_event` commitlenmemişti → origin/main ImportError ile çöküyordu. Commitlendi (15/15 test geçiyor).

---

## v1.21.1 - 2026-05-26 — Plaka Simülasyonu modlar arası 5× görsel kademe

Kullanıcı bildirimi: "10 bin yıl, 1 milyon ya da 1 milyar yıldaki hareketler aynı görünüyor, mesafeler ve kaymalar aynısı, oysa daha farklı olmalı"

**Sorunun kökü:** v1.18.2'de modlar arası görsel kayma tasarımı *kullanıcı dostu okunabilirlik* için her modun max stop'unda ~2.25° eşit görsel veriyordu (sci=10K×1000, geo=1M×10, pal=1B×0.012 hep ~2.5° görsel). Kullanıcı modlar arası **bilimsel farkı görmek istiyor**.

**Düzeltme — kademeli görsel ölçek (PALEOMAP/GPlates referansı):**

| Mod | Max yıl | Eski visual_scale | Yeni visual_scale | Max görsel kayma | Kademe |
|---|---:|---:|---:|---:|---|
| 🟢 sci | 10K | 1000 | **222** | **~0.5°** | KÜÇÜK |
| 🟡 geo | 1M | 10 | **11.1** | **~2.5°** | ORTA |
| 🔴 pal | 1B | 0.012 | **0.0222** | **~5°** | BÜYÜK |

Modlar arası **5× görsel artış**: sci (0.5°) → geo (2.5°) → pal (5°). Kullanıcı modlar arası geçince **net farkı görür**.

Mod **içinde** log-orantılı korundu — sci 1K=0.05°, 10K=0.5° (10× yıl → 10× görsel); geo 100K=0.25°, 1M=2.5° (10× → 10×); pal 100M=0.5°, 1B=5° (10× → 10×).

Bilimsel namus: Bu hâlâ approximation — gerçek tools (PALEOMAP, GPlates) gerçek mesafe + auto camera zoom kullanır. Bizim hibrit kompromis: görsel okunabilirlik + modlar arası net fark. Bilgi kutusunda gerçek metrik (250 m / 25 km / 22.500 km) zaten doğru gösteriliyor.

APP_VERSION 1.21 → 1.21.1

## v1.21 - 2026-05-26 — F-61 ShakeMap (Antigravity Ajan 8)
- USGS ShakeMap API entegrasyonu + MMI izoseist haritası (Wald 1999, Worden 2016, Bakun 1997).

## v1.20 - 2026-05-26 — F-63 KAF Sismik Açık (Antigravity Ajan 8)
- Kuzey Anadolu Fayı seismic gap haritası + Gantt zaman çizelgesi (Barka 1996, Stein 1997, Parsons 2004).

## v1.19.1 - 2026-05-26 — Plaka Simülasyonu PERF + Spekülatif mode kayma düzeltmesi

İki kullanıcı bildirimi:
1. "Simülasyon ve kaydırma çubuğu düzgün çalışmıyor" (slider yavaş)
2. "Milyar yıl hesabını yapamıyor, aynı mesafeleri tekrarlıyor" (Spekülatif mode'da küçük yıllarda hep aynı görsel)

**🚀 Performans düzeltmesi (slider yavaşlığı):**
- `_plaka_build_figure()` artık PLATE_LINES (global 241 sınır) yerine `plates_in_scope` (sadece hız vektörü tanımlı plakaların sınırları, ~10-15) üzerinde çalışır
- _PB2002_TO_VELOCITY_CODE mapping'de olan plakalar (AT/AS/EU/AF/AR) — diğer global plakalar (NA/SA/PA/IN/AU vb.) zaten hız vektörü yok, render edilse bile sabit kalır
- 241 sınır × 21 frame × 2 trace = ~10K render → ~15 × 21 × 2 = ~630 render (~16× hızlanma)
- 3 lokasyonda güncelleme: `base_lats`, `plates_by_type`, `border_dlat_yr` lookup
- Yan etki yok: önceden render edilen ama hareket etmeyen sınırlar zaten "görünmez işsizdi" → temizlik

**🔴 Spekülatif (pal) mod kayma düzeltmesi:**
- Eski stops asimetrik ve küçük yıllar (10K, 100K, 1M) içeriyordu — bu yıllarda 1B-ölçek visual_scale=0.005 ile görsel kayma ≈ 0 → "aynı tekrar" hissi
- Yeni stops 21 frame log-spaced, **sadece anlamlı görsel kayma aralığında**: ±1B, ±500M, ±200M, ±100M, ±50M, ±20M, ±10M, ±5M, ±2M, ±1M, 0
- visual_scale_factor: 0.005 → 0.012 (1B yıl 2.7° görsel kayma, daha net)
- Küçük yıllar için Bilimsel/Genişletilmiş mode kullanılmalı (label'da zaten "Eğitsel Sezgi" diyor)

APP_VERSION 1.19 → 1.19.1

## v1.19 - 2026-05-26 — F-51 Artçı Tahmin (Reasenberg-Jones + Omori-Utsu) [Antigravity Ajan 8]
- Bilim Profesörü tarafından implementasyon: Reasenberg & Jones (Science 1989) artçı olasılık + Omori-Utsu (1995) güç yasası fit.

## v1.18.2 - 2026-05-26 — Plaka Simülasyonu 3 kademe mod (kullanıcı talebi)

Kullanıcı talebi: "3 bölüm olsun, gerçek verilere dayanan kısım + diğer iki kısım"

`_PLAKA_MODES` artık 3 mod (renk-kodlu hiyerarşi, `_plaka_warning()` eşikleriyle birebir uyumlu):

| Mod | Etiket | Aralık | Stops | visual_scale | Warning rengi |
|---|---|---|---|---|---|
| **`sci`** (default) | 🟢 Bilimsel — Doğrudan Ölçüm | ±10.000 yıl | 21 frame log [-10K..+10K] | 1000 | 🟢 Bilimsel |
| `geo` | 🟡 Genişletilmiş — Paleosismik Ufuk | ±1.000.000 yıl | 21 frame log [-1M..+1M] | 10 | 🟡 Soyutlama |
| `pal` | 🔴 Spekülatif — Eğitsel Sezgi | ±1.000.000.000 yıl | 21 frame log [-1B..+1B] | 0.005 | 🔴 Spekülatif Senaryo |

- **🟢 Bilimsel (default, ilk gelir):** Lineer GNSS ekstrapolasyonu doğrudan geçerli — Reilinger 2006 GPS hız alanı tam ölçüm rejiminde. Erzincan 10K yıl × 25 mm/yıl = 250 m gerçek kayma; visual_scale=1000 ile harita üzerinde 2.25° görünür hareket.
- **🟡 Genişletilmiş:** Paleosismik kalibrasyon zonu — lineer model fay döngülerini ve viskoelastik relaksasyonu ihmal eder. 1M yıl × 2.25e-7°/yıl × 10 ≈ 2.25° görsel kayma (Bilimsel ile aynı görsel etki, farklı zaman çözünürlüğü).
- **🔴 Spekülatif:** Eğitsel sezgi — kıta sürüklenmesi non-lineer; gerçek paleotektonik için PALEOMAP/Scotese 2016 rekonstrüksiyonları gerek. Bu mode "büyük zaman ölçeğinde plakalar nereye gider" görseli.

Python dict insertion order korunur (3.7+) → "sci" en başta tanımlı = default mod. Kullanıcı uygulamayı ilk açtığında **bilimsel namus** ile başlar; sonra isterse 🟡/🔴 mod'a geçer.

Render kodunda hiçbir değişiklik yok — mode_keys = list(_PLAKA_MODES.keys()) otomatik 3 mod gelir.

APP_VERSION 1.18.1 → 1.18.2

## v1.18.1 - 2026-05-26 — Paleografik mode ±1 milyar yıl (kullanıcı talebi)

Kullanıcı talebi: "🪨 Paleografik — Spekülatif (-1.000.000 → +10.000.000 yıl), -1 milyar + 1 milyar olsun"

Değişiklikler:
- `_PLAKA_MODES["pal"]`:
  - Label: "Spekülatif (-1M → +10M)" → "**Hiper-Spekülatif** (-1 milyar → +1 milyar yıl)"
  - Stops: 20 frame [-1M..+10M] → 21 frame log-spaced [-1B, -300M, -100M, -30M, -10M, -3M, -1M, -300K, -100K, -10K, 0, +10K, +100K, +300K, +1M, +3M, +10M, +30M, +100M, +300M, +1B]
  - visual_scale_factor: 1.0 → 0.005 (zaman ufku 100x genişledi, görsel ölçek 1/200; 1B yıl × 2.25e-7°/yıl × 0.005 ≈ 1.1° görsel kayma — harita içinde kalır)

Bilimsel namus uyarısı (label'da explicit): Milyar yıl ölçeğinde lineer GNSS-türevli ekstrapolasyon **bilimsel geçerliliğini tamamen yitirir**. Kıta sürüklenmesi Pangaea'dan beri non-lineer döngülerle yürür (PALEOMAP / Scotese 2016 rekonstrüksiyonları gerekir). Bu mode yalnızca **eğitsel-sezgisel** "büyük zaman ölçeğinde plakalar nereye gider" görselidir.

`_plaka_warning()` 1M+ için zaten "🔴 Spekülatif Senaryo" çıkarıyor — bu mode'daki tüm 1M+ stop'lar bu uyarıyı tetikler. Hassas zaman çözünürlüğü için Jeodezik mode (±1M) tercih edilmeli.

## v1.18 - 2026-05-26 — Erzincan Arşivi menüsü + Akademik Özellik Backlog (Antigravity)
- Yeni menü öğesi: "🏛️ Erzincan Arşivi" (archive ikonu)
- features/ACADEMIC_FEATURES.md eklendi — Ajan 8 (Bilim Profesörü) peer-reviewed literatür kataloglu yeni özellik önerileri
- features/BACKLOG.md F-44..F-69 eklendi (26 yeni akademik özellik): PSHA, Vs30, GCMT, ZMAP Z-testi, PI, Omori-Utsu, Reasenberg-Jones, dinamik gerilme, vb.

## v1.17.5 - 2026-05-26 — Jeodezik mode ±1.000.000 yıl (kullanıcı talebi)

Kullanıcı talebi: "-1 milyon + 1 milyon yapalım"

Değişiklikler:
- `_PLAKA_MODES["geo"]`:
  - Label: "(-100.000 → +100.000 yıl)" → "(-1.000.000 → +1.000.000 yıl)"
  - Stops: 21 frame log-spaced [-1M, -300K, -100K, -30K, -10K, -3K, -1K, -300, -100, -10, 0, +10, +100, +300, +1K, +3K, +10K, +30K, +100K, +300K, +1M]
  - visual_scale_factor: 100 → 10 (zaman 10x genişledi, görsel ölçek 1/10; net etki ~aynı: 1M yıl × 2.25e-7°/yıl × 10 ≈ 2.25° görsel kayma)

Bilimsel namus: 1M yıl `_plaka_warning()` eşiğinde "🟡 Soyutlama" zonunun **üst sınırı** — paleosismik kalibrasyon yapılmadan görselleştirme. Üzerine (>1M) Paleografik mode'da kırmızı "🔴 Spekülatif" başlar.

Jeodezik ve Paleografik artık örtüşüyor (1M ortak nokta) — kullanıcı zaman çözünürlüğüne göre seçer: Jeodezik 1M'a kadar daha hassas log-spacing, Paleografik 10M'a kadar geniş.

## v1.17.4 - 2026-05-26 — Jeodezik mode zaman ufku genişletildi: ±100.000 yıl

Kullanıcı talebi: "Jeodezik (-1.000 → +10.000 yıl) yerine -100.000 → +100.000 olsa uygun mu?"

Değişiklikler:
- `_PLAKA_MODES["geo"]`:
  - Label: "🌐 Jeodezik — Bilimsel (-1.000 → +10.000)" → "🌐 Jeodezik — Genişletilmiş (-100.000 → +100.000 yıl)"
  - Stops: 21 frame log-spaced [-100K, -30K, -10K, -3K, -1K, -300, -100, -30, -10, -3, 0, +3, ... +100K]
  - visual_scale_factor: 1000 → 100 (zaman ufku 100x genişledi, görsel ölçek 1/100 azaltıldı,
    net görsel etki ~aynı: 100K yıl × 25 mm/yıl × 100x ≈ 2.5° kayma)
- `_plaka_warning()` eşikleri zaten uyumlu: 0..10K yeşil bilimsel, 10K..1M sarı soyutlama
  (kullanıcı 100K stop'u seçince sarı "Soyutlama" uyarısı çıkar — paleosismik kalibrasyon notu)

Bilimsel namus: SCI_REVIEW_PLAKA Q-SCI-8.3 cevabına göre 10K-1M arası "soyutlama" zonudur
(lineer GNSS ekstrapolasyon fay döngülerini ve viskoelastik relaksasyonu ihmal eder). Bu
yüzden label "Bilimsel" yerine "Genişletilmiş" — kullanıcının görsel kalemi sınırı bilsin.

## v1.17.3 - 2026-05-26 — Plaka Simülasyonu: TOPLU HAREKET (kullanıcı bildirimi)

**🌍 Bug fix — Plaka Simülasyonu artık çoklu plaka hareketini doğru gösteriyor:**

Önceki davranış (yanlış):
- Tek plaka (varsayılan Anadolu/AN) seçiliyor
- Tüm PLATE_LINES (241 sınır) o tek plakanın delta'sıyla kayıyor
- Görsel olarak: "Anadolu hızıyla bütün dünya kayıyor" yanılsaması

Yeni davranış (v1.17.3 toplu hareket):
- Tüm plakalar (AN/EU/AF/AR) AYNI ANDA, her biri kendi NNR-MORVEL56 hızıyla kayar
- Her sınır iki plakanın (PlateA + PlateB) hız ortalamasıyla deforme olur
  (sınır iki plakanın arasında olduğu için ortalama tektonik approximation'dır)
- Erzincan (focus pin) bulunduğu plakanın (AN) hızıyla kayar — sabit kaldığı
  yanılsama yok, kümülatif kayma gerçek değer
- Görsel artık gerçek plaka tektoniğine uygun: NAFZ boyunca AT-EU farkı,
  Bitlis-Zagros'ta AR-AT yaklaşması, Helenik Yay'da AF-AS subduction görünür

Teknik değişiklikler:
- `load_tectonic_plates()`: her sınır dict'ine `plate_a`, `plate_b` PB2002 kodları eklendi
- `_PB2002_TO_VELOCITY_CODE`: PB2002 ↔ hız dosyası kodu eşleme tablosu
  (AT → AN Anadolu, AS → AN Ege ≈ Anadolu, EU/AF/AR aynı)
- `_plaka_build_figure()`: frame-dışı pre-compute, her sınır için
  `border_dlat_yr[id(plate)]` ortalama hız lookup; frame'de
  `bd_lat = border_dlat_yr × cur_year × vis_scale` her sınıra ayrı uygulanır
- Performans: 241 plaka × 20 frame = 4820 hız hesabı yerine 241 pre-compute +
  4820 lookup → çok daha hızlı

Eşleme tablosu kapsamadığı global plakalar (NA, SA, PA, IN, AU vb.) için
sınırlar sabit kalır (kayma 0) — bilimsel namus: hız bilgisi yoksa hareket yok.

APP_VERSION 1.17.2 -> 1.17.3

## v1.17.2 - 2026-05-26 — Özel gün sayısı sınırı kaldırıldı
- Slider 1-365 → 1-3650 gün (≈10 yıl), number input sınırsız.
- Sync mantığı clamp ile: input slider üst sınırını aşarsa slider 3650'de durur, state gerçek değeri tutar.
- Paleoseismik 100+ yıl dönem analizleri için artık kısıt yok.

## v1.17.1 - 2026-05-26 — HOT-FIX: Python 3.9 uyumu + scroll bozulma rollback

**🚨 İki kritik sorun düzeltildi (kullanıcı bildirdi):**

1. **TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'**
   - `earthquake.py:3946` `float | None` PEP 604 syntax → Python 3.10+ özel; sistem Python 3.9.6
   - Fix: `from __future__ import annotations` her iki dosyaya (`earthquake.py`, `earthquake_core.py`)
   - Tüm tip hint'leri lazy/string evaluation → runtime eval atlanır

2. **Scroll bozulması — kaymayan sayfalar**
   - v1.17a'da `[data-testid="stMain"] + stMainBlockContainer + stVerticalBlock { overflow: visible !important }` eklenmişti
   - Bu Streamlit'in iç scroll mekanizmasını kırdı — bazı panellerde sayfa kaymıyordu
   - Rollback: overflow override'ları KALDIRILDI; sticky soft tarz (override'sız) bırakıldı
   - Trade-off: sticky pill bar tarayıcıya/zoom'a göre çalışabilir-çalışmayabilir, ama scroll kesin çalışır
   - Sticky kalıcı çözümü ileriki bir sürümde `position: fixed` ile yeniden ele alınacak

APP_VERSION 1.17 -> 1.17.1

## v1.17 - 2026-05-26 — F-43 Tektonik Plaka Hareketi Simülasyonu + sticky overflow fix

**🌍 Plaka Hareketi Simülasyonu paneli (Ajan 4 + 5 + 8 + 9, F-43):**
- Yeni "🌍 Plaka Simülasyonu" menü öğesi (Fay Sistemleri yanına, `globe-americas` ikonu).
- 10 / 100 / 1.000 / 10.000 yıl zaman ufukları seçilebilir; 20 frame Plotly linear interpolation animasyonu.
- 4 odak şehir: Erzincan / İstanbul / Diyarbakır / Van (sabit pin, plakalar etrafında kayar).
- Görselleştirme: statik gri "bugünkü sınırlar" + kayan turuncu "N yıl sonrası" çizgileri + sarı hız vektör oku + sabit şehir pin.
- Görsel ölçek ×1000 (Euler-türevli mm/yıl değerleri büyük zoom'da görünür); info kutusu gerçek mm/m/km değerini gösterir.
- `carto-darkmatter` mapbox tabanı, koyu tema uyumlu.

**📁 Veri & Bilim altyapısı:**
- **`data/plate_velocities.json`** — NNR-MORVEL56 Euler kutbu parametreleri (Argus, Gordon & DeMets 2011 G-cubed, Tablo 1): Anadolu, Avrasya, Afrika, Arabistan plakaları (`euler_lat`, `euler_lon`, `omega_deg_myr`).
- **`earthquake_core.py` +150 satır** — Euler rotasyon math: `plate_velocity_at_point()`, `_sph_to_cart()`, `_cross()`, kartezyen ECEF dönüşümleri.
- **`features/MISSION_PLAKA_SIMUASYON.md`** — Ajan 9 (Tasarım Master) brief: özellik tanımı, etkileşim akışı, Q-SCI-8.1..6 + Q-UI-8.1..N soru listeleri, Ajan 4/5 görev dağılımı.
- **`features/SCI_REVIEW_PLAKA.md`** — Ajan 8 (Bilim Profesörü) bilimsel fizibilite raporu: PB2002 + NNR-MORVEL56 + Reilinger 2006 yetersizlik analizi, KAF kayma hızları (Marmara 24/Niksar 20/Erzincan 18 mm/yıl), 1939 Erzincan paleoseismik kalibrasyon.
- **`features/BACKLOG.md`** — F-43 backlog girişi eklendi.

**📌 Sticky pill bar overflow düzeltmesi (UI Uzmanı — v1.16 takip):**
- v1.16'da `overflow-x: clip` konmuştu — kullanıcı bildirdi: pill bar scroll'da görünmüyor.
- CSS spec analizi: `clip` da tıpkı `hidden/auto/scroll` gibi sticky context'ini kırar.
- Fix: `[data-testid="stMain"]` + `stMainBlockContainer` + `stVerticalBlock` = `overflow: visible !important`. Scroll `stApp`/`body` seviyesinde kalır, sticky çalışır.
- v1.16'daki Antigravity'nin agresif `position: -webkit-sticky !important + z-index:9999 + box-shadow + tema-aware {BG}` styling korundu.

## v1.16 - 2026-05-26 — Sticky scroll fix + in_view bbox optimizasyonu

**📌 Sticky pill bar tam çalıştırıldı (UI Uzmanı):**
- v1.15c'de sticky CSS eklendi ama Streamlit'in varsayılan `overflow:hidden` kuralı `position:sticky`'yi kırıyordu.
- Fix: `[data-testid="stMain"] { overflow-x: clip }` ve `[data-testid="stMainBlockContainer"], [data-testid="stVerticalBlock"] { overflow: visible }` kuralları eklendi.
- Sticky CSS güçlendirildi: `position: -webkit-sticky` Safari uyumlu, `z-index: 9999`, `box-shadow: 0 4px 16px rgba(0,0,0,0.55)` (scroll'da pill bar'ın altındaki içerikten ayrıştığını net göster).
- Tema-aware arka plan `{BG}` değişkenine bağlandı (önceki sürümde hardcoded `#0e1117`/`#ffffff`).

**🚀 in_view bbox optimizasyonu (Render Uzmanı — v1.9 takip işi):**
- `_render_canli_radar` ve Bilgi Havuzu Erzincan haritasındaki `in_view(fault)` fonksiyonu `any(lat_min <= la <= lat_max for la in fault["lats"])` Python generator kullanıyordu — 14,500 fay × ortalama 10 vertex = ~145K karşılaştırma.
- v1.9'da eklenmiş `fault["min_lat"]/max_lat/min_lon/max_lon` precompute'u **kullanılmıyordu** (eksik takip).
- Yeni: bbox-overlap testi (`fault["max_lat"] >= lat_min and ...`) — 14,500 fay × 4 op = 58K; üstelik Python `any()` generator iterasyon overhead'i de ortadan kalkar — **~10× hızlanma**.
- Bonus: Bbox-overlap testi mantıksal olarak daha doğru — segmenti yalnızca *uçları* view dışındaysa eski kod gözden kaçırabiliyordu, yeni kod kesişimi yakalar.

## v1.15c - 2026-05-25 — UI Uzmanı + Render Uzmanı: Sticky pill + Sidebar reorg + Plaka glyph (F-32)

**🎨 UI iyileştirmeleri (UI Uzmanı):**
- 📌 **ANA MENÜ pill bar artık yapışkan (sticky)** — sayfa scroll edildiğinde üstte sabit kalır.
  Teknik: `st.container(key="sticky_nav")` + CSS `.st-key-sticky_nav { position: sticky; top: 0; z-index: 999; }`.
  Tema-aware arka plan (`#0e1117` koyu / `#ffffff` aydınlık) + alt ayraç çizgisi (1px solid).
  Gerekçe: v1.15a'da pill bar üste taşındı ama scroll edildiğinde kaybolup panel değiştirmek için scroll-to-top gerektiriyordu.
  Sismolojik ergonomi: 1000+ deprem listesinde gezinirken panel geçişi tek tıkla — bilişsel kesinti yok.

**🗺️ Plaka sınırı tip glyph stilizasyonu (Render Uzmanı / Ajan 5 — F-32):**
- Çizgi vertex'leri boyunca eşit aralıklı Unicode glyph dekoratörleri Plotly Scattermapbox'a eklendi.
- **▲ convergent** (her ~80 km) — kırmızı subduction üçgenleri (Helenik Yay, Kıbrıs Yayı, Bitlis-Zagros vb.)
- **◇ divergent** (her ~100 km) — mavi rift baklavaları (Batı Anadolu açılma, Mid-Atlantic Ridge vb.)
- **↔ transform** (her ~120 km) — sarı yanal kayma okları (NAFZ, EAFZ, Ölüdeniz, San Andreas vb.)
- Mapbox tile'larda native marker symbol desteği sınırlı olduğundan text glyph yöntemi tercih edildi.
- Yardımcı `_sample_along_plates()` haversine ile çizgi boyunca km bazlı sampling yapıyor.

**🧭 Sidebar reorganizasyonu (UI Uzmanı):**
- 🧭 Sidebar yeniden yapılandırıldı — Filtreler açıkta + 3 expander (Görünüm / Veri Kaynakları / Sistem).
- **Filtreler (her zaman görünür):** Yarıçap (km), Min. Büyüklük, Zaman Aralığı, Otomatik Yenileme.
  Bunlar kullanıcının en sık değiştirdiği parametreler — açıkta kalır.
- **🎨 Görünüm expander (collapsed):** Tema toggle, Harita Stili, Fay Hatları toggle, Plaka Sınırları toggle.
- **📡 Veri Kaynakları expander (collapsed):** 9 sismolojik ağ checkbox (USGS, EMSC, AFAD, Kandilli, GFZ, IRIS, INGV, USGS-Fast, AFAD-Web).
- **ℹ️ Sistem & Sürüm Notları expander (collapsed):** Sürüm bilgisi + v1.12-v1.15b kazanım özeti.
- Gerekçe: Sidebar kalabalık görünümünü temizler (önceden ~120 px görünür satır vardı), kullanıcının bilişsel yükünü azaltır.
  Filtreler ana karar noktası olduğundan üstte tutuldu (BACKLOG.md F-31 ile uyumlu).

## v1.15a - 2026-05-25 — UI Uzmanı: ANA MENÜ üst horizontal pill
- 🎨 UI Uzmanı kararı (PLAN_v2.md): ANA MENÜ sidebar altından **ana içerik üstündeki horizontal pill bar**'a taşındı.
- `streamlit-option-menu>=0.3.13` bağımlılığı eklendi (`requirements.txt`).
- 8 panel (Canlı Radar / İstatistik & Analiz / Fay / Astronomi / Erken Uyarı / Bilgi / Sistem / Rapor) Bootstrap ikonları ile pill bar olarak metrik kartlarının üstünde.
- Gerekçe: F-shaped reading + Fitts's Law — sidebar-alt konumu kullanıcıyı 800-1200 px scroll'a zorluyordu; üst pill bar mesafesi ~60 px sabit.
- Sismolog gözüyle: panel geçişlerinde göz "yo-yo" hareketi ortadan kalktı; bilişsel yük filtre kararlarına ayrılabilir hale geldi.
- v1.15b (sıradaki): sidebar 720-751 bölgesi 3 `st.expander` (Görünüm / Veri Kaynakları / Sistem) altına toplanacak.
- features/BACKLOG.md eklendi (Tasarım Master / Ajan 9): 42 feature'lık detaylı yol haritası, peer-reviewed sismolojik atıflarla.

## v1.14 - 2026-05-25 — Tektonik Veri Uzmanı (Ajan 4 — Adım A)
- PB2002 plaka sınırı tipi sınıflandırma sistemi eklendi (convergent / divergent / transform / unknown).
- 8 Türkiye + 12 dünya sınırı manuel etiketlendi: NAFZ, EAFZ, Helenik Yay, Ölüdeniz, Bitlis-Zagros, Mid-Atlantic, San Andreas, Himalaya, Andlar vb.
- Render artık tip bazında 4 ayrı trace + legend: kırmızı (subduction/collision), mavi (rift), sarı (strike-slip), gri (sınıflandırılmamış).
- Plaka sınırları varsayılan AÇIK (önceden kapalıydı, kullanıcı keşfetmiyordu).
- Hover tooltip'te plaka çifti + Türkçe sınır adı + tip açıklaması.

## v1.13 - 2026-05-25 — Vektörleştirme Uzmanı (Ajan 3)
- `fetch_all` içinde `df.apply(haversine, axis=1)` → NumPy vektörleştirilmiş radyan-matematik (~28× hızlanma, 1000 satır 8.4ms → 0.3ms).
- Dedup loop O(n²) → O(n × k) sliding-window (descending-sorted zaman array'i üzerinde 120s pencereli, doğruluk testinde 200 satır eski/yeni birebir aynı 150 unique).
- Vektörleştirme + dedup birlikte: cache-miss anında yaşanan saniye-seviyesi takılma kalkar.

## v1.12 - 2026-05-25 — Fragment Mimarı (Ajan 2)
- 3 ağır panel @st.fragment ile sarıldı: Canlı Radar, İstatistik & Analiz (üst + alt iki blok), Astronomik Analiz.
- Panel-içi widget etkileşimleri ("Çalıştır" butonu, sub-tab selectbox) sadece o fragment'i re-run eder; üst script ve diğer panel state'leri korunur.
- `load_fault_lines` ve `load_tectonic_plates` `@st.cache_data` → `@st.cache_resource` (her menü geçişindeki ~40-60ms hash overhead ortadan kalktı).
- `fetch_all` TTL 120s → 600s (cache-miss frekansı 5× azaldı).

## v1.11 - 2026-05-25
- Added `🚨 Erken Uyarı` sidebar menu with P/S wave countdown simulator.
- 15 major Turkish cities dropdown + manual coordinate input as observer location.
- Computes hypocenter distance (3D) and arrival times for P (6.0 km/s), S (3.5 km/s), Rayleigh (2.5 km/s) waves.
- Highlighted **S − P warning window** card with 4-tier color coding and action message.
- MMI intensity estimate using simplified GMPE `I = 1.5·M − 1.5·log10(R) − 3.5` (5 intensity bands).
- Plotly timeline visualization: arrival markers, warning band, danger band.
- 3 educational "Bu Süre İçinde Ne Yapmalı?" cards (Drop-Cover-Hold / Avoid windows / Vehicle-outdoor).
- Edge case warnings: epicenter (<5 km), deep focus (>200 km), far-field (>500 km).
- Clear disclaimer that this is a concept demo only, not a replacement for AFAD-EQE.

## v1.10 - 2026-05-25
- Added `🔭 Astronomik Analiz` sidebar menu with 5-component panel exploring lunar/solar/planetary correlations with seismicity.
- Component 1: Real-time celestial state cards (Moon phase, distance, altitude; Sun, Jupiter, Venus; syzygy/quadrature alignment).
- Component 2: Moon phase vs earthquake magnitude scatter plot with spring-tide bands.
- Component 3: Lunar gravitational influence time series overlaid with earthquake events.
- Component 4: FFT periodogram of daily earthquake counts with reference lines at lunar (29.5d), semi-lunar M2 (14.8d), annual (365.25d) periods.
- Component 5: Planetary gravitational influence (Jupiter + Venus, mass-weighted 1/d²) time series with event overlay.
- Heavy `ephem` calculations gated behind explicit "Çalıştır" button to protect CPU.
- Scientific honesty: explicit non-causality warnings citing Cochran 2004 and Métivier 2009.

## v1.9 - 2026-05-25
- Refactored top navigation: tabs → sidebar radio menu for true lazy loading (only active section renders).
- Sub-tabs (η Clustering / RTL Silence / AMR Power-Law / b-Value Map) under İstatistik & Analiz converted to selectbox lazy loading.
- Pre-computed bounding boxes (`min_lat`, `max_lat`, `min_lon`, `max_lon`) for fault segments in `load_fault_lines()`.
- `nearest_fault_vertex_distance()` now uses bbox pre-filter with safe-margin (111 km/° latitude, 80 km/° longitude) — typical 5-10× speedup on large fault catalogs.
- `calc_b_grid_cache()` haversine inner loop vectorized with NumPy (replacing `df.apply` Python loop) — ~100× speedup on dense spatial grids.

## v1.8 - 2026-04-30
- UX & performance: explicit "Çalıştır" button on heavy `İstatistik & Analiz` computations to prevent autorefresh-triggered freezes.
- API client timeouts tuned to keep UI responsive when one source is slow.

## v1.7 - 2026-04-29
- Added astronomical features to correlation matrix via the `ephem` Python library (Moon distance/gravity proxy, Sun distance, Jupiter + Venus mass-weighted influence).
- Added `mevsim` (DoY sinusoid), `sicaklik` (climate-model temperature proxy), and `haftalik_aktivite` environmental features.
- General performance optimizations across the radar and analysis tabs.
- Fix: pulled API request timeouts down to 4.0s and added a Run button to `stats_tab` to prevent UI locking.

## v1.6 - 2026-04-28
- Improved `Bilgi Havuzu` wave education with a clearer observer station, 3D subsurface cross-section, P/S particle-motion arrows, and Rayleigh surface-particle ellipses.
- Rebuilt the Erzincan virtual impact view as an oblique 3D basin scene with terrain deformation, fault trace, real event markers, and animated wavefronts.

## v1.5 - 2026-04-28
- Fixed Streamlit hot-reload/cache issue that could keep an older `earthquake_core` module in memory and break `event_signature` imports.
- Restarted the main local app on port 8560 with the current code path.

## v1.4 - 2026-04-28
- Fixed `Olay Detayı` event selection so the chosen earthquake remains stable across refreshes.
- Rebuilt `Bilgi Havuzu` with 3D fault-mechanics, P/S/Rayleigh wave, and Erzincan virtual impact-map simulations.
- Added explicit educational-use caveats to separate schematic simulations from scientific analysis outputs.

## v1.3 - 2026-04-28
- Added `Bilgi Havuzu` education screen with schematic fault-motion simulations.
- Added `Veri Kalitesi` screen documenting source counts, deduplication tolerances, and analysis-catalog caveats.
- Strengthened auto-refresh fade suppression for Streamlit stale elements.

## v1.2 - 2026-04-28
- Added dedicated screens for source health, event detail, fault analysis, activity/alarm, and reporting.
- Improved refresh comfort by suppressing Streamlit stale element fade during auto-refresh.
- Added version tracking with `VERSION`, `APP_VERSION`, and visible app labels.

## v1.1 - 2026-04-28
- Added USGS Fast Feed source.
- Added flexible time range controls.
- Split radar, scientific analysis, and data table into clearer tabs.
- Added source coverage and cumulative energy analysis.
