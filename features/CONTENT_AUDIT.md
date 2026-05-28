# İçerik Denetim Raporu (Faz 0 Audit)

**Tarih:** 2026-05-26
**Hazırlayan:** Ajan koordinatörü
**Standart:** [ACADEMIC_STANDARD.md](ACADEMIC_STANDARD.md) — 6 bölüm + kaynak disiplini

---

## Metodoloji

Her panelin mevcut `st.caption`, `st.markdown`, `st.info`, `st.warning`, `st.success` bloklarındaki açıklamalar incelendi. 4 seviyeli sınıflandırma:

| Seviye | Kriter |
|---|---|
| 🔴 **Eksik** | 0-2 bölüm dolu (sadece bir cümle açıklama, kaynak yok) |
| 🟠 **Temel** | 3-4 bölüm yüzeysel, kaynak az/yok |
| 🟡 **Orta** | 5-6 bölüm, peer-reviewed atıflı ama derinlik orta |
| 🟢 **Akademik** | 6 bölüm + LaTeX + 3+ DOI + disclaimer + sınırlamalar |

---

## Panel Envanteri (35 panel, 72 grafik)

### 🔴 Canlı İzleme Kategorisi

| Panel | Mevcut Seviye | Grafik Sayısı | Pilot? |
|---|---|---:|---|
| 🌍 Canlı Radar | 🟢 Akademik ✅ (v1.55) | 5 (harita + saatlik + derinlik + scroll list + scoreboard) | ✅ **PİLOT A — Hanks-Kanamori 1979 + Sinnott 1984** |
| 🌊 ShakeMap | 🟢 Akademik ✅ (v1.63) | 3 | ✅ **Sprint 3 — Wald 1999 + Worden 2012 + Bakun-Wentworth 1997** |
| 🔴 Sismik Açık | 🟢 Akademik ✅ (v1.69) | 2 (harita + Gantt) | ✅ **Sprint 7 — McCann 1979 + Kagan-Jackson 1991 eleştiri** |
| 🚨 Erken Uyarı | 🟢 Akademik ✅ (v1.63) | 2 (timeline + skor kartları) | ✅ **Sprint 3 — Allen-Kanamori 2003 + AFAD-EWS** |
| 📈 Artçı Tahmin | 🟢 Akademik ✅ (v1.69) | 2 | ✅ **Sprint 7 — Omori-Utsu + Reasenberg-Jones 1989** |
| 🔄 Sismik Döngü | 🟢 Akademik ✅ (v1.69) | 2 | ✅ **Sprint 7 — Matthews 2002 BPT + Reid 1910** |

### 🧭 Fay & Tektonik

| Panel | Mevcut Seviye | Grafik Sayısı | Pilot? |
|---|---|---:|---|
| 🧭 Fay Sistemleri | 🟢 Akademik ✅ (v1.70) | 2 | ✅ **Sprint 7 — Emre 2013 MTA + Sinnott 1984** |
| 🌍 Plaka Simülasyonu | 🟢 Akademik ✅ (v1.64) | 1 (animasyon) | ✅ **Sprint 4a — V_rel formülü + NNR-MORVEL56** |
| 🔒 Fay Kilitlenme | 🟢 Akademik ✅ (v1.67) | 2 | ✅ **Sprint 5 — Savage-Burford 1973 + Reilinger 2006** |
| 💥 Coulomb Stres | 🟢 Akademik ✅ (v1.65) | 2 | ✅ **Sprint 4b — King-Stein-Lin 1994 + Okada 1992** |
| 🥎 Odak Mekanizması | 🟢 Akademik ✅ (v1.67) | 2 | ✅ **Sprint 5 — Dziewonski 1981 + Aki-Richards 2002** |
| ⛏️ Paleosismik Kazı | 🟢 Akademik ✅ (v1.66) | 1 | ✅ **Sprint 6 — Wallace 1981 + McCalpin 2009** |
| 🏺 Erzincan Paleo | 🟢 Akademik ✅ (v1.66) | 2 | ✅ **Sprint 6 — Kozacı 2007 + Hartleb 2003** |

### 📊 Analiz & Modeller

| Panel | Mevcut Seviye | Grafik Sayısı | Pilot? |
|---|---|---:|---|
| 📊 İstatistik & Analiz | 🟡 Orta (η/RTL/AMR akademik v1.62) | 8+ (b-grid, η, RTL, AMR, korelasyon, scatter) | ✅ **Sprint 2 — η/RTL/AMR akademik** |
| 📉 b-Değeri Zaman Serisi | 🟢 Akademik ✅ (v1.56) | 3 (b-değeri zaman + dağılım) | ✅ **PİLOT B — Aki 1965 MLE + Bayrak 2002** |
| 🌐 Dinamik Tetikleme | 🟢 Akademik ✅ (v1.70) | 2 | ✅ **Sprint 7 — Hill 1993 + van der Elst 2010** |
| 🗺️ Sismik Tehlike (PSHA) | 🟢 Akademik ✅ (v1.69) | 2 | ✅ **Sprint 7 — Cornell 1968 + SHARE 2013 + TBDY-2018** |
| 🗺️ Tsunami Tehlike | 🟢 Akademik ✅ (v1.70) | 2 | ✅ **Sprint 7 — Basili 2021 NEAMTHM18** |
| ⏱️ Tsunami Varış | 🟢 Akademik ✅ (v1.70) | 1 | ✅ **Sprint 7 — sığ-su fiziği + Titov-Synolakis 1998** |
| 🌊 Tsunami Kataloğu | 🟢 Akademik ✅ (v1.70) | 2 | ✅ **Sprint 7 — Soloviev 2000 + Stiros 2001** |
| 🏚️ HAZUS Kayıp | 🟢 Akademik ✅ (v1.69) | 3 | ✅ **Sprint 7 — FEMA 2003 + Erdik 2003 + Lagomarsino 2006** |

### 🛰️ Uydu & Jeofizik

| Panel | Mevcut Seviye | Grafik Sayısı | Pilot? |
|---|---|---:|---|
| 🛰️ InSAR Deformasyon | 🟢 Akademik ✅ (v1.67) | 2 | ✅ **Sprint 5 — Massonnet 1993 + Sentinel-1** |
| 📡 InSAR Zaman Serisi | 🟢 Akademik ✅ (v1.67) | 2 | ✅ **Sprint 5 — Ferretti 2001 PS + Berardino 2002 SBAS** |
| 🌋 Moho Derinliği | 🟢 Akademik ✅ (v1.67) | 1 | ✅ **Sprint 5 — Mohorovičić 1910 + Zhu-Kanamori 2000** |
| 🌀 SKS Splitting | 🟢 Akademik ✅ (v1.67) | 2 | ✅ **Sprint 5 — Silver-Chan 1991 + Sandvol 2003** |
| 🏔️ Vs30 Zemin | 🟢 Akademik ✅ (v1.70) | 2 | ✅ **Sprint 7 — Borcherdt 1994 + Wald-Allen 2007** |
| 🗾 Erzincan Mikrozon | 🟢 Akademik ✅ (v1.66) | 2 | ✅ **Sprint 6 — Nakamura 1989 + AFAD 2010** |

### 🎓 Eğitim & Bilgi

| Panel | Mevcut Seviye | Grafik Sayısı | Pilot? |
|---|---|---:|---|
| 📚 Akademik Kütüphane | 🟡 Orta (81 kaynak) | 0 (liste) | — |
| 🎓 Bilgi Havuzu — 3D Fay Mekaniği | 🟢 Akademik ✅ (v1.57) | 1 (animasyon) | ✅ **PİLOT C tamamlandı** — Reid 1910 + 12 referans |
| 🎓 Bilgi Havuzu — P/S/Rayleigh Dalgalar | 🟡 Orta (v1.52 güncel) | 1 (animasyon) | — |
| 🎓 Bilgi Havuzu — Erzincan Sahnesi | 🟠 Temel | 1 (3D) | — |
| 📜 Tarihsel Sismisite | 🟢 Akademik ✅ (v1.66) | 2 | ✅ **Sprint 6 — Ambraseys 2009 + Ambraseys-Jackson 2000** |
| 🏛️ Erzincan Arşivi | 🟢 Akademik ✅ (v1.66) | 3 | ✅ **Sprint 6 — Barka 1996 + Grosser 1998** |
| 🎬 Ambraseys Animasyon | 🟠 Temel | 1 | — |
| 🔭 Astronomik Analiz | 🟢 Akademik ✅ (v1.70) | 5 (5 bileşen) | ✅ **Sprint 7 — Cochran 2004 + Hough 2018 (gelgit eleştiri)** |
| 📝 Raporlar | 🔴 Eksik | 0 (TXT indir) | — |

### ⚙️ Sistem

| Panel | Mevcut Seviye | Grafik Sayısı |
|---|---|---:|
| ⚙️ Sistem & Veri | 🟠 Temel | 2 |

---

## Özet Tablo

**Güncel durum (v1.67 — Faz 1 Pilot + Sprint 2/3/4/5/6 sonrası):**

| Seviye | Panel Sayısı | % | Not |
|---|---:|---:|---|
| 🟢 Akademik | **18** | **51%** | Pilot A/B/C + Sprint 2-6 (21 render bloğu; İstatistik 4 alt-panel tek satır) |
| 🟡 Orta | 4 | 11% | Sismik Açık, PSHA, HAZUS, P/S Dalgalar, Akademik Kütüphane |
| 🟠 Temel | 11 | 32% | Kalan paneller (Sprint 7 hedefi) |
| 🔴 Eksik | 2 | 6% | Raporlar + 1 |

**İlk audit (v1.0):** 0% akademik, %71 temel. **v1.67:** %51 akademik — Faz 1 + 5 sprint ile 18 panel 🟢'ye çıkarıldı (21 `render_academic_explanation` bloğu).

**Kalan (Sprint 7 — Eğitim & meta):** Sismik Açık, Sismik Döngü, Artçı Tahmin, Dinamik Tetikleme, PSHA, Tsunami (3×), HAZUS, Vs30, Fay Sistemleri, Astronomik, Bilgi Havuzu P/S + Erzincan Sahnesi, Akademik Kütüphane, Ambraseys Animasyon, Raporlar, Sistem.

---

## Faz 1 Pilot Seçimi

Codex önerisiyle (kullanıcı onaylı) **3 pilot panel**:

### Pilot A — 🌍 Canlı Radar Haritası
**Sebep:** En çok kullanılan panel. Açıklama kalitesi tüm uygulamanın "ilk izlenim"i. Magnitude renk-boyut kodlamasının bilimsel temeli (USGS magnitude scale) ve haversine mesafe hesabı (Sinnott 1984) anlatılmalı.

**Hedef seviye:** 🟢 Akademik (6 bölüm + LaTeX haversine + 4 DOI)

### Pilot B — 📉 b-Değeri / Gutenberg-Richter
**Sebep:** Sismolojinin temel istatistiksel ilişkisi. b-değeri yorumu yaygın yanlış anlaşılır ("b<0.8 büyük deprem yakında" hatası). Aki 1965 MLE formülü + Gutenberg & Richter 1944 + Bayrak 2002 Türkiye atfı + sınırlama (Mc) detaylı verilmeli.

**Hedef seviye:** 🟢 Akademik (LaTeX denklem + MLE + 5 DOI + Türkiye Mc atfı)

### Pilot C — 🎓 Elastik Geri Tepme (Reid 1910)
**Sebep:** **Şu an 🔴 Eksik seviyesinde** — sadece 3D animasyon var, teorik metin yok. Reid 1910 elastik geri tepme teorisi (sismolojinin temel modeli, 1906 San Francisco depremi gözlemine dayalı) detaylı anlatılmalı. Stress-strain-rupture-recovery döngüsü. Sismik döngü (Schwartz & Coppersmith 1984 *characteristic earthquake* modeli) ile bağ.

**Hedef seviye:** 🟢 Akademik (Reid 1910 + Schwartz & Coppersmith 1984 + Aki & Richards 2002 Bölüm 4 + Kanamori 2004 *Annu. Rev. Earth Planet. Sci.*)

---

## Pilot Sonrası Yol Haritası

Pilot 3 panel başarılı olursa (kullanıcı onayı), aşağıdaki sırayla diğer paneller akademik standarda çıkarılacak:

**Sprint 2 — Bilim ağırlıklı (5 panel):**
- 📊 İstatistik & Analiz (η Kümeleme, RTL, AMR — peer-reviewed bilimsel paneller)
- 🌍 Plaka Simülasyonu (Eurasia-fixed referans — Reilinger 2006 detayı)
- 💥 Coulomb Stres (King-Stein-Lin 1994 + Okada 1992)
- 🚨 Erken Uyarı (Allen & Kanamori 2003 + Akkar-Bommer 2010 GMPE)
- 🌊 ShakeMap (Wald 1999 + Worden 2016)

**Sprint 3 — Tektonik & jeofizik (6 panel):**
- 🔒 Fay Kilitlenme · 🌀 SKS Splitting · 🛰️ InSAR (2×) · 🌋 Moho · 🥎 Odak Mekanizması

**Sprint 4 — Türkiye & paleo (5 panel):**
- 🏺 Erzincan Paleo · 🗾 Erzincan Mikrozon · 🏛️ Erzincan Arşivi · ⛏️ Paleosismik · 📜 Tarihsel Sismisite

**Sprint 5 — Eğitim & meta (8 panel):**
- Bilgi Havuzu P/S/Rayleigh + Erzincan Sahnesi · Akademik Kütüphane glossary · Diğerleri

---

## v1.0 Audit Kararı

Bu audit `Faz 0 — Akademik Standart ve İçerik Denetimi` çıktısıdır. **Codex onayı: yön doğru, sıkılaştırılmış B seçeneği uygulanacak.** Sonraki adım: `render_academic_explanation()` helper fonksiyonu + 3 pilot panel.
