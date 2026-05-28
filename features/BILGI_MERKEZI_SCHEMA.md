# Akademik Bilgi Merkezi — İçerik Şeması ve Kaynak Standardı (Faz B / Batch 0)

**Tarih:** 2026-05-28
**Bağlayıcı:** Faz B'nin tüm batch'leri (Kitap Rehberi, Konular, Araştırma, Öğrenme Yolu) bu şemaya uymak ZORUNDADIR.
**İlgili:** [ACADEMIC_STANDARD.md](ACADEMIC_STANDARD.md) (panel-açıklama standardı) — bu doküman onun *eğitim/kaynak* tamamlayıcısıdır.

---

## 0. Mimari

Mevcut `📚 Akademik Kütüphane` paneli → **4 üst-sekmeli "Akademik Bilgi Merkezi"**:

```
📚 Akademik Bilgi Merkezi   (earthquake.py: _render_akademik_kutuphane)
 ├── 📖 Konular        → knowledge_base.TOPICS        (Batch 2: 10 → 25+)
 ├── 📚 Kitap Rehberi  → knowledge_base.BOOKS         (Batch 1)
 ├── 🔬 Güncel Araştırma & Teknoloji → knowledge_base.RESEARCH  (Batch 3)
 └── 🧭 Öğrenme Yolu + 📔 Sözlük → knowledge_base.LEARNING_PATH + GLOSSARY (Batch 4)
```

İçerik `knowledge_base.py`'de veri yapısı; render `earthquake.py`'de. **Kod ve içerik ayrı.**

---

## 1. Veri Şemaları

### 1.1 `BOOKS` — Kitap Rehberi (dict: id → kayıt)

| Alan | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| `baslik` | str | ✅ | Kitap adı |
| `yazar` | str | ✅ | "Soyad, A. & Soyad, B." |
| `yil` | int | ✅ | Baskı yılı |
| `baski` | str | — | "2nd ed." vb. |
| `isbn` | str | ✅ | Gerçek ISBN (doğrulanabilir) |
| `seviye` | str | ✅ | Başlangıç / Orta / İleri |
| `kategori` | str | ✅ | Sismoloji / Fay Mekaniği / Deprem Mühendisliği / Jeodezi-Uzaktan Algılama / Türkiye |
| `konu_etiketleri` | list[str] | ✅ | TOPICS key'leri (ör. `["sismik_dalgalar","moment_tensor"]`) |
| `neden_okunmali` | str | ✅ | 1-2 cümle: bu kitap neden değerli |
| `bolum_onerileri` | str | ✅ | "Bölüm 4 (kaynak teorisi), Bölüm 7 (yüzey dalgaları)" |
| `ilgili_paneller` | list[str] | ✅ | Uygulamadaki ilgili panel adları |
| `turkiye_iliskisi` | str\|None | ✅ | Türkiye/Erzincan bağlantısı veya `None` |
| `erisim` | str | ✅ | "Ücretli / kütüphane", "Açık erişim PDF", "Online" |
| `telif_notu` | str | ✅ | Sabit: tam metin/uzun alıntı YOK — kavram+bölüm önerisi |

### 1.2 `RESEARCH` — Güncel Araştırma & Teknoloji (dict: id → kayıt)

| Alan | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| `baslik` | str | ✅ | Makale/teknoloji başlığı |
| `yazar` | str | ✅ | "Soyad, A., et al." |
| `yil` | int | ✅ | Yayın yılı |
| `doi` | str | ✅ | Gerçek DOI (peer-reviewed) |
| `kaynak` | str | ✅ | Dergi (Nature, Science, BSSA, JGR vb.) |
| `kategori` | str | ✅ | Makine Öğrenmesi / DAS Fiber-Optik / GNSS-Jeodezi / Erken Uyarı / InSAR-Uzaktan Algılama / Diğer |
| `tip` | str | ✅ | **`modern`** (son 5 yıl, yeni teknoloji) veya **`klasik`** (temel yöntem, öncü makale) |
| `konu_etiketleri` | list[str] | ✅ | TOPICS key'leri |
| `ozet` | str | ✅ | 2-4 cümle, kendi kelimelerimizle (kopya değil) |
| `ilgili_paneller` | list[str] | ✅ | İlgili panel adları |
| `turkiye_iliskisi` | str\|None | ✅ | Türkiye/Erzincan bağlantısı veya `None` |

**KURAL (kullanıcı):** "Modern teknoloji için son 5 yıl, temel yöntem için klasik kaynaklar **birlikte** verilsin." Her teknoloji kategorisinde hem `modern` hem `klasik` tip bulunmalı (ör. ML deprem: DeVries 2018 modern + Geller 1997 klasik sınır).

### 1.3 `LEARNING_PATH` — Öğrenme Yolu (list: sıralı adımlar)

| Alan | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| `adim` | int | ✅ | Sıra numarası (1'den) |
| `baslik` | str | ✅ | "Deprem nedir? Sismik dalgalar" |
| `seviye` | str | ✅ | Başlangıç / Orta / İleri |
| `aciklama` | str | ✅ | Bu adımda ne öğrenilir |
| `konular` | list[str] | ✅ | TOPICS key'leri |
| `kitaplar` | list[str] | — | BOOKS id'leri |
| `paneller` | list[str] | — | Uygulamada keşfedilecek paneller |
| `tahmini_sure` | str | — | "1-2 saat" |

### 1.4 `GLOSSARY` — Terim Sözlüğü (dict: id → kayıt)

| Alan | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| `terim` | str | ✅ | Türkçe terim |
| `en` | str | ✅ | İngilizce karşılığı |
| `tanim` | str | ✅ | 1-2 cümle açık tanım |
| `ilgili_konu` | str\|None | ✅ | TOPICS key veya `None` |

---

## 2. Kaynak Disiplini (ACADEMIC_STANDARD §2 ile aynı katılık)

1. **Gerçek kaynak:** Her kitap gerçek ISBN; her araştırma gerçek DOI. **Uydurma yasak.**
2. **Telif:** Kitaplardan **tam metin / uzun alıntı YOK.** Sadece: bölüm önerisi, kavram haritası, "neden okunmalı", uygulamadaki ilgili paneller. Özet kendi kelimelerimizle.
3. **Türkiye odağı:** Mümkün her kayıtta Türkiye/Erzincan ilişkisi (veya açıkça `None`).
4. **Modern + klasik birlikte:** Araştırma bölümünde her teknoloji için hem güncel (son 5 yıl) hem klasik temel kaynak.
5. **Panel bağlantısı:** Her kayıt uygulamadaki ilgili panellere referans verir (öğrenme→uygulama köprüsü).

---

## 3. Render Sözleşmesi

- Üst seviye: `st.tabs(["📖 Konular", "📚 Kitap Rehberi", "🔬 Güncel Araştırma & Teknoloji", "🧭 Öğrenme Yolu + Sözlük"])`
- Her sekme kendi veri yapısını filtreli (kategori/seviye) kart/liste olarak gösterir.
- §4.9 (ACADEMIC_STANDARD): sekme içinde ağır hesap yok — sadece metin/markdown/hafif filtre.
- Boş veri yapısı → "içerik hazırlanıyor" placeholder (Batch 0'da sekmeler iskelet).

---

## 4. Batch Planı

- **Batch 0** (bu): şema + veri yapısı iskeleti (1-2 örnek kayıt) + 4-sekme render iskeleti.
- **Batch 1:** BOOKS — 10-12 temel kitap.
- **Batch 2:** TOPICS 10 → 25+ konu.
- **Batch 3:** RESEARCH — ML/DAS/GNSS/EEW (her biri modern+klasik).
- **Batch 4:** LEARNING_PATH + GLOSSARY.

---

## 5. Versiyon

- **v1.0** (2026-05-28) — Batch 0 şema. Kullanıcı onaylı: tek 4-sekmeli merkez, Codex'siz, batch batch.
