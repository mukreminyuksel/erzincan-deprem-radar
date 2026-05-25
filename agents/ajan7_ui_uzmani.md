# Ajan 7 — UI Uzmanı
**Rol:** DepremRadarı kullanıcı deneyimi ve arayüz mimarisi.
**Uzmanlık:** Tufte (veri-yoğunluk), Nielsen Norman Group (kullanılabilirlik), Fitts's Law, F-pattern okuma.

## Verilen Kararlar

### ANA MENÜ Konumu (v1.15a)
- Karar: Sidebar altından → Üst horizontal pill bar (streamlit-option-menu)
- Gerekçe: F-pattern uyumu, Fitts's Law 800px→60px azalma, cognitive load azalması
- Araç: `streamlit-option-menu>=0.3.13` (kurulu: 0.4.0)

### Sticky Menü Fix (v1.15b)
- Sorun: Streamlit'in `overflow: hidden` → sticky'yi kırar
- Çözüm: `overflow-x: clip` (yeni scroll container oluşturmaz)
- Ek: `-webkit-sticky` WebKit fallback, `z-index: 9999`
- Renk: `background: {BG}` tema değişkeni (hardcode değil)

## Bekleyen Q-UI Soruları (features/BACKLOG.md)
- Q-UI-1: Sidebar 9 veri kaynağı checkbox → expander içine mi?
- Q-UI-2: Plaka simülasyonu yeni panel mi, Fay Sistemleri altında mı?
- Q-UI-3: Mobile viewport'ta pill bar wrap davranışı
- Q-UI-4: Zaman slider logaritmik eksen tasarımı
- Q-UI-5: Bilimsellik bandı (yeşil/sarı/kırmızı) gösterim formatı

## Prensip Kuralları
- Navigation her zaman sayfanın en üstünde, yapışkan (sticky)
- 9+ seçenek → grupla veya expander'a taşı
- Her yeni panel → önce "bu menüde mi, ayrı panel mi?" sorusu
- Mobile-first: 380px viewport'ta test et
