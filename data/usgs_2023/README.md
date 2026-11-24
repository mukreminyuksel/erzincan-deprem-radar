# 6 Şubat 2023 Kahramanmaraş yüzey kırıkları (USGS)

Kaynak: Reitman, N.G., Collett, C., Mejstrik, J.D., Koehler, R.D., Akçiz, S., Gold, R.D.,
Hatem, A.E., DuRoss, C.B., Jobe, J.A.T., Barnhart, W.D., Briggs, R.W., 2024, *Fault rupture
mapping of the 6 February 2023 Kahramanmaraş, Türkiye, earthquake sequence from satellite data*
(ver. 1.1, February 2024): U.S. Geological Survey data release, https://doi.org/10.5066/P985I7U2

Lisans: ABD Devleti verisi, kamu malı (public domain).

- `kahramanmaras_2023_rupture.geojson` — WorldView uydu görüntülerinden 1:1.500 ölçekte çizilmiş
  ayrıntılı yüzey kırığı (2.262 çizgi, nokta aralığı medyan ~13 m). `deprem_mw`: 7.8 (Pazarcık)
  veya 7.5 (Elbistan).
- `kahramanmaras_2023_simple.geojson` — InSAR'dan sadeleştirilmiş fay izi.

USGS notu: ayrıntılı çizim, yüksek çözünürlüklü görüntü bulunmayan, bulutlu/karlı ya da eğik
açılı görüntülü kesimlerde eksiktir.

Üretim: `python tools/build_usgs_2023_ruptures.py` (ham zip'ler `_raw/` altında, git'e girmez).
