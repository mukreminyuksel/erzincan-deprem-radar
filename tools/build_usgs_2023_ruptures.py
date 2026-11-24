"""USGS 6 Şubat 2023 Kahramanmaraş yüzey kırığı verisini uygulama biçimine getirir.

Kaynak (kamu malı, ABD Devleti verisi):
  Reitman, N.G., Collett, C., Mejstrik, J.D., Koehler, R.D., Akçiz, S., Gold, R.D.,
  Hatem, A.E., DuRoss, C.B., Jobe, J.A.T., Barnhart, W.D., Briggs, R.W., 2024,
  Fault rupture mapping of the 6 February 2023 Kahramanmaraş, Türkiye, earthquake
  sequence from satellite data (ver. 1.1, February 2024): U.S. Geological Survey
  data release, https://doi.org/10.5066/P985I7U2
  - surface_rupture_lines: WorldView görüntülerinden 1:1.500 ölçekte, ayrıntılı kırık
  - simple_fault: InSAR'dan sadeleştirilmiş fay izi (1:750.000)
Not: Ayrıntılı çizim bulut/kar/eğik görüntü yüzünden bazı kesimlerde eksiktir (USGS).

Girdi:  data/usgs_2023/_raw/<zip içerikleri> (git'e girmez)
Çıktı:  data/usgs_2023/kahramanmaras_2023_rupture.geojson
        data/usgs_2023/kahramanmaras_2023_simple.geojson
Koordinatlar 6 ondalığa (~0,1 m) yuvarlanır; başka değişiklik yok.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "usgs_2023"
SRC = {
    "kahramanmaras_2023_rupture.geojson":
        "_raw/surface_rupture_lines_v1.1_202402_allformats/surface_rupture_lines_v1.1_202402.geojson",
    "kahramanmaras_2023_simple.geojson":
        "_raw/simple_fault_v1.1_202402_allformats/simple_fault_v1.1_202402.geojson",
}


def _round(coords):
    if not coords:
        return coords
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], 6), round(coords[1], 6)]
    return [_round(c) for c in coords]


def main():
    for out_name, src in SRC.items():
        gj = json.loads((ROOT / src).read_text(encoding="utf-8"))
        feats = [{"type": "Feature",
                  "properties": {"deprem_mw": f["properties"].get("earthquake")},
                  "geometry": {"type": f["geometry"]["type"],
                               "coordinates": _round(f["geometry"]["coordinates"])}}
                 for f in gj["features"]
                if f.get("geometry") and f["geometry"].get("coordinates")]  # boş geometriyi atla
        out = {"type": "FeatureCollection",
               "kaynak": "Reitman vd. 2024, USGS data release, doi:10.5066/P985I7U2 (ver. 1.1, kamu malı)",
               "features": feats}
        (ROOT / out_name).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")),
                                     encoding="utf-8")
        print(f"{out_name}: {len(feats)} çizgi, {(ROOT / out_name).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
