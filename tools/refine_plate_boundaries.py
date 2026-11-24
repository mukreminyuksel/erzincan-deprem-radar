"""Türkiye'deki plaka sınırlarını MTA Diri Fay Haritası izine oturtur.

Neden: tectonic_plates.geojson (PB2002 / McClusky 2000) küresel ölçekli; noktalar
arası ortalama 46 km. Türkiye'de üç plaka sınırı aslında bilinen fay zonlarıdır
ve MTA verisinde ~80 m çözünürlükle var:
    AT-EU (Anadolu–Avrasya)  → KUZEY ANADOLU FAY ZONU
    AT-AR (Anadolu–Arabistan) → DOĞU ANADOLU FAY ZONU
    EU-AR (Arabistan–Avrasya) → GÜNEYDOĞU ANADOLU BİNDİRMESİ (Bitlis-Zagros kenedi)

Yöntem: PB2002 çizgisi 2 km aralıkla örneklenir; her örnek, 30 km içindeki
ilgili zonun gerçek MTA fay noktalarından birine taşınır. Seçimde bir önceki
noktaya yakınlık ağırlıklı tutulur (paralel kollar arasında zikzak olmasın).
MTA'nın kapsamadığı yerde (deniz, Türkiye dışı) PB2002 noktası kalır. Sonuç
Douglas-Peucker ile 50 m toleransla sadeleştirilir.

Girdi:  data/tectonic_plates_pb2002.geojson (özgün, değiştirilmez)
        turkey_faults.geojson
Çıktı:  tectonic_plates.geojson (uygulamanın okuduğu dosya)

Çalıştırma: python tools/refine_plate_boundaries.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "tectonic_plates_pb2002.geojson"
FAULTS = ROOT / "turkey_faults.geojson"
OUT = ROOT / "tectonic_plates.geojson"

ZONES = {
    "AT-EU": "KUZEY ANADOLU FAY ZONU",
    "AT-AR": "DOĞU ANADOLU FAY ZONU",
    "EU-AR": "GÜNEYDOĞU ANADOLU BİNDİRMESİ",
}
STEP_KM = 2.0
SEARCH_KM = 30.0
SIMPLIFY_KM = 0.05


def _lines(geom):
    return [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]


def _km(lon0, lat0, lons, lats):
    """Küçük alanlar için eşdikdörtgen yaklaşık uzaklık (km)."""
    return np.hypot((lons - lon0) * math.cos(math.radians(lat0)) * 111.2, (lats - lat0) * 111.2)


def _densify(coords, step_km):
    out = [tuple(coords[0][:2])]
    for (x1, y1, *_), (x2, y2, *_) in zip(coords, coords[1:]):
        d = float(_km(x1, y1, np.array([x2]), np.array([y2]))[0])
        n = max(1, int(d / step_km))
        out += [(x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n) for k in range(1, n + 1)]
    return out


def _douglas_peucker(pts, tol_km):
    if len(pts) < 3:
        return pts
    a, b = np.array(pts[0]), np.array(pts[-1])
    lat0 = math.radians((a[1] + b[1]) / 2)
    scale = np.array([math.cos(lat0) * 111.2, 111.2])
    P = (np.array(pts) - a) * scale
    ab = (b - a) * scale
    L = np.hypot(*ab)
    if L == 0:
        d = np.hypot(P[:, 0], P[:, 1])
    else:
        d = np.abs(ab[0] * P[:, 1] - ab[1] * P[:, 0]) / L
    i = int(d.argmax())
    if d[i] <= tol_km:
        return [pts[0], pts[-1]]
    return _douglas_peucker(pts[: i + 1], tol_km)[:-1] + _douglas_peucker(pts[i:], tol_km)


def snap_line(coords, zone_pts):
    """PB2002 çizgisini MTA zon noktalarına oturtur. (yeni koordinatlar, oturan oran)"""
    samples = _densify(coords, STEP_KM)
    out, along, snapped, prev = [], [], 0, None
    for i, (lon, lat) in enumerate(samples):
        # yerel teğet (sınır boyunca ilerleme yönü), km ölçeğinde
        a = samples[max(i - 1, 0)]
        b = samples[min(i + 1, len(samples) - 1)]
        cosl = math.cos(math.radians(lat))
        tx, ty = (b[0] - a[0]) * cosl, (b[1] - a[1])
        tn = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tn, ty / tn
        d_line = _km(lon, lat, zone_pts[:, 0], zone_pts[:, 1])
        cand = np.nonzero(d_line <= SEARCH_KM)[0]
        if len(cand) == 0:
            out.append((lon, lat))
            along.append(i * STEP_KM)
            prev = None
            continue
        cost = d_line[cand].copy()
        if prev is not None:
            # süreklilik: önceki seçilen noktaya ~STEP_KM uzaklıkta olanı tercih et
            d_prev = _km(prev[0], prev[1], zone_pts[cand, 0], zone_pts[cand, 1])
            cost = 0.3 * cost + np.abs(d_prev - STEP_KM)
        j = cand[int(cost.argmin())]
        pt = (float(zone_pts[j, 0]), float(zone_pts[j, 1]))
        out.append(pt)
        # sınır boyunca konum = örneğin konumu + seçilen noktanın teğet üzerindeki izdüşümü
        along.append(i * STEP_KM + ((pt[0] - lon) * cosl * 111.2 * tx + (pt[1] - lat) * 111.2 * ty))
        prev = pt
        snapped += 1
    # 1) Hep ileri: sınır boyunca geri giden (başka kola/yan faya sapan) noktaları at
    kept, last = [], -1e9
    for pt, s_km in zip(out, along):
        if s_km > last + 0.5:
            kept.append(pt)
            last = s_km
    # 2) Sivri dönüşleri buda (ardışık iki adım arası açı > 120°), değişmeyene dek
    changed = True
    while changed and len(kept) > 3:
        changed = False
        res = [kept[0]]
        for k in range(1, len(kept) - 1):
            a, b, c = res[-1], kept[k], kept[k + 1]
            cl = math.cos(math.radians(b[1]))
            v1 = ((b[0] - a[0]) * cl, b[1] - a[1])
            v2 = ((c[0] - b[0]) * cl, c[1] - b[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 and n2 and (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2) < -0.5:
                changed = True
                continue
            res.append(b)
        res.append(kept[-1])
        kept = res
    return _douglas_peucker(kept, SIMPLIFY_KM), snapped / len(samples)


def main():
    pb = json.loads(SRC.read_text(encoding="utf-8"))
    faults = json.loads(FAULTS.read_text(encoding="utf-8"))["features"]
    zone_pts = {}
    for code, zone in ZONES.items():
        pts = [p[:2] for f in faults if (f["properties"].get("fay_adi") or "") == zone
               for line in _lines(f["geometry"]) for p in line]
        zone_pts[code] = np.array(pts, dtype=float)
        print(f"{code}: {zone} — {len(pts)} MTA noktası")
    for feat in pb["features"]:
        name = feat["properties"].get("Name", "")
        code = name.replace("/", "-")
        if code not in ZONES or len(zone_pts[code]) == 0:
            continue
        new_lines, ratios = [], []
        for line in _lines(feat["geometry"]):
            coords, ratio = snap_line(line, zone_pts[code])
            new_lines.append([[round(x, 5), round(y, 5)] for x, y in coords])
            ratios.append(ratio)
        n_old = sum(len(x) for x in _lines(feat["geometry"]))
        n_new = sum(len(x) for x in new_lines)
        feat["geometry"] = ({"type": "LineString", "coordinates": new_lines[0]} if len(new_lines) == 1
                            else {"type": "MultiLineString", "coordinates": new_lines})
        feat["properties"]["Source"] = (f'{feat["properties"].get("Source", "")}; Türkiye kesimi '
                                        f'MTA Diri Fay Haritası 2013 ({ZONES[code]}) izine oturtuldu')
        print(f"  {name}: {n_old} → {n_new} nokta, örneklerin %{100 * min(ratios):.0f}-"
              f"{100 * max(ratios):.0f}'i MTA izine oturdu")
    OUT.write_text(json.dumps(pb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Yazıldı: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
