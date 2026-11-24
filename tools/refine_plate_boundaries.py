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
# Öncelikli iz: 6 Şubat 2023 Mw 7.8 yüzey kırığı (USGS, 1:1.500) DAF'ın 2023'te kırılan
# kesimini metre ölçeğinde verir; o kesimde MTA yerine bu iz kullanılır.
RUPTURE_2023 = ROOT / "data" / "usgs_2023" / "kahramanmaras_2023_rupture.geojson"
PRIORITY = {"AT-AR": 7.8}
PRIORITY_SEARCH_KM = 10.0
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


def snap_line(coords, zone_pts, prio_pts=None, search_km=SEARCH_KM):
    """PB2002 çizgisini MTA zon noktalarına oturtur. (yeni koordinatlar, oturan oran)

    prio_pts verilirse (ör. 2023 yüzey kırığı) ve örneğe PRIORITY_SEARCH_KM içindeyse
    önce onlar kullanılır."""
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
        pts = zone_pts
        if prio_pts is not None and len(prio_pts):
            d_prio = _km(lon, lat, prio_pts[:, 0], prio_pts[:, 1])
            if (d_prio <= PRIORITY_SEARCH_KM).any():
                pts = prio_pts
        d_line = _km(lon, lat, pts[:, 0], pts[:, 1])
        cand = np.nonzero(d_line <= (PRIORITY_SEARCH_KM if pts is not zone_pts else search_km))[0]
        if len(cand) == 0:
            out.append((lon, lat))
            along.append(i * STEP_KM)
            prev = None
            continue
        cost = d_line[cand].copy()
        if prev is not None:
            # süreklilik: önceki seçilen noktaya ~STEP_KM uzaklıkta olanı tercih et
            d_prev = _km(prev[0], prev[1], pts[cand, 0], pts[cand, 1])
            cost = 0.3 * cost + np.abs(d_prev - STEP_KM)
        j = cand[int(cost.argmin())]
        pt = (float(pts[j, 0]), float(pts[j, 1]))
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


def find_junction(a_pts, b_pts, box):
    """İki fay zonunun verilen kutu içinde birbirine en çok yaklaştığı yerin orta noktası."""
    def inbox(P):
        m = (P[:, 0] >= box[0]) & (P[:, 0] <= box[1]) & (P[:, 1] >= box[2]) & (P[:, 1] <= box[3])
        return P[m]
    A, B = inbox(a_pts), inbox(b_pts)
    best = (1e9, None)
    for x, y in A[:: max(1, len(A) // 3000)]:
        d = _km(x, y, B[:, 0], B[:, 1])
        j = int(d.argmin())
        if d[j] < best[0]:
            best = (float(d[j]), ((x + B[j, 0]) / 2, (y + B[j, 1]) / 2))
    return best[1], best[0]


def attach_to_junction(coords, junction, at_start, zone_pts=None):
    """Çizginin bir ucunu kavşağa bağlar: kavşağa en yakın noktadan sonrasını keser, uca kavşağı koyar.

    zone_pts verilirse kavşak ile kesilen nokta arasındaki bağlantı da fay izine oturtulur
    (düz çizgi yerine); izin bulunmadığı kesim düz kalır."""
    pts = coords if at_start else coords[::-1]
    d = _km(junction[0], junction[1], np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    # yalnız uca yakın ilk %30'luk kısımda ara (çizginin ortasını kesme)
    k = int(d[: max(2, len(pts) * 3 // 10)].argmin())
    connector = [list(junction)]
    if zone_pts is not None and d[k] > STEP_KM:
        # bağlantıda yalnız 5 km içindeki izi kabul et (uzak kollara sapmasın)
        snapped, _r = snap_line([list(junction), list(pts[k])], zone_pts, search_km=5.0)
        connector = [list(junction)] + [list(p) for p in snapped[1:-1]]
    out = connector + [list(p) for p in pts[k:]]
    return out if at_start else out[::-1]


def main():
    pb = json.loads(SRC.read_text(encoding="utf-8"))
    faults = json.loads(FAULTS.read_text(encoding="utf-8"))["features"]
    zone_pts = {}
    for code, zone in ZONES.items():
        pts = [p[:2] for f in faults if (f["properties"].get("fay_adi") or "") == zone
               for line in _lines(f["geometry"]) for p in line]
        zone_pts[code] = np.array(pts, dtype=float)
        print(f"{code}: {zone} — {len(pts)} MTA noktası")
    prio = {}
    if RUPTURE_2023.exists():
        rup = json.loads(RUPTURE_2023.read_text(encoding="utf-8"))["features"]
        for code, mw in PRIORITY.items():
            prio[code] = np.array([p[:2] for f in rup if f["properties"].get("deprem_mw") == mw
                                   for line in _lines(f["geometry"]) for p in line], dtype=float)
            print(f"{code}: öncelikli iz 2023 Mw {mw} yüzey kırığı — {len(prio[code])} nokta")
    refined = {}   # code -> (feat, özgün ilk nokta, özgün son nokta)
    for feat in pb["features"]:
        name = feat["properties"].get("Name", "")
        code = name.replace("/", "-")
        if code not in ZONES or len(zone_pts[code]) == 0:
            continue
        orig = _lines(feat["geometry"])
        refined[code] = (feat, orig[0][0][:2], orig[-1][-1][:2])
        new_lines, ratios = [], []
        for line in _lines(feat["geometry"]):
            coords, ratio = snap_line(line, zone_pts[code], prio.get(code))
            new_lines.append([[round(x, 5), round(y, 5)] for x, y in coords])
            ratios.append(ratio)
        n_old = sum(len(x) for x in _lines(feat["geometry"]))
        n_new = sum(len(x) for x in new_lines)
        feat["geometry"] = ({"type": "LineString", "coordinates": new_lines[0]} if len(new_lines) == 1
                            else {"type": "MultiLineString", "coordinates": new_lines})
        feat["properties"]["Source"] = (f'{feat["properties"].get("Source", "")}; Türkiye kesimi '
                                        f'MTA Diri Fay Haritası 2013 ({ZONES[code]}) izine oturtuldu'
                                        + ('; 2023 kırılan kesim USGS yüzey kırığına (doi:10.5066/P985I7U2)'
                                           if code in prio else ''))
        print(f"  {name}: {n_old} → {n_new} nokta, örneklerin %{100 * min(ratios):.0f}-"
              f"{100 * max(ratios):.0f}'i MTA izine oturdu")
    # Karlıova üçlü kavşağı: AT-EU, AT-AR, EU-AR özgün veride aynı noktada buluşuyordu;
    # uçlar ayrı ayrı oturtulunca kopuyordu. Üçü MTA'da KAF ile DAF'ın en çok yaklaştığı
    # noktada (gerçek kavşak) yeniden birleştirilir.
    triple = [c for c in ("AT-EU", "AT-AR", "EU-AR") if c in refined]
    if len(triple) == 3:
        ends = {}
        for c in triple:
            _f, a, b = refined[c]
            ends[c] = (a, b)
        # ortak özgün uç: üç çizgide 1 km içinde tekrar eden uç
        cand = [p for c in triple for p in ends[c]]
        shared = min(cand, key=lambda p: sum(min(float(_km(p[0], p[1], np.array([q[0]]), np.array([q[1]]))[0])
                                                  for q in ends[c]) for c in triple))
        junction, gap = find_junction(zone_pts["AT-EU"], zone_pts["AT-AR"],
                                      (shared[0] - 0.8, shared[0] + 0.8, shared[1] - 0.6, shared[1] + 0.6))
        print(f"Karlıova kavşağı: özgün {shared[1]:.3f}K {shared[0]:.3f}D → MTA kavşağı "
              f"{junction[1]:.4f}K {junction[0]:.4f}D (KAF–DAF arası {gap * 1000:.0f} m)")
        for c in triple:
            feat, a, b = refined[c]
            geom = feat["geometry"]
            lines = _lines(geom)
            da = float(_km(shared[0], shared[1], np.array([a[0]]), np.array([a[1]]))[0])
            db = float(_km(shared[0], shared[1], np.array([b[0]]), np.array([b[1]]))[0])
            if da <= 1.0:
                lines[0] = [[round(x, 5), round(y, 5)] for x, y in attach_to_junction(lines[0], junction, True, zone_pts[c])]
            if db <= 1.0:
                lines[-1] = [[round(x, 5), round(y, 5)] for x, y in attach_to_junction(lines[-1], junction, False, zone_pts[c])]
            feat["geometry"] = ({"type": "LineString", "coordinates": lines[0]} if len(lines) == 1
                                else {"type": "MultiLineString", "coordinates": lines})
    OUT.write_text(json.dumps(pb, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Yazıldı: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
