"""Dünya deprem kataloğunu (USGS, M4.5+, 1990'dan bugüne) indirip sıkıştırılmış CSV yazar.

"Benzer Bölgeler" paneli bu dosyayı kullanır; uygulama her açılışta yüz binlerce
kaydı indirmek zorunda kalmaz. GitHub Actions haftada bir çalıştırır
(.github/workflows/refresh-catalog.yml). Elle çalıştırma:

    python tools/build_global_catalog.py

USGS tek sorguda en fazla 20.000 kayıt verdiği için yıl yıl indirilir.
Çıktı: data/global_catalog_m45.csv.gz  (sütunlar: t, lat, lon, depth, mag, place;
t = Unix zamanı, saniye; place = "Kokopo, Papua New Guinea" gibi bölge adı,
baştaki "45 km SSW of" kısmı atılır).
"""
from __future__ import annotations

import csv
import gzip
import io
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
START_YEAR = 1990
MIN_MAG = 4.5
_PLACE_PREFIX = re.compile(r"^\s*\d+(\.\d+)?\s*km\s+[NSEW]{1,3}\s+of\s+", re.I)


def short_place(place: str) -> str:
    """'45 km SSW of Kokopo, Papua New Guinea' -> 'Kokopo, Papua New Guinea'."""
    return _PLACE_PREFIX.sub("", place or "").strip()


OUT = Path(__file__).resolve().parent.parent / "data" / "global_catalog_m45.csv.gz"


def fetch_year(year: int, end: datetime) -> list[tuple]:
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    stop = min(datetime(year + 1, 1, 1, tzinfo=timezone.utc), end)
    params = {
        "format": "csv", "orderby": "time-asc", "minmagnitude": MIN_MAG,
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": stop.strftime("%Y-%m-%dT%H:%M:%S"),
        "eventtype": "earthquake", "limit": 20000,
    }
    for attempt in range(4):
        try:
            r = requests.get(USGS_URL, params=params, timeout=120,
                             headers={"User-Agent": "ErzincanDepremRadari-catalog/1.0"})
            r.raise_for_status()
            r.encoding = "utf-8"  # USGS CSV UTF-8; başlıkta charset yok, latin-1 sanılıyordu
            break
        except requests.RequestException as exc:
            if attempt == 3:
                raise
            print(f"  {year}: {exc} — yeniden deneniyor", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    rows = []
    for rec in csv.DictReader(io.StringIO(r.text)):
        try:
            t = datetime.fromisoformat(rec["time"].replace("Z", "+00:00")).timestamp()
            depth = float(rec["depth"]) if rec["depth"] else float("nan")
            rows.append((int(t), round(float(rec["latitude"]), 3), round(float(rec["longitude"]), 3),
                         round(depth, 1), round(float(rec["mag"]), 1),
                         short_place(rec.get("place", ""))))
        except (KeyError, ValueError):
            continue
    if len(rows) >= 20000:
        raise RuntimeError(f"{year}: 20.000 sınırına ulaşıldı, yıl bölünmeli")
    return rows


def main() -> int:
    end = datetime.now(timezone.utc)
    all_rows: list[tuple] = []
    for year in range(START_YEAR, end.year + 1):
        rows = fetch_year(year, end)
        print(f"{year}: {len(rows)} deprem")
        all_rows.extend(rows)
        time.sleep(1)  # USGS'e nazik davran
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    # mtime=0: içerik aynıysa dosya da bayt bayt aynı olur (gereksiz commit olmaz)
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        w = csv.writer(text)
        w.writerow(["t", "lat", "lon", "depth", "mag", "place"])
        w.writerows(all_rows)
        text.flush()
        text.detach()
    tmp.replace(OUT)
    print(f"Toplam {len(all_rows)} deprem → {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
