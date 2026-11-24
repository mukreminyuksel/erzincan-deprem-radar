"""EMSC dünya kataloğunu (M2.0+, 2004'ten bugüne) yıl yıl indirir.

"Benzer Bölgeler → Zaman örtüşmesi" paneli küçük büyüklükler (M2–M4.5) için bunu
kullanır. USGS dünya kataloğu M4.5 altını kapsamadığı için ayrı tutulur.

UYARI (veri niteliği): EMSC'nin küçük deprem kapsamı bölgeden bölgeye ve yıldan yıla
değişir (Avrupa-Akdeniz ve Türkiye iyi; Japonya gibi bazı bölgeler M4 altında çok
eksik; kayıt sayısı 2015'ten 2024'e ~3 kat artmış). Bölgeler arası karşılaştırmada
bu fark yapay benzerlik/farklılık üretebilir.

Çıktı: data/emsc_m20/YYYY.csv.gz (sütunlar: t, lat, lon, depth, mag, place).
Yıl dosyaları ayrı tutulur: geçmiş yıllar değişmez, haftalık güncelleme yalnız son
iki yılı yeniden yazar (depo geçmişi şişmez).

Çalıştırma:  python tools/build_emsc_catalog.py            # eksik yıllar + son iki yıl
             python tools/build_emsc_catalog.py --all      # hepsini baştan
"""
from __future__ import annotations

import csv
import gzip
import io
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

URL = "https://www.seismicportal.eu/fdsnws/event/1/query"
START_YEAR = 2004
MIN_MAG = 2.0
LIMIT = 20000
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "emsc_m20"


def fetch(start: datetime, end: datetime) -> list[tuple]:
    """[start, end) aralığı; sınır dolarsa aralığı ikiye böler."""
    params = {"starttime": start.strftime("%Y-%m-%dT%H:%M:%S"), "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
              "minmag": MIN_MAG, "format": "text", "limit": LIMIT, "orderby": "time-asc"}
    for attempt in range(5):
        try:
            r = requests.get(URL, params=params, timeout=180,
                             headers={"User-Agent": "ErzincanDepremRadari-catalog/1.0"})
            if r.status_code == 204:
                return []
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 4:
                raise
            print(f"  {start:%Y-%m-%d}: {exc} — yeniden", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    r.encoding = "utf-8"
    lines = [ln for ln in r.text.split("\n") if ln and not ln.startswith("#")]
    if len(lines) >= LIMIT and (end - start) > timedelta(hours=1):
        mid = start + (end - start) / 2
        return fetch(start, mid) + fetch(mid, end)
    rows = []
    for ln in lines:
        p = ln.split("|")
        try:
            t = datetime.fromisoformat(p[1].replace("Z", "+00:00")).timestamp()
            depth = abs(float(p[4])) if p[4] else float("nan")
            rows.append((int(t), round(float(p[2]), 3), round(float(p[3]), 3), round(depth, 1),
                         round(float(p[10]), 1), p[12].strip().title() if len(p) > 12 else ""))
        except (IndexError, ValueError):
            continue
    return rows


def build_year(year: int, now: datetime) -> int:
    rows, month = [], datetime(year, 1, 1, tzinfo=timezone.utc)
    stop = min(datetime(year + 1, 1, 1, tzinfo=timezone.utc), now)
    while month < stop:
        nxt = min(datetime(month.year + (month.month == 12), month.month % 12 + 1, 1, tzinfo=timezone.utc), stop)
        rows += fetch(month, nxt)
        month = nxt
        time.sleep(0.5)
    rows.sort()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{year}.csv.gz"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        w = csv.writer(text)
        w.writerow(["t", "lat", "lon", "depth", "mag", "place"])
        w.writerows(rows)
        text.flush()
        text.detach()
    tmp.replace(out)
    print(f"{year}: {len(rows)} deprem ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return len(rows)


def main() -> int:
    now = datetime.now(timezone.utc)
    rebuild_all = "--all" in sys.argv
    total = 0
    for year in range(START_YEAR, now.year + 1):
        exists = (OUT_DIR / f"{year}.csv.gz").exists()
        if exists and not rebuild_all and year < now.year - 1:
            continue
        total += build_year(year, now)
    print(f"Tamam: {total} deprem yazıldı → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
