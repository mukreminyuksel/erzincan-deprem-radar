"""Benzer deprem bölgeleri: profil benzerliği, eşzamanlılık ve bağlı bölge durumu.

Streamlit'ten bağımsız saf hesap modülü (tests/test_similarity.py ile sınanır).

ÖNEMLİ (bilimsel sınır): İki bölgenin geçmiş depremlerinin benzer olması ya da
aynı günlerde deprem görmüş olması, birindeki depremin ötekini etkilediğini
göstermez. Bu modül karşılaştırma ve izleme içindir; tahmin üretmez.

Veri: USGS dünya kataloğu, M4.5+ (tools/build_global_catalog.py). M4.5 dünya
genelinde katalog tamlığının makul olduğu eşik olarak seçildi.
"""
from __future__ import annotations

import csv
import gzip
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

EARTH_R = 6371.0
MC = 4.5                                   # katalog alt sınırı (tamlık eşiği)
MAG_BINS = [4.5, 5.0, 5.5, 6.0, 6.5, 99.0]
DEPTH_BINS = [0.0, 15.0, 35.0, 70.0, 150.0, 300.0, 1e4]

# Eşzamanlılık eşleşme toleransları
SYNC_WINDOW_S = 86400                      # ±1 gün
SYNC_DMAG = 0.3                            # büyüklük farkı
SYNC_DDEPTH_KM = 15.0                      # derinlik farkı


# ─── Katalog ─────────────────────────────────────────────────────────────────
@dataclass
class Catalog:
    t: np.ndarray        # Unix saniye (int64)
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray    # km, bilinmiyorsa NaN
    mag: np.ndarray
    place: list          # bölge adı (ör. "Kokopo, Papua New Guinea")

    def __len__(self):
        return len(self.t)

    @property
    def span_years(self) -> float:
        if len(self.t) < 2:
            return 1.0
        return max((self.t.max() - self.t.min()) / (365.25 * 86400), 1.0)

    def subset(self, idx) -> "Catalog":
        return Catalog(self.t[idx], self.lat[idx], self.lon[idx], self.depth[idx],
                       self.mag[idx], [self.place[i] for i in np.atleast_1d(idx)])


def load_catalog(path) -> Catalog:
    t, lat, lon, depth, mag, place = [], [], [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t.append(int(row["t"]))
            lat.append(float(row["lat"]))
            lon.append(float(row["lon"]))
            depth.append(float(row["depth"]) if row["depth"] not in ("", "nan") else np.nan)
            mag.append(float(row["mag"]))
            place.append(row.get("place", ""))
    order = np.argsort(np.asarray(t, dtype=np.int64), kind="stable")
    return Catalog(np.asarray(t, dtype=np.int64)[order], np.asarray(lat)[order],
                   np.asarray(lon)[order], np.asarray(depth, dtype=float)[order],
                   np.asarray(mag)[order], [place[i] for i in order])


def haversine_km(lat0, lon0, lats, lons):
    """Bir noktadan dizi noktalara büyük çember uzaklığı (km)."""
    p0, l0 = math.radians(lat0), math.radians(lon0)
    p, lam = np.radians(lats), np.radians(lons)
    a = np.sin((p - p0) / 2) ** 2 + math.cos(p0) * np.cos(p) * np.sin((lam - l0) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def region_index(cat: Catalog, lat, lon, radius_km) -> np.ndarray:
    return np.nonzero(haversine_km(lat, lon, cat.lat, cat.lon) <= radius_km)[0]


def region_name(cat: Catalog, idx) -> str:
    """Bölgedeki depremlerin en sık geçen yer adı."""
    names = [cat.place[i] for i in idx if cat.place[i]]
    if not names:
        return ""
    return Counter(names).most_common(1)[0][0]


# ─── Profil ──────────────────────────────────────────────────────────────────
def b_value(mags: np.ndarray, mc: float = MC, dm: float = 0.1) -> float:
    """Aki (1965) en çok olabilirlik b-değeri, Utsu bin düzeltmesiyle. n<10 ise NaN."""
    m = mags[mags >= mc]
    if len(m) < 10:
        return float("nan")
    denom = m.mean() - (mc - dm / 2)
    return float(math.log10(math.e) / denom) if denom > 0 else float("nan")


def _hist(values, bins):
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return np.zeros(len(bins) - 1)
    h, _ = np.histogram(v, bins=bins)
    return h / h.sum()


@dataclass
class Profile:
    n: int
    rate_per_year: float
    mag_hist: np.ndarray
    depth_hist: np.ndarray
    b: float
    max_mag: float
    median_depth: float


def profile(cat: Catalog, idx, span_years: float) -> Profile:
    mags, depths = cat.mag[idx], cat.depth[idx]
    return Profile(
        n=int(len(idx)),
        rate_per_year=len(idx) / span_years,
        mag_hist=_hist(mags, MAG_BINS),
        depth_hist=_hist(depths, DEPTH_BINS),
        b=b_value(mags),
        max_mag=float(mags.max()) if len(mags) else float("nan"),
        median_depth=float(np.nanmedian(depths)) if np.any(~np.isnan(depths)) else float("nan"),
    )


def profile_similarity(a: Profile, b: Profile) -> float:
    """0–100 benzerlik puanı.

    Ağırlıklar: büyüklük dağılımı %35, derinlik dağılımı %35, yıllık sıklık %20
    (log ölçekte, 2 kat = 0.3 fark), b-değeri %10. b-değeri birinde
    hesaplanamıyorsa ağırlığı diğerlerine dağıtılır.
    """
    parts = [
        (0.35, 0.5 * np.abs(a.mag_hist - b.mag_hist).sum()),           # 0..1
        (0.35, 0.5 * np.abs(a.depth_hist - b.depth_hist).sum()),       # 0..1
        (0.20, min(abs(math.log10(a.rate_per_year) - math.log10(b.rate_per_year)) / 2.0, 1.0)),
    ]
    if not (math.isnan(a.b) or math.isnan(b.b)):
        parts.append((0.10, min(abs(a.b - b.b), 1.0)))
    wsum = sum(w for w, _ in parts)
    dist = sum(w * d for w, d in parts) / wsum
    return round(100.0 * (1.0 - dist), 1)


# ─── Eşzamanlılık ────────────────────────────────────────────────────────────
def _poisson_sf(k: int, mu: float) -> float:
    """P(X >= k), X ~ Poisson(mu)."""
    if k <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    term, cdf = math.exp(-mu), 0.0
    for i in range(k):
        cdf += term
        term *= mu / (i + 1)
    return max(0.0, 1.0 - cdf)


@dataclass
class Synchrony:
    days: int             # eşleşmeli farklı gün sayısı (referans takvimine göre)
    expected: float       # takvim kaydırma ile tesadüfen beklenen gün sayısı
    p_value: float        # kaydırmaların kaçında gözlenen kadar ya da fazla gün çıktı
    examples: list        # [(t_ref, t_cand, m_ref, m_cand), ...] en fazla 5
    tested: bool = False  # kaydırma testi yapıldı mı (yalnız en umut verici adaylarda)


def _match_matrix(cat, ref_idx, cand_idx):
    """Zamandan bağımsız büyüklük/derinlik eşleşmesi (NaN derinlik eşleşmez)."""
    rm, rd = cat.mag[ref_idx], cat.depth[ref_idx]
    cm, cd = cat.mag[cand_idx], cat.depth[cand_idx]
    return (np.abs(rm[:, None] - cm[None, :]) <= SYNC_DMAG + 1e-9) & \
           (np.abs(rd[:, None] - cd[None, :]) <= SYNC_DDEPTH_KM)


def _matched_days(rt, ct, match, collect_examples=False, rm=None, cm=None):
    """Referans olaylarından, ±pencere içinde eşleşen aday olayı olanların farklı gün sayısı.

    ct zamana göre sıralı olmalı; match[i, j] ct sırasına göre.
    """
    lo = np.searchsorted(ct, rt - SYNC_WINDOW_S, side="left")
    hi = np.searchsorted(ct, rt + SYNC_WINDOW_S, side="right")
    days, examples = set(), []
    for i in np.nonzero(hi > lo)[0]:
        js = np.nonzero(match[i, lo[i]:hi[i]])[0]
        if len(js):
            days.add(int(rt[i] // 86400))
            if collect_examples and len(examples) < 5:
                j = js[0] + lo[i]
                examples.append((int(rt[i]), int(ct[j]), float(rm[i]), float(cm[j])))
    return len(days), examples


def synchrony(cat: Catalog, ref_idx, cand_idx, t_min: float, t_max: float,
              n_shifts: int = 0, seed: int = 0) -> Synchrony:
    """Aynı günlerde (±1 gün), benzer büyüklük (±0.3) ve derinlikte (±15 km) deprem.

    Sayılan şey farklı gün sayısıdır: bir artçı fırtınası tek gün sayılır.
    n_shifts > 0 ise tesadüf beklentisi takvim kaydırma testiyle ölçülür: aday
    bölgenin bütün olay zamanları aynı rastgele miktarda (dairesel) kaydırılır,
    gün sayısı yeniden hesaplanır. Bu, depremlerin kümelenmesini (artçı dizileri)
    koruduğu için bağımsızlık varsayan formüllerden çok daha dürüst bir karşılaştırmadır.
    """
    if len(ref_idx) == 0 or len(cand_idx) == 0:
        return Synchrony(0, 0.0, 1.0, [])
    rt, ct = cat.t[ref_idx].astype(float), cat.t[cand_idx].astype(float)
    match = _match_matrix(cat, ref_idx, cand_idx)
    days, examples = _matched_days(rt, ct, match, True, cat.mag[ref_idx], cat.mag[cand_idx])
    if n_shifts <= 0:
        return Synchrony(days, float("nan"), float("nan"), examples)
    span = max(t_max - t_min, 1.0)
    rng = np.random.default_rng(seed)
    null = np.empty(n_shifts)
    for k in range(n_shifts):
        shift = rng.uniform(30 * 86400, span - 30 * 86400) if span > 120 * 86400 else rng.uniform(0, span)
        shifted = t_min + (ct - t_min + shift) % span
        order = np.argsort(shifted, kind="stable")
        null[k], _ = _matched_days(rt, shifted[order], match[:, order])
    p = (1 + int((null >= days).sum())) / (1 + n_shifts)
    return Synchrony(days, float(null.mean()), p, examples, tested=True)


# ─── Arama ───────────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    lat: float
    lon: float
    name: str
    profile: Profile
    score: float
    sync: Synchrony
    distance_km: float     # referans merkezine uzaklık


def _cell_index(cat: Catalog):
    cells = defaultdict(list)
    for i, (la, lo) in enumerate(zip(np.floor(cat.lat).astype(int), np.floor(cat.lon).astype(int))):
        cells[(la, lo)].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}


def find_similar(cat: Catalog, ref_lat, ref_lon, radius_km, *, min_events=10,
                 top_n=15, cells=None, n_shifts=100, n_tested=40):
    """Referans bölgeye (merkez + yarıçap) benzeyen bölgeleri dünyada arar.

    Aday merkezler: deprem içeren 1°×1° hücrelerin ortaları. Her aday, referansla
    aynı yarıçaplı bir daire. Referansa yakın adaylar (2×yarıçap, en az 300 km)
    ve birbirine yakın adaylar elenir (en iyisi kalır).
    Dönüş: (referans profili, referans olay sayısı, profil listesi, eşzamanlılık
    listesi, taranan aday sayısı).
    """
    span = cat.span_years
    t_min, t_max = float(cat.t.min()), float(cat.t.max())
    ref_idx = region_index(cat, ref_lat, ref_lon, radius_km)
    ref_prof = profile(cat, ref_idx, span)
    if len(ref_idx) < min_events:
        return ref_prof, len(ref_idx), [], [], 0

    cells = cells if cells is not None else _cell_index(cat)
    excl_km = max(2 * radius_km, 300.0)
    sep_km = max(2 * radius_km, 300.0)
    dlat_cells = int(math.ceil(radius_km / 111.0)) + 1
    results = []
    for (cla, clo) in cells:
        c_lat, c_lon = cla + 0.5, clo + 0.5
        d_ref = float(haversine_km(ref_lat, ref_lon, np.array([c_lat]), np.array([c_lon]))[0])
        if d_ref < excl_km:
            continue
        coslat = max(math.cos(math.radians(c_lat)), 0.05)
        dlon_cells = min(int(math.ceil(radius_km / (111.0 * coslat))) + 1, 180)
        near = [cells[(a, ((b + 180) % 360) - 180)]
                for a in range(cla - dlat_cells, cla + dlat_cells + 1)
                for b in range(clo - dlon_cells, clo + dlon_cells + 1)
                if (a, ((b + 180) % 360) - 180) in cells]
        if not near:
            continue
        pool = np.concatenate(near)
        if len(pool) < min_events:
            continue
        idx = pool[haversine_km(c_lat, c_lon, cat.lat[pool], cat.lon[pool]) <= radius_km]
        if len(idx) < min_events:
            continue
        idx.sort()                                   # zamana göre sıralı (katalog sıralı)
        prof = profile(cat, idx, span)
        results.append((c_lat, c_lon, idx, prof, profile_similarity(ref_prof, prof), d_ref))
    scanned = len(results)

    def _pick(sorted_items):
        chosen = []
        for item in sorted_items:
            if all(float(haversine_km(item[0], item[1], np.array([c[0]]), np.array([c[1]]))[0]) >= sep_km
                   for c in chosen):
                chosen.append(item)
            if len(chosen) >= top_n:
                break
        return chosen

    def _to_candidate(item, sync):
        c_lat, c_lon, idx, prof, score, d_ref = item
        return Candidate(c_lat, c_lon, region_name(cat, idx), prof, score, sync, d_ref)

    by_profile = [_to_candidate(x, synchrony(cat, ref_idx, x[2], t_min, t_max))
                  for x in _pick(sorted(results, key=lambda r: -r[4]))]

    # Eşzamanlılık: önce tüm adaylarda gözlenen gün sayısı (ucuz), sonra en çok
    # günü olan n_tested adayda takvim kaydırma testi (pahalı). En az 2 gün.
    quick = []
    for item in results:
        s = synchrony(cat, ref_idx, item[2], t_min, t_max)
        if s.days >= 2:
            quick.append((s.days, item))
    quick.sort(key=lambda x: -x[0])
    tested = []
    for _days, item in quick[:n_tested]:
        s = synchrony(cat, ref_idx, item[2], t_min, t_max, n_shifts=n_shifts,
                      seed=int(abs(item[0] * 1000 + item[1])))
        tested.append((s.p_value, -s.days / max(s.expected, 0.5), item, s))
    tested.sort(key=lambda x: (x[0], x[1]))
    sync_of = {id(x[2]): x[3] for x in tested}
    by_sync = [_to_candidate(c, sync_of[id(c)]) for c in _pick([x[2] for x in tested])]
    return ref_prof, len(ref_idx), by_profile, by_sync, scanned


# ─── Bağlı bölge durumu ──────────────────────────────────────────────────────
@dataclass
class RegionStatus:
    n_recent: int              # son `days` gündeki M4.5+ sayısı
    expected_recent: float     # uzun dönem ortalamasına göre beklenen
    max_recent: float          # son dönemdeki en büyük büyüklük (yoksa NaN)
    last_event_t: float        # son olayın Unix zamanı (yoksa NaN)
    ratio: float               # n_recent / expected (beklenen 0 ise NaN)


def region_status(recent_t, recent_lat, recent_lon, recent_mag, lat, lon, radius_km,
                  long_term_rate_per_year, days=30) -> RegionStatus:
    """Son olaylar (ör. USGS 30 günlük M4.5+ akışı) ile uzun dönem ortalamasını karşılaştırır."""
    recent_t = np.asarray(recent_t, dtype=float)
    mask = haversine_km(lat, lon, np.asarray(recent_lat), np.asarray(recent_lon)) <= radius_km
    mags = np.asarray(recent_mag)[mask]
    times = recent_t[mask]
    expected = long_term_rate_per_year * days / 365.25
    n = int(mask.sum())
    return RegionStatus(
        n_recent=n,
        expected_recent=expected,
        max_recent=float(mags.max()) if n else float("nan"),
        last_event_t=float(times.max()) if n else float("nan"),
        ratio=(n / expected) if expected > 0 else float("nan"),
    )
