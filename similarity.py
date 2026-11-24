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
from array import array
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
    """Sıkıştırılmış CSV kataloğu okur. Bellek için tipli diziler ve tekilleştirilmiş yer adları
    kullanılır (1 milyon satırda Python listeleri yüzlerce MB tutuyordu)."""
    t, lat, lon, depth, mag = array("q"), array("d"), array("d"), array("d"), array("d")
    place, seen = [], {}
    nan = float("nan")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        head = next(reader)
        it, ila, ilo, idp, img = (head.index(k) for k in ("t", "lat", "lon", "depth", "mag"))
        ipl = head.index("place") if "place" in head else None
        for row in reader:
            t.append(int(row[it]))
            lat.append(float(row[ila]))
            lon.append(float(row[ilo]))
            d = row[idp]
            depth.append(float(d) if d not in ("", "nan") else nan)
            mag.append(float(row[img]))
            name = row[ipl] if ipl is not None and len(row) > ipl else ""
            place.append(seen.setdefault(name, name))
    tt = np.frombuffer(t, dtype=np.int64) if len(t) else np.array([], dtype=np.int64)
    order = np.argsort(tt, kind="stable")

    def arr(a):
        return (np.frombuffer(a, dtype=float) if len(a) else np.array([], dtype=float))[order]

    return Catalog(tt[order], arr(lat), arr(lon), arr(depth), arr(mag), [place[i] for i in order])


def load_catalogs(paths) -> Catalog:
    """Birden çok katalog dosyasını (ör. yıl yıl) tek katalogda birleştirir, zamana göre sıralar."""
    parts = [load_catalog(p) for p in sorted(paths)]
    t = np.concatenate([c.t for c in parts])
    order = np.argsort(t, kind="stable")
    place = [x for c in parts for x in c.place]
    return Catalog(t[order], np.concatenate([c.lat for c in parts])[order],
                   np.concatenate([c.lon for c in parts])[order],
                   np.concatenate([c.depth for c in parts])[order],
                   np.concatenate([c.mag for c in parts])[order], [place[i] for i in order])


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


# ─── İki bölgeyi doğrudan karşılaştırma ──────────────────────────────────────
@dataclass
class Comparison:
    n_a: int
    n_b: int
    profile_a: Profile
    profile_b: Profile
    score: float           # profil benzerliği 0–100 (bir tarafta <10 olay varsa NaN)
    sync: Synchrony        # takvim kaydırma testiyle
    distance_km: float
    years: np.ndarray      # yıl ekseni
    yearly_a: np.ndarray   # yıllık deprem sayıları
    yearly_b: np.ndarray


def compare_regions(cat: Catalog, lat_a, lon_a, lat_b, lon_b, radius_km, *,
                    min_events=10, n_shifts=300) -> Comparison:
    """Seçilen iki bölgeyi (aynı yarıçap) karşılaştırır: profil + eşzamanlılık + yıllık seri."""
    span = cat.span_years
    t_min, t_max = float(cat.t.min()), float(cat.t.max())
    ia = region_index(cat, lat_a, lon_a, radius_km)
    ib = region_index(cat, lat_b, lon_b, radius_km)
    pa, pb = profile(cat, ia, span), profile(cat, ib, span)
    enough = len(ia) >= min_events and len(ib) >= min_events
    score = profile_similarity(pa, pb) if enough else float("nan")
    sync = synchrony(cat, ia, ib, t_min, t_max, n_shifts=n_shifts if enough else 0)
    y0 = int(np.datetime64(int(t_min), "s").astype("datetime64[Y]").astype(int)) + 1970
    y1 = int(np.datetime64(int(t_max), "s").astype("datetime64[Y]").astype(int)) + 1970
    years = np.arange(y0, y1 + 1)

    def yearly(idx):
        ys = cat.t[idx].astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
        return np.array([(ys == y).sum() for y in years])

    dist = float(haversine_km(lat_a, lon_a, np.array([lat_b]), np.array([lon_b]))[0])
    return Comparison(len(ia), len(ib), pa, pb, score, sync, dist, years, yearly(ia), yearly(ib))


# ─── Zaman örtüşmesi (etkinlik serisi korelasyonu) ───────────────────────────
@dataclass
class TimeMatch:
    lat: float
    lon: float
    name: str
    n: int                 # dönemdeki deprem sayısı
    r: float               # aynı dönemler arası korelasyon (-1..1)
    p_value: float         # zaman kaydırma testine göre tesadüf olasılığı
    series: np.ndarray     # dönem başına deprem sayısı
    distance_km: float
    r_first: float = float("nan")        # dönemin ilk yarısında korelasyon
    r_second: float = float("nan")       # ikinci yarısında (ikisi de yüksekse ilişki tutarlı)
    best_lag: int = 0                    # en iyi örtüşmenin kayması (dönem; + = bu bölge önce)
    r_best_lag: float = float("nan")
    depth_overlap: float = float("nan")  # derinlik dağılımı örtüşmesi, %
    mag_overlap: float = float("nan")    # büyüklük dağılımı örtüşmesi, %
    depth_hist: np.ndarray = None        # DEPTH_FINE aralıklarında pay (toplam 1)
    median_depth: float = float("nan")
    ev_t: np.ndarray = None              # tek tek depremler (nokta grafikleri için)
    ev_mag: np.ndarray = None
    ev_depth: np.ndarray = None


@dataclass
class TimeSearch:
    edges: np.ndarray
    ref_series: np.ndarray
    n_ref: int
    matches: list
    scanned: int
    chance: tuple            # (tesadüfi en yüksek r medyanı, %95'i)
    ref_depth_hist: np.ndarray = None
    ref_median_depth: float = float("nan")
    ref_ev_t: np.ndarray = None
    ref_ev_mag: np.ndarray = None
    ref_ev_depth: np.ndarray = None


DEPTH_FINE = [0, 5, 10, 15, 20, 30, 40, 60, 100, 200, 700]


def overlap_pct(h1, h2):
    """İki dağılımın (toplamı 1) ortak alanı, yüzde. 100 = aynı dağılım."""
    return float(100.0 * (1.0 - 0.5 * np.abs(np.asarray(h1) - np.asarray(h2)).sum()))


def _half_r(a, b):
    xa, xb = _norm_series(a), _norm_series(b)
    return float("nan") if xa is None or xb is None or len(a) < 6 else float(xa @ xb)


def _best_lag(a_counts, b_counts, max_lag=3):
    """±max_lag dönem içinde en yüksek korelasyonu veren kayma. + ise b serisi önce davranıyor."""
    a, b = _norm_series(a_counts), _norm_series(b_counts)
    if a is None or b is None:
        return 0, float("nan")
    best = (0, float(a @ b))
    for k in range(1, min(max_lag, len(a) - 6) + 1):
        for lag, r in ((k, float(a[k:] @ b[:-k])), (-k, float(a[:-k] @ b[k:]))):
            if r > best[1] + 1e-9:
                best = (lag, r)
    return best


def _norm_series(counts):
    """log1p + ortalama çıkar + birim norm. Tek bir artçı fırtınası korelasyonu ele geçirmesin diye log."""
    x = np.log1p(np.asarray(counts, dtype=float))
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / n if n > 0 else None


def series_correlation(a_counts, b_counts):
    """İki sayım serisinin korelasyonu ve kaydırma testi p-değeri. Dönüş: (r, p).

    r: log1p sayımların Pearson korelasyonu (aynı dönemler).
    p: b serisi zamanda dairesel kaydırılınca (en az 2 dönem) r kadar ya da daha yüksek
       korelasyon veren kaydırmaların oranı. Serinin kendi iç yapısı (kümelenme)
       korunduğu için dürüst bir tesadüf ölçüsüdür. Tüm kaydırmalar FFT ile tek seferde.
    """
    a, b = _norm_series(a_counts), _norm_series(b_counts)
    if a is None or b is None or len(a) < 8:
        return float("nan"), float("nan")
    n = len(a)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n)   # cc[k] = a ile k kaydırılmış b
    r = float(cc[0])
    guard = max(2, n // 20)
    null = cc[guard: n - guard + 1]
    if len(null) == 0:
        return r, float("nan")
    return r, float((1 + (null >= r - 1e-12).sum()) / (1 + len(null)))


def find_time_correlated(cat: Catalog, ref_lat, ref_lon, radius_km, *, t0, t1, min_mag=MC,
                         bin_days=30, min_events=10, min_active_bins=5, top_n=15):
    """Etkinlik zaman serisi referansa en çok benzeyen bölgeleri dünyada arar.

    Her bölge için [t0, t1) aralığında bin_days'lik dönemlerde deprem sayısı çıkarılır,
    referans seriyle korelasyonu hesaplanır. Dönüş: (kenarlar, referans serisi,
    referans olay sayısı, eşleşmeler [r'ye göre azalan], taranan aday sayısı,
    (tesadüfi en yüksek r'nin medyanı, %95'i)).

    Son değer: referans seri zamanda rastgele kaydırılıp tüm adaylarla yeniden
    karşılaştırıldığında çıkan en yüksek korelasyon. Binlerce aday tarandığı için en
    iyi sonuç tesadüfen de yüksek çıkar; gerçek sonuç ancak bu eşiği aşıyorsa dikkat çekicidir.
    """
    step = bin_days * 86400.0
    edges = np.arange(float(t0), float(t1) + step, step)
    sel = np.nonzero((cat.t >= edges[0]) & (cat.t < edges[-1]) & (cat.mag >= min_mag))[0]
    sub = cat.subset(sel)
    ref_idx = region_index(sub, ref_lat, ref_lon, radius_km)
    ref_series, _ = np.histogram(sub.t[ref_idx], edges)
    if len(ref_idx) < min_events or (ref_series > 0).sum() < min_active_bins:
        return TimeSearch(edges, ref_series, len(ref_idx), [], 0, (float("nan"), float("nan")))
    ref_dh = _hist(sub.depth[ref_idx], DEPTH_FINE)
    mag_bins = list(np.arange(min_mag, 8.01, 0.5)) + [10.0]
    ref_mh = _hist(sub.mag[ref_idx], mag_bins)
    half = len(ref_series) // 2
    cells = _cell_index(sub)
    excl_km = sep_km = max(2 * radius_km, 300.0)
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
        pool = np.concatenate(near)
        if len(pool) < min_events:
            continue
        idx = pool[haversine_km(c_lat, c_lon, sub.lat[pool], sub.lon[pool]) <= radius_km]
        if len(idx) < min_events:
            continue
        series, _ = np.histogram(sub.t[idx], edges)
        if (series > 0).sum() < min_active_bins:
            continue
        r, p = series_correlation(ref_series, series)
        if not math.isnan(r):
            results.append((r, p, c_lat, c_lon, idx, series, d_ref))
    chance = (float("nan"), float("nan"))
    ref_n = _norm_series(ref_series)
    if results and ref_n is not None:
        M = np.array([_norm_series(x[5]) for x in results])
        rng = np.random.default_rng(0)
        guard = max(2, len(ref_n) // 20)
        shifts = rng.integers(guard, len(ref_n) - guard + 1, 40)
        maxima = [float((M @ np.roll(ref_n, int(k))).max()) for k in shifts]
        chance = (float(np.median(maxima)), float(np.percentile(maxima, 95)))
    results.sort(key=lambda x: -x[0])
    chosen = []
    for r, p, c_lat, c_lon, idx, series, d_ref in results:
        if all(float(haversine_km(c_lat, c_lon, np.array([m.lat]), np.array([m.lon]))[0]) >= sep_km
               for m in chosen):
            dh = _hist(sub.depth[idx], DEPTH_FINE)
            lag, r_lag = _best_lag(ref_series, series)
            depths = sub.depth[idx]
            chosen.append(TimeMatch(
                c_lat, c_lon, region_name(sub, idx), len(idx), r, p, series, d_ref,
                r_first=_half_r(ref_series[:half], series[:half]),
                r_second=_half_r(ref_series[half:], series[half:]),
                best_lag=lag, r_best_lag=r_lag,
                depth_overlap=overlap_pct(ref_dh, dh) if dh.sum() and ref_dh.sum() else float("nan"),
                mag_overlap=overlap_pct(ref_mh, _hist(sub.mag[idx], mag_bins)),
                depth_hist=dh,
                median_depth=float(np.nanmedian(depths)) if np.any(~np.isnan(depths)) else float("nan"),
                ev_t=sub.t[idx], ev_mag=sub.mag[idx], ev_depth=depths))
        if len(chosen) >= top_n:
            break
    ref_d = sub.depth[ref_idx]
    return TimeSearch(edges, ref_series, len(ref_idx), chosen, len(results), chance, ref_dh,
                      float(np.nanmedian(ref_d)) if np.any(~np.isnan(ref_d)) else float("nan"),
                      sub.t[ref_idx], sub.mag[ref_idx], ref_d)


def pair_series(cat: Catalog, lat_a, lon_a, lat_b, lon_b, radius_km, *, t0, t1, min_mag=MC, bin_days=30):
    """İki bölgenin aynı kenarlarla etkinlik serileri: (kenarlar, a, b)."""
    step = bin_days * 86400.0
    edges = np.arange(float(t0), float(t1) + step, step)
    sel = np.nonzero((cat.t >= edges[0]) & (cat.t < edges[-1]) & (cat.mag >= min_mag))[0]
    t, la, lo = cat.t[sel], cat.lat[sel], cat.lon[sel]
    a, _ = np.histogram(t[haversine_km(lat_a, lon_a, la, lo) <= radius_km], edges)
    b, _ = np.histogram(t[haversine_km(lat_b, lon_b, la, lo) <= radius_km], edges)
    return edges, a, b


def correlation_before_after(edges, a, b, split_t, min_bins=6):
    """Ataçlama anından önceki ve sonraki korelasyon. Yeterli dönem yoksa NaN.

    Dönüş: (r_önce, dönem_önce, r_sonra, dönem_sonra)
    """
    k = int(np.searchsorted(edges[:-1], split_t, side="left"))

    def _r(x, y):
        if len(x) < min_bins:
            return float("nan")
        xn, yn = _norm_series(x), _norm_series(y)
        return float("nan") if xn is None or yn is None else float(xn @ yn)

    return _r(a[:k], b[:k]), k, _r(a[k:], b[k:]), len(a) - k


# ─── Deprem deprem eşleştirme ────────────────────────────────────────────────
@dataclass
class EventRegion:
    lat: float
    lon: float
    name: str
    n_events: int            # bölgedeki deprem sayısı (aynı aralık, aynı büyüklük eşiği)
    matched: int             # eşi bulunan referans depremi sayısı
    coverage: float          # matched / referans deprem sayısı, %
    expected: float          # tesadüfen beklenen eşleşme sayısı (takvim kaydırma)
    p_value: float
    z: float                 # (gözlenen − beklenen) / sqrt(beklenen)
    distance_km: float
    ev_t: np.ndarray         # bölgedeki tüm depremler
    ev_mag: np.ndarray
    ev_depth: np.ndarray
    pair_ref: np.ndarray     # eşleşen çiftler: referans deprem sırası
    pair_ev: np.ndarray      # ... ve bölgedeki deprem sırası (ev_* dizilerine göre)


@dataclass
class EventMatchSearch:
    n_ref: int
    ref_t: np.ndarray
    ref_mag: np.ndarray
    ref_depth: np.ndarray
    regions: list
    scanned: int             # eşleşme içeren aday bölge sayısı
    tested: int              # tesadüf testi yapılan bölge sayısı
    span_days: int


def _pairs(ref_day, ref_mag, ref_depth, ev_day, ev_mag, ev_depth, tol_days, dmag, ddepth):
    """Zaman (gün), büyüklük ve derinlik toleransına uyan (referans, olay) çiftleri. ev_day sıralı olmalı."""
    lo = np.searchsorted(ev_day, ref_day - tol_days, side="left")
    hi = np.searchsorted(ev_day, ref_day + tol_days, side="right")
    pi, pj = [], []
    for i in np.nonzero(hi > lo)[0]:
        sl = slice(lo[i], hi[i])
        ok = (np.abs(ev_mag[sl] - ref_mag[i]) <= dmag + 1e-9) & (np.abs(ev_depth[sl] - ref_depth[i]) <= ddepth)
        js = np.nonzero(ok)[0]
        if len(js):
            pi.append(np.full(len(js), i))
            pj.append(js + lo[i])
    if not pi:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(pi), np.concatenate(pj)


def _shift_null(ref_day, ref_mag, ref_depth, ev_day, ev_mag, ev_depth, tol_days, dmag, ddepth, day0, span):
    """Referans takvimi 0..span-1 gün dairesel kaydırıldığında eşi bulunan referans depremi sayıları.

    Dönüş: uzunluğu span olan dizi; [0] gözlenen değerdir. Tüm kaydırmalar tek geçişte hesaplanır:
    büyüklük/derinliği uyan her çift için, o çifti eşleştiren kaydırmalar işaretlenir.
    """
    hits = np.zeros((len(ref_day), span), dtype=bool)
    ks = np.arange(-tol_days, tol_days + 1)
    for a in range(0, len(ref_day), 200):
        sl = slice(a, a + 200)
        ok = (np.abs(ev_mag[None, :] - ref_mag[sl, None]) <= dmag + 1e-9) & \
             (np.abs(ev_depth[None, :] - ref_depth[sl, None]) <= ddepth)
        ii, jj = np.nonzero(ok)
        if len(ii) == 0:
            continue
        delta = (ev_day[jj] - ref_day[sl][ii]) % span          # bu kadar kaydırınca aynı güne gelir
        s = (delta[:, None] + ks[None, :]) % span
        hits[(ii + a)[:, None].repeat(len(ks), 1), s] = True
    return hits.sum(axis=0)


def find_event_matches(cat: Catalog, ref_lat, ref_lon, radius_km, *, t0, t1, min_mag, tol_days=1,
                       dmag=0.3, ddepth=10.0, top_n=15, n_tested=80, min_matched=2) -> EventMatchSearch:
    """Referans bölgenin her depremi için, dünyada yakın zamanda (±tol_days gün) benzer büyüklük
    (±dmag) ve derinlikte (±ddepth km) deprem olan bölgeleri bulur.

    Bir bölgenin puanı: eşi o bölgede bulunan *farklı* referans depremi sayısı. Çok deprem üreten
    bir bölge her şeyle tesadüfen eşleşir; bu yüzden her aday için beklenen değer, referans
    takvimi gün gün kaydırılarak ölçülür ve bölgeler gözlenen−beklenen farkına (z) göre sıralanır.
    """
    sel = np.nonzero((cat.t >= t0) & (cat.t < t1) & (cat.mag >= min_mag))[0]
    sub = cat.subset(sel)
    day = (sub.t // 86400).astype(np.int64)
    day0 = int(t0 // 86400)
    span = max(int(math.ceil((t1 - t0) / 86400)), 2 * tol_days + 2)
    d_ref = haversine_km(ref_lat, ref_lon, sub.lat, sub.lon)
    ref_idx = np.nonzero(d_ref <= radius_km)[0]
    out = EventMatchSearch(len(ref_idx), sub.t[ref_idx], sub.mag[ref_idx], sub.depth[ref_idx], [], 0, 0, span)
    if len(ref_idx) == 0:
        return out
    excl_km = sep_km = max(2 * radius_km, 300.0)
    w = np.nonzero(d_ref > excl_km)[0]                      # dünya (referans çevresi hariç), zamana göre sıralı
    if len(w) == 0:
        return out
    r_day, r_mag, r_dep = day[ref_idx], sub.mag[ref_idx], sub.depth[ref_idx]
    pi, pj = _pairs(r_day, r_mag, r_dep, day[w], sub.mag[w], sub.depth[w], tol_days, dmag, ddepth)
    if len(pi) == 0:
        return out
    # hücre dizini (1°×1°): tüm dünya olayları ve eşleşen olaylar
    cla, clo = np.floor(sub.lat[w]).astype(int), np.floor(sub.lon[w]).astype(int)
    cells = defaultdict(list)
    for k, key in enumerate(zip(cla, clo)):
        cells[key].append(k)
    cells = {k: np.asarray(v) for k, v in cells.items()}
    refs_of = defaultdict(set)                               # w içi olay sırası -> eşleştiği referanslar
    for i, j in zip(pi, pj):
        refs_of[int(j)].add(int(i))
    centers = {(int(cla[j]), int(clo[j])) for j in refs_of}
    dlat_cells = int(math.ceil(radius_km / 111.0)) + 1
    cands = []
    for (a0, b0) in centers:
        c_lat, c_lon = a0 + 0.5, b0 + 0.5
        coslat = max(math.cos(math.radians(c_lat)), 0.05)
        dlon_cells = min(int(math.ceil(radius_km / (111.0 * coslat))) + 1, 180)
        pool = np.concatenate([cells[(a, ((b + 180) % 360) - 180)]
                               for a in range(a0 - dlat_cells, a0 + dlat_cells + 1)
                               for b in range(b0 - dlon_cells, b0 + dlon_cells + 1)
                               if (a, ((b + 180) % 360) - 180) in cells])
        reg = np.sort(pool[haversine_km(c_lat, c_lon, sub.lat[w][pool], sub.lon[w][pool]) <= radius_km])
        matched = set()
        for j in reg:
            if int(j) in refs_of:
                matched |= refs_of[int(j)]
        if len(matched) >= min_matched:
            cands.append((len(matched), c_lat, c_lon, reg))
    out.scanned = len(cands)
    cands.sort(key=lambda x: -x[0])
    guard = 2 * tol_days + 1
    tested = []
    for score, c_lat, c_lon, reg in cands[:n_tested]:
        gi = w[reg]
        null = _shift_null(r_day, r_mag, r_dep, day[gi], sub.mag[gi], sub.depth[gi],
                           tol_days, dmag, ddepth, day0, span)
        body = null[guard: span - guard + 1] if span > 2 * guard + 2 else null[1:]
        exp = float(body.mean()) if len(body) else float("nan")
        p = float((1 + (body >= score).sum()) / (1 + len(body))) if len(body) else float("nan")
        z = (score - exp) / math.sqrt(max(exp, 0.25))
        tested.append((z, score, exp, p, c_lat, c_lon, reg))
    out.tested = len(tested)
    tested.sort(key=lambda x: (-x[0], -x[1]))
    chosen = []
    for z, score, exp, p, c_lat, c_lon, reg in tested:
        if any(float(haversine_km(c_lat, c_lon, np.array([r.lat]), np.array([r.lon]))[0]) < sep_km
               for r in chosen):
            continue
        gi = w[reg]
        pos = {int(j): k for k, j in enumerate(reg)}
        keep = [(int(i), pos[int(j)]) for i, j in zip(pi, pj) if int(j) in pos]
        chosen.append(EventRegion(
            c_lat, c_lon, region_name(sub, gi), len(gi), score, 100.0 * score / len(ref_idx), exp, p, z,
            float(haversine_km(ref_lat, ref_lon, np.array([c_lat]), np.array([c_lon]))[0]),
            sub.t[gi], sub.mag[gi], sub.depth[gi],
            np.array([k[0] for k in keep], dtype=int), np.array([k[1] for k in keep], dtype=int)))
        if len(chosen) >= top_n:
            break
    out.regions = chosen
    return out
