"""similarity.py için yapay katalogla testler (ağ erişimi yok)."""
import gzip
import math

import numpy as np
import pytest

import similarity as sim

DAY = 86400


def make_catalog(events):
    """events: [(t, lat, lon, depth, mag, place), ...]"""
    t, lat, lon, depth, mag, place = zip(*events)
    order = np.argsort(t, kind="stable")
    return sim.Catalog(np.asarray(t, dtype=np.int64)[order], np.asarray(lat)[order],
                       np.asarray(lon)[order], np.asarray(depth, dtype=float)[order],
                       np.asarray(mag)[order], [place[i] for i in order])


def cluster(lat, lon, n, *, t0=0, step_days=30, depth=10.0, mags=(4.6, 4.8, 5.2), place="X"):
    return [(t0 + i * step_days * DAY, lat + 0.01 * (i % 5), lon + 0.01 * (i % 7),
             depth, mags[i % len(mags)], place) for i in range(n)]


def test_haversine_known_distance():
    # Erzincan → Erzurum ≈ 155 km
    d = sim.haversine_km(39.7333, 39.4917, np.array([39.9043]), np.array([41.2679]))[0]
    assert 145 < d < 165


def test_b_value_needs_ten_events_and_is_positive():
    assert math.isnan(sim.b_value(np.array([4.6, 4.7, 5.0])))
    rng = np.random.default_rng(1)
    mags = np.round(4.5 + rng.exponential(1 / math.log(10), 2000), 1)   # b ≈ 1
    b = sim.b_value(mags)
    assert 0.8 < b < 1.2


def test_identical_profiles_score_100_and_different_lower():
    cat = make_catalog(cluster(0, 0, 40) + cluster(40, 40, 40, place="Y")
                       + cluster(-30, 100, 40, depth=500.0, mags=(6.6, 6.8), place="Z"))
    span = cat.span_years
    a = sim.profile(cat, sim.region_index(cat, 0, 0, 100), span)
    b = sim.profile(cat, sim.region_index(cat, 40, 40, 100), span)
    c = sim.profile(cat, sim.region_index(cat, -30, 100, 100), span)
    assert sim.profile_similarity(a, b) == 100.0
    assert sim.profile_similarity(a, c) < 50.0


def test_synchrony_counts_same_day_matches_only():
    # Rastgele zamanlı 20 referans olayı (20 yıla yayılmış); düzenli aralık kaydırma
    # testini yanıltır, o yüzden rastgele.
    rng = np.random.default_rng(7)
    times = np.sort(rng.integers(0, 20 * 365 * DAY, 20))
    ref = [(int(t), 0.0, 0.0, 10.0, (4.6, 4.8, 5.2)[i % 3], "R") for i, t in enumerate(times)]
    # aday: ilk 5 olay referansla aynı gün, kalanlar bağımsız rastgele zamanlarda
    other = rng.integers(0, 20 * 365 * DAY, 15)
    cand = [(int(t) + 3600, 50.0, 50.0, d, m, "C") for (t, la, lo, d, m, _p) in ref[:5]]
    cand += [(int(t), 50.0, 50.0, 10.0, 4.8, "C") for t in other]
    cat = make_catalog(ref + cand)
    r_idx = sim.region_index(cat, 0, 0, 100)
    c_idx = sim.region_index(cat, 50, 50, 100)
    s = sim.synchrony(cat, r_idx, c_idx, float(cat.t.min()), float(cat.t.max()), n_shifts=200)
    assert s.days == 5
    assert s.tested and s.expected < s.days
    assert s.p_value < 0.05
    assert len(s.examples) == 5


def test_synchrony_ignores_different_depth():
    ref = cluster(0, 0, 10, depth=10.0)
    cand = [(t, la + 50, lo + 50, 200.0, m, "C") for (t, la, lo, d, m, _p) in ref]
    cat = make_catalog(ref + cand)
    s = sim.synchrony(cat, sim.region_index(cat, 0, 0, 100), sim.region_index(cat, 50, 50, 100),
                      float(cat.t.min()), float(cat.t.max()))
    assert s.days == 0


def test_synchrony_counts_a_burst_as_one_day():
    # Referansta tek günde 10 olay (artçı fırtınası), adayda aynı gün 10 olay: 1 gün sayılır
    ref = [(1000 * DAY + i * 600, 0.0, 0.0, 10.0, 4.8, "R") for i in range(10)]
    ref += cluster(0, 0, 20, t0=2000 * DAY)
    cand = [(1000 * DAY + i * 700, 50.0, 50.0, 10.0, 4.8, "C") for i in range(10)]
    cand += cluster(50, 50, 20, t0=5000 * DAY, place="C")
    cat = make_catalog(ref + cand)
    s = sim.synchrony(cat, sim.region_index(cat, 0, 0, 100), sim.region_index(cat, 50, 50, 100),
                      float(cat.t.min()), float(cat.t.max()))
    assert s.days == 1


def test_find_similar_ranks_twin_first_and_excludes_reference_area():
    events = (cluster(39.7, 39.5, 40, place="Erzincan")            # referans
              + cluster(39.9, 41.3, 40, place="Yakın")             # 155 km: dışlanmalı
              + cluster(35.0, 135.7, 40, place="İkiz")             # aynı profil
              + cluster(-20, -70, 40, depth=120.0, mags=(6.1, 6.7), place="Farklı"))
    cat = make_catalog(events)
    ref_prof, n_ref, by_profile, by_sync, scanned = sim.find_similar(cat, 39.7, 39.5, 100)
    assert n_ref == 40 and scanned >= 2
    names = [c.name for c in by_profile]
    assert names[0] == "İkiz"
    assert "Yakın" not in names and "Erzincan" not in names
    assert by_profile[0].score > by_profile[-1].score


def test_find_similar_too_few_reference_events():
    cat = make_catalog(cluster(0, 0, 3) + cluster(50, 50, 40))
    _ref, n_ref, by_profile, by_sync, scanned = sim.find_similar(cat, 0, 0, 100)
    assert n_ref == 3 and by_profile == [] and by_sync == [] and scanned == 0


def test_region_status_ratio():
    st = sim.region_status([1, 2, 3], [0, 0, 30], [0, 0, 30], [4.6, 5.4, 6.0],
                           0, 0, 100, long_term_rate_per_year=12.175, days=30)
    assert st.n_recent == 2 and st.max_recent == 5.4
    assert st.ratio == pytest.approx(2.0, rel=0.01)


def test_load_catalog_roundtrip(tmp_path):
    p = tmp_path / "c.csv.gz"
    with gzip.open(p, "wt", encoding="utf-8", newline="") as f:
        f.write("t,lat,lon,depth,mag,place\n20,1,2,nan,4.6,B\n10,3,4,10.0,5.0,A\n")
    cat = sim.load_catalog(p)
    assert list(cat.t) == [10, 20] and cat.place == ["A", "B"]
    assert math.isnan(cat.depth[1])


def test_compare_regions_twin_and_yearly_series():
    cat = make_catalog(cluster(39.7, 39.5, 40, place="A") + cluster(35.9, -120.4, 40, place="B"))
    c = sim.compare_regions(cat, 39.7, 39.5, 35.9, -120.4, 100, n_shifts=50)
    assert c.n_a == 40 and c.n_b == 40
    assert c.score == 100.0
    assert c.distance_km > 10000
    assert c.yearly_a.sum() == 40 and c.yearly_b.sum() == 40
    assert len(c.years) == len(c.yearly_a)
    assert c.sync.tested


def test_compare_regions_too_few_events_gives_nan_score():
    cat = make_catalog(cluster(0, 0, 3) + cluster(50, 50, 40))
    c = sim.compare_regions(cat, 0, 0, 50, 50, 100)
    assert math.isnan(c.score) and not c.sync.tested


def _burst_catalog():
    """A ve B aynı aylarda etkin (ortak ritim), C farklı aylarda; hepsi A'dan uzakta."""
    rng = np.random.default_rng(3)
    month = 30 * DAY
    active = sorted(rng.choice(120, 30, replace=False))
    other = sorted(rng.choice(120, 30, replace=False))
    ev = []
    for m in active:
        for k in range(int(rng.integers(1, 5))):
            ev.append((m * month + k * DAY, 0.0, 0.0, 10.0, 4.8, "A"))
            ev.append((m * month + k * DAY + 3600, 40.0, 60.0, 10.0, 4.8, "B"))
    for m in other:
        for k in range(int(rng.integers(1, 5))):
            ev.append((m * month + k * DAY, -30.0, -70.0, 10.0, 4.8, "C"))
    return make_catalog(ev), 120 * month


def test_series_correlation_identical_and_shifted():
    rng = np.random.default_rng(0)
    a = rng.poisson(1.0, 120)
    r, p = sim.series_correlation(a, a)
    assert r == pytest.approx(1.0) and p < 0.05
    r2, p2 = sim.series_correlation(a, np.roll(a, 7))
    assert r2 < 0.5 and p2 > 0.01
    assert math.isnan(sim.series_correlation(np.zeros(50), a[:50])[0])


def test_find_time_correlated_finds_synchronous_region():
    cat, t1 = _burst_catalog()
    res = sim.find_time_correlated(cat, 0, 0, 100, t0=0, t1=t1)
    best = res.matches[0]
    assert best.r > res.chance[1]            # gerçek ortak ritim tesadüf eşiğini aşar
    assert res.n_ref >= 30 and res.scanned >= 2 and len(res.ref_series) == len(res.edges) - 1
    assert best.name == "B" and best.r > 0.9 and best.p_value < 0.05
    assert best.r_first > 0.8 and best.r_second > 0.8      # iki yarıda da tutarlı
    assert best.best_lag == 0
    assert best.depth_overlap == pytest.approx(100.0) and best.mag_overlap == pytest.approx(100.0)
    c = [m for m in res.matches if m.name == "C"]
    assert not c or c[0].r < 0.5


def test_overlap_pct_and_best_lag():
    assert sim.overlap_pct([0.5, 0.5, 0], [0.5, 0.5, 0]) == pytest.approx(100.0)
    assert sim.overlap_pct([1, 0, 0], [0, 0, 1]) == pytest.approx(0.0)
    rng = np.random.default_rng(5)
    a = rng.poisson(2.0, 100)
    lag, r = sim._best_lag(a, np.roll(a, -2))      # b, a'dan 2 dönem önce davranıyor
    assert lag == 2 and r > 0.9


def test_load_catalogs_merges_and_sorts(tmp_path):
    for name, rows in (("a.csv.gz", "30,1,2,5.0,2.5,X\n"), ("b.csv.gz", "10,3,4,7.0,3.1,Y\n")):
        with gzip.open(tmp_path / name, "wt", encoding="utf-8", newline="") as f:
            f.write("t,lat,lon,depth,mag,place\n" + rows)
    cat = sim.load_catalogs([tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"])
    assert list(cat.t) == [10, 30] and cat.place == ["Y", "X"]


def test_correlation_before_after_split():
    cat, t1 = _burst_catalog()
    edges, a, b = sim.pair_series(cat, 0, 0, 40, 60, 100, t0=0, t1=t1)
    rb, nb, ra, na = sim.correlation_before_after(edges, a, b, split_t=t1 / 2)
    assert rb > 0.9 and ra > 0.9 and nb + na == len(a)
    _rb, _nb, ra2, na2 = sim.correlation_before_after(edges, a, b, split_t=t1 - 60 * DAY)
    assert math.isnan(ra2) and na2 < 6


def test_find_event_matches_detects_paired_region():
    """A'nın her depreminin eşi B'de (aynı gün, aynı büyüklük/derinlik); C kalabalık ama ilgisiz."""
    rng = np.random.default_rng(11)
    days = np.sort(rng.choice(100, 12, replace=False))
    ev = []
    for k, d in enumerate(days):
        m, dp = 3.0 + 0.2 * (k % 5), 8.0 + (k % 3)
        ev.append((int(d) * DAY + 3600, 0.0, 0.0, dp, m, "A"))
        ev.append((int(d) * DAY + 7200, 40.0, 60.0, dp + 1, m + 0.1, "B"))
    for k in range(300):                                     # C: her gün birkaç deprem, farklı derinlik
        ev.append((int(rng.integers(0, 100)) * DAY + 500 + k, -30.0, -70.0, 120.0, 3.0 + 0.1 * (k % 10), "C"))
    cat = make_catalog(ev)
    res = sim.find_event_matches(cat, 0, 0, 100, t0=0, t1=100 * DAY, min_mag=2.0, tol_days=0, dmag=0.3, ddepth=5)
    assert res.n_ref == 12
    best = res.regions[0]
    assert best.name == "B" and best.matched == 12 and best.coverage == pytest.approx(100.0)
    assert best.expected < 3 and best.p_value < 0.05 and best.z > 3
    assert len(best.pair_ref) == 12 and set(best.pair_ref) == set(range(12))
    assert all(r.name != "C" for r in res.regions)           # derinliği uymadığı için hiç eşleşmez


def test_find_event_matches_busy_region_is_not_significant():
    """Çok deprem üreten bölge çok eşleşir ama beklenen de yüksektir (z düşük)."""
    rng = np.random.default_rng(12)
    ev = [(int(d) * DAY + 100, 0.0, 0.0, 10.0, 3.0, "A") for d in rng.choice(100, 15, replace=False)]
    ev += [(d * DAY + 50 * k, 40.0, 60.0, 10.0, 3.0, "Kalabalık") for d in range(100) for k in range(3)]
    res = sim.find_event_matches(make_catalog(ev), 0, 0, 100, t0=0, t1=100 * DAY, min_mag=2.0,
                                 tol_days=0, dmag=0.3, ddepth=5)
    r = res.regions[0]
    assert r.matched == 15 and r.expected == pytest.approx(15.0) and abs(r.z) < 0.5 and r.p_value > 0.5


def test_find_event_matches_no_reference_events():
    cat = make_catalog(cluster(50, 50, 20))
    res = sim.find_event_matches(cat, 0, 0, 100, t0=0, t1=1000 * DAY, min_mag=2.0)
    assert res.n_ref == 0 and res.regions == []
