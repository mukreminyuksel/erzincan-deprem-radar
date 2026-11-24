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
