"""Fay ve plaka sınırı veri dosyalarının bütünlüğü (ağ gerekmez)."""
import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def faults():
    return json.loads((ROOT / "turkey_faults.geojson").read_text(encoding="utf-8"))["features"]


@pytest.fixture(scope="module")
def plates():
    return {f["properties"]["Name"]: f for f in
            json.loads((ROOT / "tectonic_plates.geojson").read_text(encoding="utf-8"))["features"]}


def _points(geom):
    lines = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
    return [p[:2] for line in lines for p in line]


def test_fault_names_have_no_control_characters(faults):
    bad = {v for f in faults for v in f["properties"].values()
           if isinstance(v, str) and any(unicodedata.category(ch).startswith("C") for ch in v)}
    assert not bad, sorted(bad)[:5]


def test_fault_coordinates_are_lon_lat_in_turkey(faults):
    pts = np.array([p for f in faults[::50] for p in _points(f["geometry"])])
    assert 25 < pts[:, 0].min() and pts[:, 0].max() < 46      # boylam
    assert 35 < pts[:, 1].min() and pts[:, 1].max() < 43      # enlem


@pytest.mark.parametrize("code,zone", [
    ("AT-EU", "KUZEY ANADOLU FAY ZONU"),
    ("AT-AR", "DOĞU ANADOLU FAY ZONU"),
])
def test_refined_plate_boundary_follows_mta_trace(plates, faults, code, zone):
    """Türkiye içindeki plaka sınırı noktaları MTA fay izinin üzerinde (<0.5 km)."""
    Z = np.array([p for f in faults if f["properties"]["fay_adi"] == zone for p in _points(f["geometry"])])
    inside = [p for p in _points(plates[code]["geometry"]) if 27 <= p[0] <= 44 and 36.5 <= p[1] <= 41.5]
    assert len(inside) > 100
    d = [np.hypot((Z[:, 0] - x) * math.cos(math.radians(y)) * 111.2, (Z[:, 1] - y) * 111.2).min()
         for x, y in inside[::5]]
    assert np.median(d) < 0.5


def test_original_pb2002_kept_for_reproducibility():
    assert (ROOT / "data" / "tectonic_plates_pb2002.geojson").exists()


def test_karliova_triple_junction_is_connected(plates):
    """AT-EU, AT-AR ve EU-AR Karlıova'da aynı noktada buluşmalı (v1.92'de kopukluk düzeltildi)."""
    ends = {}
    for code in ("AT-EU", "AT-AR", "EU-AR"):
        pts = _points(plates[code]["geometry"])
        ends[code] = {tuple(round(v, 4) for v in pts[0]), tuple(round(v, 4) for v in pts[-1])}
    common = ends["AT-EU"] & ends["AT-AR"] & ends["EU-AR"]
    assert len(common) == 1, ends
    lon, lat = next(iter(common))
    assert 40.8 < lon < 41.5 and 39.0 < lat < 39.6     # Karlıova çevresi


def test_refined_boundaries_have_no_backtracking(plates):
    for code in ("AT-EU", "AT-AR", "EU-AR"):
        c = _points(plates[code]["geometry"])
        for a, b, d in zip(c, c[1:], c[2:]):
            cl = math.cos(math.radians(b[1]))
            v1 = ((b[0] - a[0]) * cl, b[1] - a[1])
            v2 = ((d[0] - b[0]) * cl, d[1] - b[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 and n2:
                assert (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2) > -0.5, (code, b)
