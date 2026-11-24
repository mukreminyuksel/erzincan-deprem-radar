"""Duman testi: uygulamayı her panel için baştan sona çalıştırır, çökme olmamalı.

Ağ erişimi gerekir (canlı kaynaklar). Yavaş olduğu için yalnız açıkça istenince
çalışır:   pytest -m smoke
GitHub Actions her push'ta çalıştırır (.github/workflows/tests.yml).

Menü bir iframe bileşeni olduğu için testten tıklanamıyor; panel, uygulamadaki
DR_TEST_PANEL ortam değişkeniyle seçilir.
"""
import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "earthquake.py"


def _panels():
    """earthquake.py içindeki _KATEGORILER sözlüğünden panel adlarını okur (uygulamayı çalıştırmadan)."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_KATEGORILER" for t in node.targets):
            kategoriler = ast.literal_eval(node.value)
            return [p[0] for k in kategoriler.values() for p in k["panels"]]
    raise AssertionError("_KATEGORILER bulunamadı")


PANELS = _panels()


def test_panel_list_is_not_empty():
    assert len(PANELS) >= 30


@pytest.mark.smoke
@pytest.mark.parametrize("panel", PANELS)
def test_panel_runs_without_exception(panel, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DR_TEST_PANEL", panel)
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    errors = [str(e.value)[:500] for e in at.exception]
    assert not errors, f"{panel}: {errors}"
    # Panel başlığı ya da içeriği gerçekten çizilmiş olmalı (boş sayfa değil)
    assert len(at.markdown) > 5, f"{panel}: sayfa neredeyse boş"


@pytest.mark.smoke
def test_center_and_large_radius(monkeypatch):
    """Kyoto merkezi + 2000 km yarıçap: veri çekme ve harita hatasız."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("DR_TEST_PANEL", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.query_params.update({"sb_center": "Kyoto, Japonya", "sb_clat": "35.0116",
                            "sb_clon": "135.7681", "sb_radius": "2000"})
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert at.session_state["center"][0] == "Kyoto, Japonya"
    assert at.session_state["sb_radius"] == 2000



@pytest.mark.smoke
def test_long_period_includes_1939_erzincan(monkeypatch):
    """110 yıllık M5+ sorgusunda 1939 Erzincan (M7.7–7.8) gelmeli.

    v1.88 öncesi kaynaklar en yeni 1000 olayı aldığı için eski büyük depremler
    hiç görünmüyordu (kullanıcı bildirimi, 24.11.2026).
    """
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("DR_TEST_PANEL", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=240)
    at.query_params.update({"sb_radius": "300", "sb_min_mag": "5.0",
                            "sb_zaman": "Özel gün sayısı", "sb_days": "40000"})
    at.run()
    at.radio(key="list_sort").set_value("En büyük önce").run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    cards = "".join(str(m.value) for m in at.markdown if "eq-card" in str(m.value)
                    and "eq-scroll-container" in str(m.value))
    assert "26.12.1939" in cards, "1939 Erzincan depremi listede yok"
