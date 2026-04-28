import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
import unittest

from earthquake_core import (
    activity_level,
    duration_from_quick_window,
    estimate_energy_joules,
    event_signature,
    nearest_fault_vertex_distance,
    has_active_sources,
    parse_usgs_feed_features,
    safe_html,
    source_agreement_summary,
    to_utc_naive,
    usgs_feed_url_for_window,
)


class EarthquakeCoreTests(unittest.TestCase):
    def test_safe_html_escapes_external_text(self):
        self.assertEqual(
            safe_html('<script>alert("x")</script> & Erzincan'),
            '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; Erzincan',
        )

    def test_safe_html_handles_missing_values(self):
        self.assertEqual(safe_html(None), "")

    def test_to_utc_naive_converts_aware_datetime(self):
        local_time = datetime(2026, 4, 28, 15, 30, tzinfo=timezone.utc)
        self.assertEqual(to_utc_naive(local_time), datetime(2026, 4, 28, 15, 30))

    def test_to_utc_naive_keeps_naive_datetime_for_service_compatibility(self):
        naive = datetime(2026, 4, 28, 12, 0)
        self.assertEqual(to_utc_naive(naive), naive)

    def test_has_active_sources_rejects_empty_selection(self):
        self.assertFalse(has_active_sources(()))
        self.assertFalse(has_active_sources([]))
        self.assertTrue(has_active_sources(("AFAD",)))

    def test_duration_from_quick_window_supports_hours_and_days(self):
        self.assertEqual(duration_from_quick_window("Son 1 saat").total_seconds(), 3600)
        self.assertEqual(duration_from_quick_window("Son 12 saat").total_seconds(), 43200)
        self.assertEqual(duration_from_quick_window("Son 5 gün").days, 5)
        self.assertEqual(duration_from_quick_window("Son 30 gün").days, 30)

    def test_estimate_energy_joules_increases_with_magnitude(self):
        self.assertGreater(estimate_energy_joules(4.0), estimate_energy_joules(3.0))
        self.assertEqual(round(estimate_energy_joules(0), 3), round(10 ** 4.8, 3))

    def test_source_agreement_summary_counts_sources(self):
        events = [
            {"kaynak": "AFAD", "buyukluk": 3.0},
            {"kaynak": "Kandilli", "buyukluk": 3.2},
            {"kaynak": "USGS-Fast", "buyukluk": 2.8},
        ]
        summary = source_agreement_summary(events)
        self.assertEqual(summary["source_count"], 3)
        self.assertEqual(summary["magnitude_min"], 2.8)
        self.assertEqual(summary["magnitude_max"], 3.2)

    def test_activity_level_labels_score(self):
        self.assertEqual(activity_level(10), "Sakin")
        self.assertEqual(activity_level(35), "Dikkat")
        self.assertEqual(activity_level(65), "Yüksek")
        self.assertEqual(activity_level(90), "Çok Yüksek")

    def test_nearest_fault_vertex_distance_uses_fault_vertices(self):
        faults = [
            {"fay_adi": "Yakın Fay", "lats": [39.7333], "lons": [39.4917]},
            {"fay_adi": "Uzak Fay", "lats": [41.0], "lons": [29.0]},
        ]
        nearest = nearest_fault_vertex_distance(39.7333, 39.4917, faults)
        self.assertEqual(nearest["fault_name"], "Yakın Fay")
        self.assertEqual(nearest["distance_km"], 0.0)

    def test_event_signature_is_stable_for_same_event_values(self):
        first = event_signature("2026-04-28 12:00:00", 39.73331, 39.49172, 3.24)
        second = event_signature("2026-04-28 12:00:00", 39.73334, 39.49174, 3.23)
        self.assertEqual(first, second)

    def test_usgs_feed_url_prefers_smallest_realtime_window(self):
        now = datetime(2026, 4, 28, 12, 0)
        self.assertTrue(
            usgs_feed_url_for_window(now.replace(hour=11), now).endswith("/all_hour.geojson")
        )
        self.assertTrue(
            usgs_feed_url_for_window(datetime(2026, 4, 27, 12, 1), now).endswith("/all_day.geojson")
        )
        self.assertTrue(
            usgs_feed_url_for_window(datetime(2026, 4, 22, 12, 1), now).endswith("/all_week.geojson")
        )
        self.assertTrue(
            usgs_feed_url_for_window(datetime(2026, 4, 1, 12, 1), now).endswith("/all_month.geojson")
        )
        self.assertIsNone(usgs_feed_url_for_window(datetime(2026, 3, 1, 12, 0), now))

    def test_parse_usgs_feed_features_filters_and_normalizes_rows(self):
        features = [
            {
                "properties": {
                    "mag": 3.2,
                    "place": "10 km W of Erzincan <unsafe>",
                    "time": 1777388400000,
                },
                "geometry": {"coordinates": [39.49, 39.73, 7.4]},
            },
            {
                "properties": {"mag": 1.1, "place": "Too small", "time": 1777388400000},
                "geometry": {"coordinates": [39.49, 39.73, 5]},
            },
            {
                "properties": {"mag": 4.0, "place": "Too far", "time": 1777388400000},
                "geometry": {"coordinates": [29.0, 41.0, 5]},
            },
        ]

        rows = parse_usgs_feed_features(features, 39.7333, 39.4917, 50, 2.0)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kaynak"], "USGS-Fast")
        self.assertEqual(rows[0]["buyukluk"], 3.2)
        self.assertEqual(rows[0]["derinlik"], 7.4)
        self.assertIn("Erzincan", rows[0]["konum"])


if __name__ == "__main__":
    unittest.main()
