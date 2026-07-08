from __future__ import annotations

import unittest

from tools.download_osm_buildings import _deduplicate_features, _tile_bboxes


class DownloadOsmBuildingsTests(unittest.TestCase):
    def test_tile_bboxes_splits_bbox_with_overlap(self) -> None:
        tiles = _tile_bboxes(
            south=0.0,
            west=0.0,
            north=0.05,
            east=0.05,
            tile_size_deg=0.03,
            overlap_deg=0.005,
        )

        self.assertEqual(
            tiles,
            [
                (0.0, 0.0, 0.03, 0.03),
                (0.0, 0.024999999999999998, 0.03, 0.05),
                (0.024999999999999998, 0.0, 0.05, 0.03),
                (0.024999999999999998, 0.024999999999999998, 0.05, 0.05),
            ],
        )

    def test_deduplicate_features_keeps_unique_osm_way_ids(self) -> None:
        features = [
            {
                "type": "Feature",
                "properties": {"osm_way_id": 1},
                "geometry": {"type": "Polygon", "coordinates": []},
            },
            {
                "type": "Feature",
                "properties": {"osm_way_id": 2},
                "geometry": {"type": "Polygon", "coordinates": []},
            },
            {
                "type": "Feature",
                "properties": {"osm_way_id": 1},
                "geometry": {"type": "Polygon", "coordinates": []},
            },
        ]

        deduplicated = _deduplicate_features(features)
        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(
            sorted(feature["properties"]["osm_way_id"] for feature in deduplicated),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
