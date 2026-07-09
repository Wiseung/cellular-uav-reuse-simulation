from __future__ import annotations

import unittest

from tools.prepare_overture_3dep_buildings import enrich_overture_feature_collection


class PrepareOverture3depBuildingsTests(unittest.TestCase):
    def test_enrich_overture_feature_collection_normalizes_height_and_ground_elevation(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "num_floors": 4,
                        "facade_material": "brick",
                        "roof_material": "metal",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-84.0001, 35.0000],
                                [-84.0001, 35.0001],
                                [-84.0000, 35.0001],
                                [-84.0000, 35.0000],
                                [-84.0001, 35.0000],
                            ]
                        ],
                    },
                }
            ],
        }

        enriched = enrich_overture_feature_collection(
            payload,
            default_floor_height_m=3.5,
            default_height_m=12.0,
            ground_elevation_lookup=lambda latitude_deg, longitude_deg: 256.25,
        )

        properties = enriched["features"][0]["properties"]
        self.assertEqual(properties["height"], 14.0)
        self.assertEqual(properties["height_source"], "num_floors")
        self.assertEqual(properties["ground_elevation_m"], 256.25)
        self.assertEqual(properties["absolute_roof_elevation_m"], 270.25)
        self.assertEqual(properties["facade_material"], "brick")
        self.assertEqual(properties["roof_material"], "metal")


if __name__ == "__main__":
    unittest.main()
