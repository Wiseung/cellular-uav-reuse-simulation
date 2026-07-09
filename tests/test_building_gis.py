from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cellular_uav_sir.building_gis import (
    BuildingDataset,
    BuildingFootprint,
    evaluate_gis_los,
    evaluate_gis_los_and_loss,
    load_building_dataset,
)


class BuildingGisTests(unittest.TestCase):
    def test_evaluate_gis_los_blocks_segment(self) -> None:
        building = BuildingFootprint(
            polygon_xy_m=np.array(
                [
                    [45.0, -10.0],
                    [55.0, -10.0],
                    [55.0, 10.0],
                    [45.0, 10.0],
                ],
                dtype=float,
            ),
            height_m=40.0,
            min_x_m=45.0,
            max_x_m=55.0,
            min_y_m=-10.0,
            max_y_m=10.0,
        )
        dataset = BuildingDataset(
            buildings=(building,),
            min_x_m=0.0,
            max_x_m=100.0,
            min_y_m=-20.0,
            max_y_m=20.0,
        )

        covered, los = evaluate_gis_los(
            site_positions_xy_m=np.array([[0.0, 0.0]], dtype=float),
            user_point_xy_m=np.array([100.0, 0.0], dtype=float),
            tx_height_m=25.0,
            rx_height_m=25.0,
            building_dataset=dataset,
        )

        self.assertEqual(covered.tolist(), [True])
        self.assertEqual(los.tolist(), [False])

    def test_evaluate_gis_los_falls_back_outside_dataset_bounds(self) -> None:
        building = BuildingFootprint(
            polygon_xy_m=np.array(
                [
                    [10.0, -10.0],
                    [20.0, -10.0],
                    [20.0, 10.0],
                    [10.0, 10.0],
                ],
                dtype=float,
            ),
            height_m=40.0,
            min_x_m=10.0,
            max_x_m=20.0,
            min_y_m=-10.0,
            max_y_m=10.0,
        )
        dataset = BuildingDataset(
            buildings=(building,),
            min_x_m=0.0,
            max_x_m=30.0,
            min_y_m=-20.0,
            max_y_m=20.0,
        )

        covered, los = evaluate_gis_los(
            site_positions_xy_m=np.array([[50.0, 0.0]], dtype=float),
            user_point_xy_m=np.array([40.0, 0.0], dtype=float),
            tx_height_m=25.0,
            rx_height_m=25.0,
            building_dataset=dataset,
        )

        self.assertEqual(covered.tolist(), [False])
        self.assertEqual(los.tolist(), [True])

    def test_evaluate_gis_los_and_loss_adds_excess_loss_for_blocked_path(self) -> None:
        building = BuildingFootprint(
            polygon_xy_m=np.array(
                [
                    [45.0, -10.0],
                    [55.0, -10.0],
                    [55.0, 10.0],
                    [45.0, 10.0],
                ],
                dtype=float,
            ),
            height_m=40.0,
            min_x_m=45.0,
            max_x_m=55.0,
            min_y_m=-10.0,
            max_y_m=10.0,
        )
        dataset = BuildingDataset(
            buildings=(building,),
            min_x_m=0.0,
            max_x_m=100.0,
            min_y_m=-20.0,
            max_y_m=20.0,
        )

        covered, los, excess_loss_db = evaluate_gis_los_and_loss(
            site_positions_xy_m=np.array([[0.0, 0.0]], dtype=float),
            user_point_xy_m=np.array([100.0, 0.0], dtype=float),
            tx_height_m=25.0,
            rx_height_m=25.0,
            building_dataset=dataset,
            carrier_frequency_ghz=3.5,
            penetration_loss_per_meter_db=0.35,
            penetration_loss_cap_db=18.0,
            diffraction_loss_cap_db=24.0,
            total_excess_loss_cap_db=32.0,
        )

        self.assertEqual(covered.tolist(), [True])
        self.assertEqual(los.tolist(), [False])
        self.assertGreater(float(excess_loss_db[0]), 0.0)

    def test_evaluate_gis_los_and_loss_uses_building_materials_for_penetration_loss(self) -> None:
        glass_building = BuildingFootprint(
            polygon_xy_m=np.array(
                [
                    [45.0, -10.0],
                    [55.0, -10.0],
                    [55.0, 10.0],
                    [45.0, 10.0],
                ],
                dtype=float,
            ),
            height_m=40.0,
            min_x_m=45.0,
            max_x_m=55.0,
            min_y_m=-10.0,
            max_y_m=10.0,
            facade_material="glass",
        )
        concrete_building = BuildingFootprint(
            polygon_xy_m=np.array(
                [
                    [45.0, -10.0],
                    [55.0, -10.0],
                    [55.0, 10.0],
                    [45.0, 10.0],
                ],
                dtype=float,
            ),
            height_m=40.0,
            min_x_m=45.0,
            max_x_m=55.0,
            min_y_m=-10.0,
            max_y_m=10.0,
            facade_material="concrete",
        )

        glass_dataset = BuildingDataset(
            buildings=(glass_building,),
            min_x_m=0.0,
            max_x_m=100.0,
            min_y_m=-20.0,
            max_y_m=20.0,
        )
        concrete_dataset = BuildingDataset(
            buildings=(concrete_building,),
            min_x_m=0.0,
            max_x_m=100.0,
            min_y_m=-20.0,
            max_y_m=20.0,
        )

        _, _, glass_loss_db = evaluate_gis_los_and_loss(
            site_positions_xy_m=np.array([[0.0, 0.0]], dtype=float),
            user_point_xy_m=np.array([100.0, 0.0], dtype=float),
            tx_height_m=25.0,
            rx_height_m=25.0,
            building_dataset=glass_dataset,
            carrier_frequency_ghz=3.5,
            penetration_loss_per_meter_db=0.35,
            penetration_loss_cap_db=18.0,
            diffraction_loss_cap_db=2.0,
            total_excess_loss_cap_db=32.0,
        )
        _, _, concrete_loss_db = evaluate_gis_los_and_loss(
            site_positions_xy_m=np.array([[0.0, 0.0]], dtype=float),
            user_point_xy_m=np.array([100.0, 0.0], dtype=float),
            tx_height_m=25.0,
            rx_height_m=25.0,
            building_dataset=concrete_dataset,
            carrier_frequency_ghz=3.5,
            penetration_loss_per_meter_db=0.35,
            penetration_loss_cap_db=18.0,
            diffraction_loss_cap_db=2.0,
            total_excess_loss_cap_db=32.0,
        )

        self.assertGreater(float(concrete_loss_db[0]), float(glass_loss_db[0]))

    def test_evaluate_gis_los_and_loss_uses_external_material_profile(self) -> None:
        building = BuildingFootprint(
            polygon_xy_m=np.array(
                [
                    [45.0, -10.0],
                    [55.0, -10.0],
                    [55.0, 10.0],
                    [45.0, 10.0],
                ],
                dtype=float,
            ),
            height_m=40.0,
            min_x_m=45.0,
            max_x_m=55.0,
            min_y_m=-10.0,
            max_y_m=10.0,
            facade_material="glass",
        )
        dataset = BuildingDataset(
            buildings=(building,),
            min_x_m=0.0,
            max_x_m=100.0,
            min_y_m=-20.0,
            max_y_m=20.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "material_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "materials": [
                            {
                                "tokens": ["glass"],
                                "penetration_multiplier": 2.0,
                                "entry_loss_db": 6.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            _, _, custom_loss_db = evaluate_gis_los_and_loss(
                site_positions_xy_m=np.array([[0.0, 0.0]], dtype=float),
                user_point_xy_m=np.array([100.0, 0.0], dtype=float),
                tx_height_m=25.0,
                rx_height_m=25.0,
                building_dataset=dataset,
                carrier_frequency_ghz=3.5,
                penetration_loss_per_meter_db=0.35,
                penetration_loss_cap_db=18.0,
                diffraction_loss_cap_db=2.0,
                total_excess_loss_cap_db=40.0,
                material_loss_profile_path=profile_path,
            )

        self.assertGreater(float(custom_loss_db[0]), 10.0)

    def test_load_building_dataset_uses_geojson_height_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            site_layout_path = temp_path / "site_layout.csv"
            site_layout_path.write_text(
                "site_id,x_m,y_m,latdec,londec\n0,0,0,35.0,-84.0\n",
                encoding="utf-8",
            )

            geojson_path = temp_path / "buildings.geojson"
            geojson_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"height": "18 m"},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [-84.00010, 35.00000],
                                            [-84.00010, 35.00005],
                                            [-84.00005, 35.00005],
                                            [-84.00005, 35.00000],
                                            [-84.00010, 35.00000],
                                        ]
                                    ],
                                },
                            },
                            {
                                "type": "Feature",
                                "properties": {"building:levels": "3"},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [-84.00020, 35.00000],
                                            [-84.00020, 35.00005],
                                            [-84.00015, 35.00005],
                                            [-84.00015, 35.00000],
                                            [-84.00020, 35.00000],
                                        ]
                                    ],
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = load_building_dataset(
                building_geojson_path=geojson_path,
                site_layout_csv=site_layout_path,
                default_height_m=12.0,
                level_height_m=3.0,
            )

        heights_m = sorted(round(building.height_m, 3) for building in dataset.buildings)
        self.assertEqual(heights_m, [9.0, 18.0])

    def test_load_building_dataset_uses_ground_elevation_and_overture_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            site_layout_path = temp_path / "site_layout.csv"
            site_layout_path.write_text(
                "site_id,x_m,y_m,latdec,londec,ground_elevation_m\n0,0,0,35.0,-84.0,250.0\n",
                encoding="utf-8",
            )

            geojson_path = temp_path / "buildings.geojson"
            geojson_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {
                                    "num_floors": "5",
                                    "ground_elevation_m": "258.5",
                                    "building:material": "glass",
                                    "roof_material": "metal",
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [-84.00010, 35.00000],
                                            [-84.00010, 35.00005],
                                            [-84.00005, 35.00005],
                                            [-84.00005, 35.00000],
                                            [-84.00010, 35.00000],
                                        ]
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = load_building_dataset(
                building_geojson_path=geojson_path,
                site_layout_csv=site_layout_path,
                default_height_m=12.0,
                level_height_m=3.0,
            )

        building = dataset.buildings[0]
        self.assertEqual(building.height_m, 15.0)
        self.assertEqual(building.obstruction_height_m, 23.5)
        self.assertEqual(building.facade_material, "glass")
        self.assertEqual(building.roof_material, "metal")
        self.assertEqual(building.height_source, "num_floors")


if __name__ == "__main__":
    unittest.main()
