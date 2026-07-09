from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cellular_uav_sir.public_data_pipeline import run_public_data_pipeline


class PublicDataPipelineTests(unittest.TestCase):
    def test_run_public_data_pipeline_writes_report_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_source_csv = temp_path / "raw_cells.csv"
            raw_source_csv.write_text(
                "\n".join(
                    [
                        "radio,mcc,net,area,cell,unit,lon,lat,samples,averageSignal,azimuth",
                        "LTE,310,260,1001,2001,1,-84.0000,35.0000,10,-92,0",
                        "LTE,310,260,1002,3001,2,-83.9985,35.0000,8,-94,180",
                    ]
                ),
                encoding="utf-8",
            )
            overture_geojson = temp_path / "overture.geojson"
            overture_geojson.write_text(
                json.dumps(
                    {
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
                                            [-83.9994, 34.9999],
                                            [-83.9994, 35.0001],
                                            [-83.9992, 35.0001],
                                            [-83.9992, 34.9999],
                                            [-83.9994, 34.9999],
                                        ]
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            artifacts = run_public_data_pipeline(
                raw_source_csv=raw_source_csv,
                source="opencellid",
                overture_geojson=overture_geojson,
                output_root=temp_path / "pipeline_output",
                include_ground_elevation=False,
                default_profile_json=None,
                simulation_overrides={
                    "monte_carlo_samples": 8,
                    "reuse_factors": (1, 3),
                    "default_reuse_factor": 1,
                    "uav_altitudes_m": (0, 100),
                    "cdf_altitudes_m": (0,),
                    "pathloss_exponent_sweep": (3.5,),
                    "dynamic_time_steps": 3,
                    "dynamic_path_half_length_m": 60.0,
                    "dynamic_path_lateral_offset_m": 0.0,
                    "dynamic_altitude_m": 0.0,
                    "beamforming_enabled": False,
                    "interferer_random_beams": False,
                },
            )

            report_text = artifacts.comparison_report_md.read_text(encoding="utf-8")
            comparison_metrics = pd.read_csv(artifacts.comparison_metrics_csv)
            building_payload = json.loads(artifacts.enhanced_buildings_geojson.read_text(encoding="utf-8"))
            self.assertTrue(artifacts.enhanced_site_layout_csv.exists())
            self.assertTrue(artifacts.initial_profile_json.exists())
            self.assertTrue(artifacts.calibrated_profile_json.exists())
            self.assertTrue(artifacts.baseline_results_dir.joinpath("table_6_dynamic_summary.csv").exists())
            self.assertTrue(artifacts.enhanced_results_dir.joinpath("table_6_dynamic_summary.csv").exists())
            self.assertIn("Before: enhanced site layout only", report_text)
            self.assertIn("After: enhanced site layout + Overture/3DEP buildings", report_text)
            self.assertIn("facade:brick", report_text)
            self.assertFalse(comparison_metrics.empty)
            self.assertEqual(
                building_payload["features"][0]["properties"]["height_source"],
                "num_floors",
            )

    def test_run_public_data_pipeline_accepts_prepared_building_geojson(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_source_csv = temp_path / "raw_cells.csv"
            raw_source_csv.write_text(
                "\n".join(
                    [
                        "radio,mcc,net,area,cell,unit,lon,lat,samples,averageSignal,azimuth",
                        "LTE,310,260,1001,2001,1,-84.0000,35.0000,10,-92,0",
                    ]
                ),
                encoding="utf-8",
            )
            prepared_building_geojson = temp_path / "osm_buildings.geojson"
            prepared_building_geojson.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {
                                    "height": 18,
                                    "building:material": "brick",
                                    "roof:material": "metal",
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
                ),
                encoding="utf-8",
            )

            artifacts = run_public_data_pipeline(
                raw_source_csv=raw_source_csv,
                source="opencellid",
                overture_geojson=None,
                prepared_building_geojson=prepared_building_geojson,
                building_source_label="OSM material buildings",
                output_root=temp_path / "pipeline_output",
                include_ground_elevation=False,
                default_profile_json=None,
                simulation_overrides={
                    "monte_carlo_samples": 4,
                    "reuse_factors": (1,),
                    "default_reuse_factor": 1,
                    "uav_altitudes_m": (0,),
                    "cdf_altitudes_m": (0,),
                    "pathloss_exponent_sweep": (3.5,),
                    "dynamic_time_steps": 2,
                    "beamforming_enabled": False,
                    "interferer_random_beams": False,
                },
            )

            report_text = artifacts.comparison_report_md.read_text(encoding="utf-8")

        self.assertIn("OSM material buildings", report_text)


if __name__ == "__main__":
    unittest.main()
