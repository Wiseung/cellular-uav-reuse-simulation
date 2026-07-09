from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cellular_uav_sir.material_scenarios import (
    _classify_material_effect,
    _select_material_sensitive_scenarios,
    run_material_sensitive_scenarios,
)


class MaterialScenariosTests(unittest.TestCase):
    def test_classify_material_effect_distinguishes_relief_and_penalty(self) -> None:
        self.assertEqual(_classify_material_effect(0.5, 0.0, 1.0), "interferer_relief")
        self.assertEqual(_classify_material_effect(-0.5, 1.0, 0.0), "serving_penalty")
        self.assertEqual(_classify_material_effect(0.01, 0.2, 0.2), "mixed_blockage")

    def test_select_material_sensitive_scenarios_enforces_separation(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "delta_sinr_db": 1.0,
                    "delta_serving_loss_db": 0.0,
                    "delta_neighbor_loss_db": 1.0,
                    "effect_class": "interferer_relief",
                },
                {
                    "x_m": 5.0,
                    "y_m": 0.0,
                    "delta_sinr_db": 0.9,
                    "delta_serving_loss_db": 0.0,
                    "delta_neighbor_loss_db": 0.9,
                    "effect_class": "interferer_relief",
                },
                {
                    "x_m": 50.0,
                    "y_m": 0.0,
                    "delta_sinr_db": -1.2,
                    "delta_serving_loss_db": 1.2,
                    "delta_neighbor_loss_db": 0.0,
                    "effect_class": "serving_penalty",
                },
            ]
        )
        selected = _select_material_sensitive_scenarios(
            candidates,
            min_separation_m=10.0,
        )
        self.assertEqual(len(selected), 2)
        self.assertIn("scenario_id", selected.columns)
        self.assertNotIn(0.9, selected["delta_sinr_db"].tolist())
        self.assertIn("scenario_role", selected.columns)

    def test_select_material_sensitive_scenarios_skips_zero_effect_fallbacks(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "delta_sinr_db": 0.2,
                    "delta_serving_loss_db": 0.0,
                    "delta_neighbor_loss_db": 0.2,
                    "effect_class": "interferer_relief",
                },
                {
                    "x_m": 40.0,
                    "y_m": 0.0,
                    "delta_sinr_db": 0.0,
                    "delta_serving_loss_db": 0.0,
                    "delta_neighbor_loss_db": 0.0,
                    "effect_class": "weak_or_balanced",
                },
                {
                    "x_m": 80.0,
                    "y_m": 0.0,
                    "delta_sinr_db": 0.0,
                    "delta_serving_loss_db": 0.0,
                    "delta_neighbor_loss_db": 0.0,
                    "effect_class": "weak_or_balanced",
                },
            ]
        )
        selected = _select_material_sensitive_scenarios(
            candidates,
            min_separation_m=10.0,
            min_abs_delta_sinr_db=0.05,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            set(selected["scenario_role"].tolist()),
            {"interferer_relief_peak", "neutral_control"},
        )

    def test_run_material_sensitive_scenarios_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            site_layout_csv = temp_path / "site_layout.csv"
            site_layout_csv.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,radio,arfcn_list,sector_azimuths_deg,ground_elevation_m",
                        "s0,0,0,35.0,-84.0,LTE,100,0,250.0",
                        "s1,80,0,35.0,-83.99912,LTE,100,180,250.0",
                    ]
                ),
                encoding="utf-8",
            )
            building_geojson = temp_path / "buildings.geojson"
            building_geojson.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {
                                    "height": 18,
                                    "building:material": "brick",
                                    "roof:material": "concrete",
                                    "ground_elevation_m": 250.0,
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [-83.99967, 34.99982],
                                            [-83.99967, 35.00018],
                                            [-83.99945, 35.00018],
                                            [-83.99945, 34.99982],
                                            [-83.99967, 34.99982],
                                        ]
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            calibrated_profile_json = temp_path / "profile.json"
            calibrated_profile_json.write_text(
                json.dumps(
                    {
                        "beam": {"beamforming_enabled": False, "interferer_random_beams": False},
                        "handover": {"handover_time_to_trigger_steps": 1, "dynamic_min_dwell_steps": 0},
                        "load": {"dynamic_load_mean": 0.5, "dynamic_load_std": 0.0, "dynamic_load_correlation": 0.0},
                    }
                ),
                encoding="utf-8",
            )

            artifacts = run_material_sensitive_scenarios(
                site_layout_csv=site_layout_csv,
                building_geojson=building_geojson,
                calibrated_profile_json=calibrated_profile_json,
                output_dir=temp_path / "scenario_suite",
                scan_x_min_m=-40.0,
                scan_x_max_m=0.0,
                scan_y_min_m=-20.0,
                scan_y_max_m=20.0,
                scan_step_m=20.0,
                dynamic_altitude_m=0.0,
                trajectory_half_span_m=20.0,
                trajectory_steps=5,
            )

            self.assertTrue(artifacts.candidate_metrics_csv.exists())
            self.assertTrue(artifacts.scenario_definitions_csv.exists())
            self.assertTrue(artifacts.scenario_summary_csv.exists())
            self.assertTrue(artifacts.scenario_traces_csv.exists())
            self.assertTrue(artifacts.scenario_report_md.exists())
            self.assertTrue(artifacts.summary_onepager_png.exists())
            self.assertTrue(artifacts.summary_onepager_pdf.exists())
            self.assertTrue(artifacts.bilingual_onepager_png.exists())
            self.assertTrue(artifacts.bilingual_onepager_pdf.exists())
            definitions = pd.read_csv(artifacts.scenario_definitions_csv)
            self.assertFalse(definitions.empty)
            self.assertIn("scenario_role", definitions.columns)


if __name__ == "__main__":
    unittest.main()
