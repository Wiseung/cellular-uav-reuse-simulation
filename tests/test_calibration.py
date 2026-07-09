from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cellular_uav_sir.calibration import build_parameter_profile


class CalibrationTests(unittest.TestCase):
    def test_build_parameter_profile_reads_site_and_trace_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            site_layout_path = temp_path / "site_layout.csv"
            site_layout_path.write_text(
                "\n".join(
                    [
                        "site_id,x_m,y_m,latdec,londec,sector_azimuths_deg,mechanical_downtilt_deg,electrical_downtilt_deg",
                        "s0,0,0,35.0,-84.0,30|150|270,5,9",
                        "s1,100,0,35.0005,-84.0005,30|150|270,6,8",
                    ]
                ),
                encoding="utf-8",
            )
            dynamic_trace_path = temp_path / "trace.csv"
            dynamic_trace_path.write_text(
                "\n".join(
                    [
                        "step,serving_load,mean_neighbor_load,scheduled_users_serving,serving_measurement_dbm,best_neighbor_measurement_dbm,handover_flag",
                        "0,0.60,0.55,6,-80,-78,0",
                        "1,0.70,0.60,7,-81,-77,0",
                        "2,0.75,0.65,8,-82,-78,1",
                        "3,0.65,0.58,6,-79,-78,0",
                    ]
                ),
                encoding="utf-8",
            )

            profile = build_parameter_profile(
                site_layout_csv=site_layout_path,
                dynamic_trace_csv=dynamic_trace_path,
            )

        self.assertEqual(profile["beam"]["sector_azimuths_deg"], [30.0, 150.0, 270.0])
        self.assertAlmostEqual(profile["beam"]["mechanical_downtilt_deg"], 5.5)
        self.assertAlmostEqual(profile["beam"]["electrical_downtilt_deg"], 8.5)
        self.assertIn("dynamic_load_mean", profile["load"])
        self.assertEqual(profile["load"]["dynamic_max_users_per_site"], 8)
        self.assertAlmostEqual(profile["handover"]["handover_hysteresis_db"], 4.0)
        self.assertEqual(profile["handover"]["handover_time_to_trigger_steps"], 3)


if __name__ == "__main__":
    unittest.main()
