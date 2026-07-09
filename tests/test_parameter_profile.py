from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cellular_uav_sir.config import SimulationConfig
from cellular_uav_sir.parameter_profile import apply_parameter_profile


class ParameterProfileTests(unittest.TestCase):
    def test_apply_parameter_profile_overrides_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            profile_path = temp_path / "profile.json"
            results_dir = temp_path / "results"
            profile_path.write_text(
                json.dumps(
                    {
                        "beam": {
                            "sector_azimuths_deg": [30.0, 150.0, 270.0],
                            "mechanical_downtilt_deg": 6.0,
                        },
                        "load": {
                            "dynamic_load_mean": 0.55,
                            "dynamic_max_users_per_site": 12,
                        },
                        "handover": {
                            "handover_hysteresis_db": 1.5,
                            "handover_time_to_trigger_steps": 4,
                        },
                        "paths": {
                            "results_dir": str(results_dir),
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = apply_parameter_profile(SimulationConfig(), profile_path)

        self.assertEqual(config.sector_azimuths_deg, (30.0, 150.0, 270.0))
        self.assertEqual(config.mechanical_downtilt_deg, 6.0)
        self.assertEqual(config.dynamic_load_mean, 0.55)
        self.assertEqual(config.dynamic_max_users_per_site, 12)
        self.assertEqual(config.handover_hysteresis_db, 1.5)
        self.assertEqual(config.handover_time_to_trigger_steps, 4)
        self.assertEqual(config.results_dir, results_dir)
        self.assertTrue(config.external_profile_applied)


if __name__ == "__main__":
    unittest.main()
