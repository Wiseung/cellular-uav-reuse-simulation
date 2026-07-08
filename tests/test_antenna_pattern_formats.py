from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cellular_uav_sir.antenna import _load_pattern_file_cached, _peak_gain_db
from cellular_uav_sir.config import SimulationConfig


class AntennaPatternFormatTests(unittest.TestCase):
    def test_load_msi_pattern_file_reads_gain_and_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pattern_path = Path(temp_dir) / "panel.msi"
            pattern_path.write_text(
                "\n".join(
                    [
                        "NAME Reference panel",
                        "GAIN 18 dBi",
                        "HORIZONTAL 4",
                        "0 0",
                        "90 10",
                        "180 20",
                        "270 30",
                        "VERTICAL 4",
                        "0 0",
                        "90 12",
                        "180 24",
                        "270 36",
                    ]
                ),
                encoding="utf-8",
            )
            pattern = _load_pattern_file_cached(str(pattern_path.resolve()))

        self.assertEqual(pattern.peak_gain_db, 18.0)
        self.assertEqual(pattern.cuts["horizontal"][0].tolist(), [0.0, 90.0, 180.0])
        self.assertEqual(pattern.cuts["horizontal"][1].tolist(), [0.0, 20.0, 20.0])
        self.assertEqual(pattern.cuts["vertical"][1].tolist(), [0.0, 24.0, 24.0])

    def test_peak_gain_prefers_pattern_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pattern_path = Path(temp_dir) / "panel.msi"
            pattern_path.write_text(
                "\n".join(
                    [
                        "GAIN 19.5 dBi",
                        "HORIZONTAL 2",
                        "0 0",
                        "180 30",
                        "VERTICAL 2",
                        "0 0",
                        "180 30",
                    ]
                ),
                encoding="utf-8",
            )
            config = replace(
                SimulationConfig(),
                antenna_pattern_file=pattern_path,
                antenna_peak_gain_db=17.0,
            )

            self.assertEqual(_peak_gain_db(config), 19.5)


if __name__ == "__main__":
    unittest.main()
