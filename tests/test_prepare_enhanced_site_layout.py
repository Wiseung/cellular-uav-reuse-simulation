from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from tools.prepare_enhanced_site_layout import prepare_enhanced_site_layout


class PrepareEnhancedSiteLayoutTests(unittest.TestCase):
    def test_prepare_enhanced_site_layout_groups_public_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "opencellid.csv"
            input_csv.write_text(
                "\n".join(
                    [
                        "radio,mcc,net,area,cell,unit,lon,lat,samples,averageSignal,azimuth,tx_power_dbm,antenna_height_m,antenna_peak_gain_db",
                        "LTE,310,260,1001,2001,1,-84.0000,35.0000,10,-92,30,48,28,18",
                        "LTE,310,260,1001,2002,2,-84.0001,35.0001,12,-90,150,50,32,19",
                        "LTE,310,260,1002,3001,3,-84.0100,35.0100,8,-95,270,46,25,17",
                    ]
                ),
                encoding="utf-8",
            )
            output_csv = temp_path / "site_layout.csv"

            grouped_sites = prepare_enhanced_site_layout(
                input_csv=input_csv,
                source="opencellid",
                output_csv=output_csv,
                center_latitude_deg=35.0,
                center_longitude_deg=-84.0,
                radius_km=3.0,
                limit_sites=2,
                grouping_decimals=3,
                include_ground_elevation=False,
            )

            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(grouped_sites), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cell_count"], "2")
        self.assertEqual(rows[0]["sector_azimuths_deg"], "30.0|150.0")
        self.assertEqual(rows[0]["ground_elevation_m"], "")
        self.assertEqual(rows[0]["tx_power_dbm"], "49.0")
        self.assertEqual(rows[0]["antenna_height_m"], "30.0")
        self.assertEqual(rows[0]["antenna_peak_gain_db"], "18.5")
        self.assertNotEqual(rows[0]["x_m"], "")
        self.assertNotEqual(rows[0]["y_m"], "")

    def test_prepare_enhanced_site_layout_reads_headerless_opencellid_gzip_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "opencellid.csv.gz"
            with gzip.open(input_csv, "wt", encoding="utf-8", newline="") as handle:
                handle.write(
                    "\n".join(
                        [
                            "LTE,310,260,1001,2001,1,-84.0000,35.0000,500,10,1,1372103025,1779922627,-92",
                            "LTE,310,260,1001,2002,2,-84.0001,35.0001,600,12,1,1372103081,1779922516,-90",
                        ]
                    )
                )
            output_csv = temp_path / "site_layout.csv"

            grouped_sites = prepare_enhanced_site_layout(
                input_csv=input_csv,
                source="opencellid",
                output_csv=output_csv,
                center_latitude_deg=35.0,
                center_longitude_deg=-84.0,
                radius_km=3.0,
                limit_sites=1,
                grouping_decimals=3,
                include_ground_elevation=False,
            )

        self.assertEqual(len(grouped_sites), 1)
        self.assertEqual(grouped_sites[0]["cell_count"], 2)


if __name__ == "__main__":
    unittest.main()
