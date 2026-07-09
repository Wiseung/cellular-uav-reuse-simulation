from __future__ import annotations

import argparse
from pathlib import Path

from cellular_uav_sir.calibration import build_parameter_profile, write_parameter_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate an external beam/load/handover profile from site-layout and trace data."
    )
    parser.add_argument("--site-layout-csv", type=Path, help="Enhanced site-layout CSV with azimuth/tilt metadata.")
    parser.add_argument(
        "--dynamic-trace-csv",
        type=Path,
        help="Dynamic trace CSV, such as table_7_dynamic_trace.csv, for load and handover calibration.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output profile JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.site_layout_csv is None and args.dynamic_trace_csv is None:
        raise SystemExit("Provide at least one of --site-layout-csv or --dynamic-trace-csv.")

    profile = build_parameter_profile(
        site_layout_csv=args.site_layout_csv,
        dynamic_trace_csv=args.dynamic_trace_csv,
    )
    write_parameter_profile(profile, args.output)
    print(f"Wrote calibrated parameter profile to {args.output}")


if __name__ == "__main__":
    main()
