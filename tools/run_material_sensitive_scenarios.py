from __future__ import annotations

import argparse
from pathlib import Path

from cellular_uav_sir.material_scenarios import run_material_sensitive_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dedicated suite of material-sensitive trajectory scenarios on a real-data closure."
    )
    parser.add_argument("--site-layout-csv", type=Path, required=True)
    parser.add_argument("--building-geojson", type=Path, required=True)
    parser.add_argument("--calibrated-profile-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-x-min-m", type=float, default=-220.0)
    parser.add_argument("--scan-x-max-m", type=float, default=220.0)
    parser.add_argument("--scan-y-min-m", type=float, default=-220.0)
    parser.add_argument("--scan-y-max-m", type=float, default=220.0)
    parser.add_argument("--scan-step-m", type=float, default=20.0)
    parser.add_argument("--dynamic-altitude-m", type=float, default=0.0)
    parser.add_argument("--trajectory-half-span-m", type=float, default=60.0)
    parser.add_argument("--trajectory-steps", type=int, default=21)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_material_sensitive_scenarios(
        site_layout_csv=args.site_layout_csv,
        building_geojson=args.building_geojson,
        calibrated_profile_json=args.calibrated_profile_json,
        output_dir=args.output_dir,
        scan_x_min_m=args.scan_x_min_m,
        scan_x_max_m=args.scan_x_max_m,
        scan_y_min_m=args.scan_y_min_m,
        scan_y_max_m=args.scan_y_max_m,
        scan_step_m=args.scan_step_m,
        dynamic_altitude_m=args.dynamic_altitude_m,
        trajectory_half_span_m=args.trajectory_half_span_m,
        trajectory_steps=args.trajectory_steps,
    )
    print(f"Candidates: {artifacts.candidate_metrics_csv}")
    print(f"Definitions: {artifacts.scenario_definitions_csv}")
    print(f"Summary: {artifacts.scenario_summary_csv}")
    print(f"Traces: {artifacts.scenario_traces_csv}")
    print(f"Report: {artifacts.scenario_report_md}")
    print(f"One-page PNG: {artifacts.summary_onepager_png}")
    print(f"One-page PDF: {artifacts.summary_onepager_pdf}")
    print(f"Bilingual one-page PNG: {artifacts.bilingual_onepager_png}")
    print(f"Bilingual one-page PDF: {artifacts.bilingual_onepager_pdf}")
    print(f"Map figure: {artifacts.scenario_map_png}")
    print(f"Delta bar figure: {artifacts.scenario_delta_bar_png}")


if __name__ == "__main__":
    main()
