from __future__ import annotations

import argparse
from pathlib import Path

from cellular_uav_sir.public_data_pipeline import run_public_data_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full public-data pipeline from raw public cells to before/after gain report."
    )
    parser.add_argument("--source", choices=("opencellid", "cellmapper"), required=True)
    parser.add_argument("--raw-source-csv", type=Path, required=True)
    parser.add_argument("--overture-geojson", type=Path)
    parser.add_argument(
        "--prepared-building-geojson",
        type=Path,
        help="Use a prebuilt building GeoJSON directly instead of running the Overture preparer.",
    )
    parser.add_argument(
        "--building-source-label",
        default="Overture/3DEP buildings",
        help="Human-readable label written into the report for the building source.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--center-lat", type=float)
    parser.add_argument("--center-lon", type=float)
    parser.add_argument("--radius-km", type=float)
    parser.add_argument("--limit-sites", type=int)
    parser.add_argument("--grouping-decimals", type=int, default=4)
    parser.add_argument(
        "--skip-ground-elevation",
        action="store_true",
        help="Skip USGS 3DEP EPQS lookups for site and building elevation enrichment.",
    )
    parser.add_argument(
        "--default-profile-json",
        type=Path,
        default=Path("cellular_uav_sir/data/default_parameter_profile.json"),
        help="Optional base profile to merge with calibrated sections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overture_geojson is None and args.prepared_building_geojson is None:
        raise SystemExit("Provide --overture-geojson or --prepared-building-geojson.")
    artifacts = run_public_data_pipeline(
        raw_source_csv=args.raw_source_csv,
        source=args.source,
        overture_geojson=args.overture_geojson,
        output_root=args.output_root,
        prepared_building_geojson=args.prepared_building_geojson,
        building_source_label=args.building_source_label,
        center_latitude_deg=args.center_lat,
        center_longitude_deg=args.center_lon,
        radius_km=args.radius_km,
        limit_sites=args.limit_sites,
        grouping_decimals=args.grouping_decimals,
        include_ground_elevation=not args.skip_ground_elevation,
        default_profile_json=args.default_profile_json,
    )
    print(f"Enhanced site layout: {artifacts.enhanced_site_layout_csv}")
    print(f"Enhanced buildings: {artifacts.enhanced_buildings_geojson}")
    print(f"Initial profile: {artifacts.initial_profile_json}")
    print(f"Calibrated profile: {artifacts.calibrated_profile_json}")
    print(f"Baseline results: {artifacts.baseline_results_dir}")
    print(f"Enhanced results: {artifacts.enhanced_results_dir}")
    print(f"Comparison report: {artifacts.comparison_report_md}")
    print(f"Comparison metrics JSON: {artifacts.comparison_metrics_json}")
    print(f"Comparison metrics CSV: {artifacts.comparison_metrics_csv}")


if __name__ == "__main__":
    main()
