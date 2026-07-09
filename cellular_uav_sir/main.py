from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import SimulationConfig
from .experiments import run_all_experiments
from .parameter_profile import resolve_runtime_config
from .plots import generate_all_plots


def _save_table(dataframe, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cellular + UAV reuse simulation.")
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="Optional external parameter profile JSON for beam/load/handover/path overrides.",
    )
    parser.add_argument(
        "--site-layout-csv",
        type=Path,
        help="Optional dynamic site-layout CSV override.",
    )
    parser.add_argument(
        "--building-geojson",
        type=Path,
        help="Optional building-footprint GeoJSON override.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Optional results directory override.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    defaults = SimulationConfig()
    config = SimulationConfig(
        parameter_profile_json=args.profile_json
        if args.profile_json is not None
        else defaults.parameter_profile_json,
    )
    config = resolve_runtime_config(config)
    runtime_overrides = {}
    if args.site_layout_csv is not None:
        runtime_overrides["dynamic_site_layout_csv"] = args.site_layout_csv
    if args.building_geojson is not None:
        runtime_overrides["building_footprint_geojson"] = args.building_geojson
    if args.results_dir is not None:
        runtime_overrides["results_dir"] = args.results_dir
    if runtime_overrides:
        config = replace(config, **runtime_overrides)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    experiment_bundle = run_all_experiments(config)
    _save_table(
        experiment_bundle.sir_vs_reuse,
        config.results_dir / "table_1_sir_vs_reuse.csv",
    )
    _save_table(
        experiment_bundle.ase_vs_reuse,
        config.results_dir / "table_2_ase_vs_reuse.csv",
    )
    _save_table(
        experiment_bundle.sinr_vs_height,
        config.results_dir / "table_3_sir_vs_height.csv",
    )
    _save_table(
        experiment_bundle.cdf_samples,
        config.results_dir / "table_4_sir_cdf_samples.csv",
    )
    _save_table(
        experiment_bundle.pathloss_sweep,
        config.results_dir / "table_5_pathloss_sweep.csv",
    )
    _save_table(
        experiment_bundle.dynamic_summary,
        config.results_dir / "table_6_dynamic_summary.csv",
    )
    _save_table(
        experiment_bundle.dynamic_trace,
        config.results_dir / "table_7_dynamic_trace.csv",
    )
    _save_table(
        experiment_bundle.dynamic_site_layout,
        config.results_dir / "table_8_dynamic_site_layout.csv",
    )

    generate_all_plots(
        config=config,
        sir_vs_reuse=experiment_bundle.sir_vs_reuse,
        ase_vs_reuse=experiment_bundle.ase_vs_reuse,
        sinr_vs_height=experiment_bundle.sinr_vs_height,
        cdf_samples=experiment_bundle.cdf_samples,
        pathloss_sweep=experiment_bundle.pathloss_sweep,
        dynamic_trace=experiment_bundle.dynamic_trace,
        dynamic_site_layout=experiment_bundle.dynamic_site_layout,
    )

    print(f"Results written to: {config.results_dir}")
    print("Generated tables:")
    for filename in (
        "table_1_sir_vs_reuse.csv",
        "table_2_ase_vs_reuse.csv",
        "table_3_sir_vs_height.csv",
        "table_4_sir_cdf_samples.csv",
        "table_5_pathloss_sweep.csv",
        "table_6_dynamic_summary.csv",
        "table_7_dynamic_trace.csv",
        "table_8_dynamic_site_layout.csv",
    ):
        print(f"  - {filename}")
    print("Generated figures:")
    for filename in (
        "figure_1_reuse_geometry.png",
        "figure_2_sir_vs_reuse.png",
        "figure_3_ase_vs_reuse.png",
        "figure_4_sir_vs_height.png",
        "figure_5_sir_cdf.png",
        "figure_6_pathloss_sweep.png",
        "figure_7_los_probability_vs_height.png",
        "figure_8_dynamic_sinr_timeline.png",
        "figure_9_dynamic_layout_map.png",
    ):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
