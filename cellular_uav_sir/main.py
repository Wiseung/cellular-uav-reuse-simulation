from __future__ import annotations

from pathlib import Path

from .config import SimulationConfig
from .experiments import run_all_experiments
from .plots import generate_all_plots


def _save_table(dataframe, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def main() -> None:
    config = SimulationConfig()
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
