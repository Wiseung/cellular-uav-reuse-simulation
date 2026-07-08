from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from .config import SimulationConfig
from .geometry import (
    cochannel_interferers,
    edge_user_point,
    hexagon_vertices,
    perturb_site_positions,
    reuse_distance,
)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.frameon": True,
        }
    )


def plot_reuse_geometry(config: SimulationConfig, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    central_vertices = hexagon_vertices(config.cell_radius_m)
    nominal_first_tier = cochannel_interferers(
        config.default_reuse_factor,
        config.cell_radius_m,
        count=6,
    )
    first_tier = perturb_site_positions(
        nominal_first_tier,
        jitter_radius_m=(
            reuse_distance(config.default_reuse_factor, config.cell_radius_m)
            * config.site_perturbation_fraction
        ),
        rng=config.rng(offset=1),
    )

    ax.add_patch(
        Polygon(
            central_vertices,
            closed=True,
            facecolor="#d9edf7",
            edgecolor="#1f77b4",
            linewidth=2.0,
            label="Serving cell",
        )
    )
    for index, center in enumerate(first_tier):
        ax.add_patch(
            Polygon(
                hexagon_vertices(config.cell_radius_m, center=center),
                closed=True,
                facecolor="#fbe3d4",
                edgecolor="#d95f02",
                linewidth=1.5,
                alpha=0.9,
                label="Co-channel cell" if index == 0 else None,
            )
        )
        ax.plot(center[0], center[1], "o", color="#d95f02", markersize=4)

    edge_point = edge_user_point(config.cell_radius_m)
    ax.plot(0.0, 0.0, "o", color="#1f77b4", markersize=8, label="Serving BS")
    ax.plot(edge_point[0], edge_point[1], "^", color="#2ca02c", markersize=8, label="Edge UE")
    ax.plot(
        [0.0, edge_point[0]],
        [0.0, edge_point[1]],
        linestyle="--",
        color="#2ca02c",
        linewidth=1.5,
    )

    reference_interferer = first_tier[0]
    ax.plot(
        [0.0, reference_interferer[0]],
        [0.0, reference_interferer[1]],
        linestyle="--",
        color="#7f7f7f",
        linewidth=1.5,
    )
    ax.text(
        0.0,
        config.cell_radius_m * 0.55,
        "R",
        color="#2ca02c",
        fontsize=11,
        ha="left",
    )
    ax.text(
        reference_interferer[0] * 0.45,
        reference_interferer[1] * 0.45,
        f"D = {reuse_distance(config.default_reuse_factor, config.cell_radius_m):.0f} m",
        color="#444444",
        fontsize=10,
    )
    ax.set_title("N = 7 Perturbed Reuse Geometry and Co-Channel Cells")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    _save_figure(fig, output_path)


def plot_sir_vs_reuse(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots()
    x = data["reuse_factor"]
    ax.plot(x, data["analytic_sir_db"], marker="o", linewidth=2.2, label="Analytic edge SIR")
    ax.plot(
        x,
        data["baseline_hex_edge_sir_db"],
        marker="s",
        linewidth=2.0,
        label="Hex edge SIR baseline",
    )
    ax.plot(
        x,
        data["median_user_sir_db"],
        marker="D",
        linewidth=1.8,
        linestyle="-.",
        label="Median user SIR",
    )
    ax.plot(
        x,
        data["p05_user_sir_db"],
        marker="^",
        linewidth=1.8,
        linestyle="--",
        label="5th percentile user SIR",
    )
    ax.set_title("SIR versus Reuse Factor")
    ax.set_xlabel("Reuse factor N")
    ax.set_ylabel("SIR (dB)")
    ax.legend(loc="upper left")
    _save_figure(fig, output_path)


def plot_ase_vs_reuse(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax1 = plt.subplots()
    x = data["reuse_factor"].astype(str)
    ax1.plot(
        x,
        data["effective_ase_bphz_per_km2"],
        marker="o",
        color="#1f77b4",
        linewidth=2.2,
        label="Effective ASE",
    )
    ax1.set_xlabel("Reuse factor N")
    ax1.set_ylabel("Effective ASE (bit/s/Hz/km^2)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.bar(
        x,
        data["channel_share"],
        width=0.45,
        alpha=0.25,
        color="#ff7f0e",
        label="Channel share 1/N",
    )
    ax2.set_ylabel("Channel share per cell", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("Effective Area Spectral Efficiency versus Reuse Factor")
    _save_figure(fig, output_path)


def plot_sir_vs_height(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(
        data["height_m"],
        data["median_sir_db"],
        marker="o",
        linewidth=2.2,
        label="Median SIR",
    )
    ax.plot(
        data["height_m"],
        data["p05_sir_db"],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label="5th percentile SIR",
    )
    ax.set_title("Ground-to-UAV SIR Change versus Altitude")
    ax.set_xlabel("UAV altitude (m)")
    ax.set_ylabel("SIR (dB)")
    ax.legend(loc="upper right")
    _save_figure(fig, output_path)


def plot_los_probability_vs_height(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(
        data["height_m"],
        data["mean_serving_los_probability"],
        marker="o",
        linewidth=2.2,
        label="Serving-link LOS probability",
    )
    ax.plot(
        data["height_m"],
        data["mean_interferer_los_probability"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        label="Interferer LOS probability",
    )
    ax.set_title("LOS Probability Change versus UAV Altitude")
    ax.set_xlabel("UAV altitude (m)")
    ax.set_ylabel("Mean LOS probability")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    _save_figure(fig, output_path)


def plot_cdf(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots()
    for scenario, group in data.groupby("scenario"):
        sir_values = np.sort(group["sir_db"].to_numpy())
        cdf = np.arange(1, len(sir_values) + 1) / len(sir_values)
        ax.plot(sir_values, cdf, linewidth=2.0, label=scenario)
    ax.set_title("SIR CDF for Ground and UAV Users")
    ax.set_xlabel("SIR (dB)")
    ax.set_ylabel("Empirical CDF")
    ax.legend(loc="lower right")
    _save_figure(fig, output_path)


def plot_pathloss_sweep(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots()
    for exponent, group in data.groupby("pathloss_exponent"):
        ordered = group.sort_values("reuse_factor")
        ax.plot(
            ordered["reuse_factor"],
            ordered["analytic_sir_db"],
            marker="o",
            linewidth=2.0,
            label=f"n = {exponent:.1f}",
        )
    ax.set_title("Analytic SIR Sensitivity to Pathloss Exponent")
    ax.set_xlabel("Reuse factor N")
    ax.set_ylabel("Analytic edge SIR (dB)")
    ax.legend(loc="upper left")
    _save_figure(fig, output_path)


def generate_all_plots(
    config: SimulationConfig,
    sir_vs_reuse: pd.DataFrame,
    ase_vs_reuse: pd.DataFrame,
    sir_vs_height: pd.DataFrame,
    cdf_samples: pd.DataFrame,
    pathloss_sweep: pd.DataFrame,
) -> None:
    configure_style()
    plot_reuse_geometry(config, config.results_dir / "figure_1_reuse_geometry.png")
    plot_sir_vs_reuse(sir_vs_reuse, config.results_dir / "figure_2_sir_vs_reuse.png")
    plot_ase_vs_reuse(ase_vs_reuse, config.results_dir / "figure_3_ase_vs_reuse.png")
    plot_sir_vs_height(sir_vs_height, config.results_dir / "figure_4_sir_vs_height.png")
    plot_cdf(cdf_samples, config.results_dir / "figure_5_sir_cdf.png")
    plot_pathloss_sweep(
        pathloss_sweep,
        config.results_dir / "figure_6_pathloss_sweep.png",
    )
    plot_los_probability_vs_height(
        sir_vs_height,
        config.results_dir / "figure_7_los_probability_vs_height.png",
    )
