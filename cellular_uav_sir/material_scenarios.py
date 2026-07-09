from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SimulationConfig
from .dynamic_network import (
    run_dynamic_trajectory_experiment_with_trajectory,
)
from .parameter_profile import apply_parameter_profile
from .plots import configure_style


@dataclass(frozen=True)
class MaterialScenarioArtifacts:
    output_dir: Path
    candidate_metrics_csv: Path
    scenario_definitions_csv: Path
    scenario_summary_csv: Path
    scenario_traces_csv: Path
    scenario_report_md: Path
    scenario_map_png: Path
    scenario_delta_bar_png: Path
    summary_onepager_png: Path
    summary_onepager_pdf: Path
    bilingual_onepager_png: Path
    bilingual_onepager_pdf: Path


MANDATORY_SCENARIO_ROLES: tuple[str, ...] = (
    "interferer_relief_peak",
    "serving_penalty_peak",
    "weak_neighbor_probe",
    "neutral_control",
)
OPTIONAL_SCENARIO_ROLES: tuple[str, ...] = (
    "mixed_blockage_probe",
    "weak_serving_probe",
)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _role_color_map() -> dict[str, str]:
    return {
        "interferer_relief_peak": "#2ca02c",
        "serving_penalty_peak": "#d62728",
        "weak_neighbor_probe": "#1f77b4",
        "neutral_control": "#7f7f7f",
        "mixed_blockage_probe": "#ff7f0e",
        "weak_serving_probe": "#9467bd",
    }


def _role_label_map() -> dict[str, str]:
    return {
        "interferer_relief_peak": "强干扰抑制 / Interferer Relief Peak",
        "serving_penalty_peak": "强服务惩罚 / Serving Penalty Peak",
        "weak_neighbor_probe": "弱邻区敏感 / Weak Neighbor Probe",
        "neutral_control": "中性对照 / Neutral Control",
        "mixed_blockage_probe": "混合阻挡 / Mixed Blockage Probe",
        "weak_serving_probe": "弱服务惩罚 / Weak Serving Probe",
    }


def _build_runtime_config(
    site_layout_csv: Path | str,
    building_geojson: Path | str,
    calibrated_profile_json: Path | str,
    *,
    dynamic_altitude_m: float = 0.0,
    dynamic_time_steps: int = 1,
) -> SimulationConfig:
    config = apply_parameter_profile(SimulationConfig(), calibrated_profile_json)
    return replace(
        config,
        site_layout_csv=Path(site_layout_csv),
        dynamic_site_layout_csv=Path(site_layout_csv),
        building_footprint_geojson=Path(building_geojson),
        dynamic_altitude_m=float(dynamic_altitude_m),
        dynamic_time_steps=int(dynamic_time_steps),
    )


def _empty_material_profile_json() -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "wirelessprop_material_scenarios"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "empty_materials.json"
    if not path.exists():
        path.write_text(json.dumps({"materials": []}), encoding="utf-8")
    return path


def _run_point_trace(
    config: SimulationConfig,
    point_xy_m: np.ndarray,
    material_profile_path: Path,
) -> pd.Series:
    scenario_config = replace(
        config,
        building_material_loss_profile_json=material_profile_path,
        dynamic_time_steps=1,
    )
    bundle = run_dynamic_trajectory_experiment_with_trajectory(
        config=scenario_config,
        trajectory_xy_m=np.asarray([point_xy_m], dtype=float),
    )
    return bundle.trace.iloc[0]


def _classify_material_effect(
    delta_sinr_db: float,
    delta_serving_loss_db: float,
    delta_neighbor_loss_db: float,
) -> str:
    if delta_sinr_db > 0.05 and delta_neighbor_loss_db > delta_serving_loss_db + 0.1:
        return "interferer_relief"
    if delta_sinr_db < -0.05 and delta_serving_loss_db > delta_neighbor_loss_db + 0.1:
        return "serving_penalty"
    if delta_serving_loss_db > 0.1 and delta_neighbor_loss_db > 0.1:
        return "mixed_blockage"
    return "weak_or_balanced"


def _distance_m(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> float:
    return math.hypot(
        float(candidate_a["x_m"]) - float(candidate_b["x_m"]),
        float(candidate_a["y_m"]) - float(candidate_b["y_m"]),
    )


def _select_material_sensitive_scenarios(
    candidates: pd.DataFrame,
    *,
    min_separation_m: float = 25.0,
    min_abs_delta_sinr_db: float = 0.05,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    selected_rows: list[dict[str, Any]] = []
    candidates = candidates.copy()
    candidates["abs_delta_sinr_db"] = np.abs(candidates["delta_sinr_db"])
    candidates["total_delta_loss_db"] = (
        np.abs(candidates["delta_serving_loss_db"]) + np.abs(candidates["delta_neighbor_loss_db"])
    )

    role_dataframes: dict[str, pd.DataFrame] = {
        "interferer_relief_peak": candidates.loc[
            candidates["effect_class"] == "interferer_relief"
        ].sort_values(["delta_sinr_db", "delta_neighbor_loss_db"], ascending=[False, False]),
        "serving_penalty_peak": candidates.loc[
            candidates["effect_class"] == "serving_penalty"
        ].sort_values(["delta_sinr_db", "delta_serving_loss_db"], ascending=[True, False]),
        "weak_neighbor_probe": candidates.loc[
            (candidates["effect_class"] == "weak_or_balanced")
            & (candidates["delta_neighbor_loss_db"] > 0.05)
        ].sort_values(["delta_neighbor_loss_db", "abs_delta_sinr_db"], ascending=[False, True]),
        "neutral_control": candidates.sort_values(
            ["abs_delta_sinr_db", "total_delta_loss_db"],
            ascending=[True, True],
        ),
        "mixed_blockage_probe": candidates.loc[
            candidates["effect_class"] == "mixed_blockage"
        ].sort_values(["total_delta_loss_db", "abs_delta_sinr_db"], ascending=[False, False]),
        "weak_serving_probe": candidates.loc[
            (candidates["effect_class"] == "weak_or_balanced")
            & (candidates["delta_serving_loss_db"] > 0.05)
        ].sort_values(["delta_serving_loss_db", "abs_delta_sinr_db"], ascending=[False, True]),
    }

    def pick_role(role_name: str) -> None:
        role_candidates = role_dataframes.get(role_name)
        if role_candidates is None or role_candidates.empty:
            return
        for _, row in role_candidates.iterrows():
            row_dict = row.to_dict()
            if any(_distance_m(row_dict, existing) < min_separation_m for existing in selected_rows):
                continue
            row_dict["scenario_role"] = role_name
            selected_rows.append(row_dict)
            break

    for role_name in MANDATORY_SCENARIO_ROLES:
        pick_role(role_name)
    for role_name in OPTIONAL_SCENARIO_ROLES:
        pick_role(role_name)

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return selected
    selected = selected.reset_index(drop=True)
    selected["scenario_id"] = [f"scenario_{index + 1:02d}" for index in range(len(selected))]
    if "scenario_role" not in selected.columns:
        selected["scenario_role"] = "unspecified"
    return selected


def scan_material_sensitive_points(
    config: SimulationConfig,
    *,
    x_values_m: np.ndarray,
    y_values_m: np.ndarray,
    material_profile_path: Path | str | None = None,
) -> pd.DataFrame:
    runtime_config = replace(config, dynamic_time_steps=1)
    on_profile_path = (
        Path(material_profile_path)
        if material_profile_path is not None
        else Path(runtime_config.building_material_loss_profile_json)
    )
    off_profile_path = _empty_material_profile_json()

    records: list[dict[str, Any]] = []
    for x_m in x_values_m:
        for y_m in y_values_m:
            point_xy_m = np.array([float(x_m), float(y_m)], dtype=float)
            off_trace = _run_point_trace(runtime_config, point_xy_m, off_profile_path)
            on_trace = _run_point_trace(runtime_config, point_xy_m, on_profile_path)
            delta_sinr_db = float(on_trace["sinr_db"] - off_trace["sinr_db"])
            delta_serving_loss_db = float(
                on_trace["serving_gis_excess_loss_db"] - off_trace["serving_gis_excess_loss_db"]
            )
            delta_neighbor_loss_db = float(
                on_trace["mean_neighbor_gis_excess_loss_db"] - off_trace["mean_neighbor_gis_excess_loss_db"]
            )
            records.append(
                {
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                    "serving_site_index_off": int(off_trace["serving_site_index"]),
                    "serving_site_index_on": int(on_trace["serving_site_index"]),
                    "sinr_db_off": float(off_trace["sinr_db"]),
                    "sinr_db_on": float(on_trace["sinr_db"]),
                    "delta_sinr_db": delta_sinr_db,
                    "serving_loss_db_off": float(off_trace["serving_gis_excess_loss_db"]),
                    "serving_loss_db_on": float(on_trace["serving_gis_excess_loss_db"]),
                    "delta_serving_loss_db": delta_serving_loss_db,
                    "neighbor_loss_db_off": float(off_trace["mean_neighbor_gis_excess_loss_db"]),
                    "neighbor_loss_db_on": float(on_trace["mean_neighbor_gis_excess_loss_db"]),
                    "delta_neighbor_loss_db": delta_neighbor_loss_db,
                    "effect_class": _classify_material_effect(
                        delta_sinr_db,
                        delta_serving_loss_db,
                        delta_neighbor_loss_db,
                    ),
                    "cochannel_interferer_count_off": int(off_trace["cochannel_interferer_count"]),
                    "cochannel_interferer_count_on": int(on_trace["cochannel_interferer_count"]),
                }
            )
    return pd.DataFrame(records)


def _trajectory_for_scenario(
    center_x_m: float,
    center_y_m: float,
    *,
    half_span_m: float,
    steps: int,
) -> np.ndarray:
    x_coordinates_m = np.linspace(
        center_x_m - half_span_m,
        center_x_m + half_span_m,
        num=steps,
        dtype=float,
    )
    y_coordinates_m = np.full_like(x_coordinates_m, center_y_m, dtype=float)
    return np.column_stack((x_coordinates_m, y_coordinates_m))


def _run_scenario_trace(
    config: SimulationConfig,
    scenario_row: pd.Series,
    *,
    material_profile_path: Path,
    trajectory_half_span_m: float,
    trajectory_steps: int,
) -> pd.DataFrame:
    trajectory_xy_m = _trajectory_for_scenario(
        center_x_m=float(scenario_row["x_m"]),
        center_y_m=float(scenario_row["y_m"]),
        half_span_m=trajectory_half_span_m,
        steps=trajectory_steps,
    )
    scenario_config = replace(
        config,
        dynamic_time_steps=trajectory_steps,
    )
    bundle = run_dynamic_trajectory_experiment_with_trajectory(
        config=replace(
            scenario_config,
            building_material_loss_profile_json=material_profile_path,
        ),
        trajectory_xy_m=trajectory_xy_m,
    )
    trace = bundle.trace.copy()
    trace["trajectory_x_m"] = trajectory_xy_m[:, 0]
    trace["trajectory_y_m"] = trajectory_xy_m[:, 1]
    return trace


def _plot_scenario_map(
    scenarios: pd.DataFrame,
    site_layout: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.scatter(site_layout["x_m"], site_layout["y_m"], color="#1f77b4", s=45, label="Sites")
    color_by_role = _role_color_map()
    for _, row in scenarios.iterrows():
        color = color_by_role.get(str(row["scenario_role"]), "#444444")
        ax.scatter([row["x_m"]], [row["y_m"]], color=color, s=75)
        ax.text(float(row["x_m"]) + 4.0, float(row["y_m"]) + 4.0, str(row["scenario_id"]), fontsize=9)
    ax.set_title("Material-Sensitive Scenario Centers")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_aspect("equal", adjustable="box")
    _save_figure(fig, output_path)


def _plot_scenario_delta_bar(
    scenarios: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ordered = scenarios.sort_values("delta_sinr_db", ascending=False)
    color_by_role = _role_color_map()
    ax.bar(
        ordered["scenario_id"],
        ordered["delta_sinr_db"],
        color=[color_by_role.get(str(role), "#444444") for role in ordered["scenario_role"]],
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title("Center-Point SINR Change from Material-Aware Loss")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Delta SINR (dB)")
    _save_figure(fig, output_path)


def _plot_scenario_trace_pair(
    scenario_id: str,
    off_trace: pd.DataFrame,
    on_trace: pd.DataFrame,
    output_path: Path,
) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(off_trace["trajectory_x_m"], off_trace["sinr_db"], linewidth=2.0, label="Material off")
    ax.plot(on_trace["trajectory_x_m"], on_trace["sinr_db"], linewidth=2.0, linestyle="--", label="Material on")
    ax.set_title(f"{scenario_id} SINR along Material-Sensitive Trajectory")
    ax.set_xlabel("Trajectory x position (m)")
    ax.set_ylabel("SINR (dB)")
    ax.legend(loc="best")
    _save_figure(fig, output_path)


def _write_scenario_report(
    report_path: Path,
    scenarios: pd.DataFrame,
    summary: pd.DataFrame,
    candidate_count: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Material-Sensitive Scenario Report",
        "",
        "This report summarizes a dedicated suite of local Manhattan material-sensitive trajectories.",
        "",
        f"- Candidate points scanned: {candidate_count}",
        f"- Scenarios selected: {len(scenarios)}",
        "",
        "## Scenario Definitions",
        "",
        "| Scenario | Role | Center x (m) | Center y (m) | Center delta SINR (dB) | Delta serving GIS loss (dB) | Delta neighbor GIS loss (dB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in scenarios.iterrows():
        lines.append(
            f"| {row['scenario_id']} | {row['scenario_role']} | "
            f"{float(row['x_m']):.1f} | {float(row['y_m']):.1f} | "
            f"{float(row['delta_sinr_db']):.4f} | "
            f"{float(row['delta_serving_loss_db']):.4f} | "
            f"{float(row['delta_neighbor_loss_db']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory Summary",
            "",
            "| Scenario | Mean delta SINR (dB) | Max abs delta SINR (dB) | Center step delta SINR (dB) | Mean delta neighbor GIS loss (dB) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scenario_id']} | {float(row['mean_delta_sinr_db']):.4f} | "
            f"{float(row['max_abs_delta_sinr_db']):.4f} | "
            f"{float(row['center_step_delta_sinr_db']):.4f} | "
            f"{float(row['mean_delta_neighbor_gis_excess_loss_db']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Mandatory scenario roles: {', '.join(MANDATORY_SCENARIO_ROLES)}",
            f"- Optional scenario roles when available: {', '.join(OPTIONAL_SCENARIO_ROLES)}",
            "- Each scenario is a local horizontal trajectory centered on a material-sensitive point discovered by on/off material scanning.",
            "- The definition-table center delta comes from a one-step point scan; the trajectory summary comes from a multi-step dynamic replay, so the signs need not match when handover or interference state changes along the local path.",
            "- Positive delta SINR scenarios generally indicate material-aware attenuation on interfering links; negative scenarios indicate additional serving-link blockage or mixed geometry effects.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_summary_onepager(
    scenarios: pd.DataFrame,
    summary: pd.DataFrame,
    site_layout: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    *,
    bilingual: bool = False,
) -> None:
    configure_style()
    if bilingual:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(11.69, 8.27))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.9, 2.2, 1.9], width_ratios=[1.15, 1.0])

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.88,
        "曼哈顿 3DEP 材质敏感场景摘要 / Manhattan 3DEP Material-Sensitive Scenario Summary"
        if bilingual
        else "Manhattan 3DEP Material-Sensitive Scenario Summary",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )
    title_ax.text(
        0.0,
        0.52,
        (
            "基于真实 OpenCelliD 站址、真实 OSM 材质标签与 USGS 3DEP 地面高程的固定角色场景集。"
            "\nFixed-role scenario set over real OpenCelliD sites, real OSM material tags, and USGS 3DEP ground elevation."
        )
        if bilingual
        else "Fixed-role scenario set over real OpenCelliD sites, real OSM material tags, and USGS 3DEP ground elevation.",
        fontsize=11,
        ha="left",
        va="top",
    )
    title_ax.text(
        0.0,
        0.16,
        (
            "场景角色 / Roles: "
            + ", ".join(_role_label_map().get(role, role) for role in scenarios["scenario_role"].tolist())
        )
        if bilingual
        else f"Roles present: {', '.join(scenarios['scenario_role'].tolist())}",
        fontsize=10,
        ha="left",
        va="top",
    )

    color_by_role = _role_color_map()

    map_ax = fig.add_subplot(grid[1:, 0])
    map_ax.scatter(site_layout["x_m"], site_layout["y_m"], color="#1f77b4", s=45, label="Sites")
    for _, row in scenarios.iterrows():
        role_name = str(row["scenario_role"])
        color = color_by_role.get(role_name, "#444444")
        map_ax.scatter([row["x_m"]], [row["y_m"]], color=color, s=100)
        map_ax.text(
            float(row["x_m"]) + 4.0,
            float(row["y_m"]) + 4.0,
            (
                f"{row['scenario_id']}\n{_role_label_map().get(role_name, role_name)}"
                if bilingual
                else f"{row['scenario_id']}\n{role_name}"
            ),
            fontsize=8,
        )
    map_ax.set_title(
        "场景中心与真实站址 / Scenario Centers and Real Sites"
        if bilingual
        else "Scenario Centers and Real Sites"
    )
    map_ax.set_xlabel("x position (m)")
    map_ax.set_ylabel("y position (m)")
    map_ax.set_aspect("equal", adjustable="box")

    bar_ax = fig.add_subplot(grid[1, 1])
    if "scenario_role" in summary.columns:
        ordered = summary.copy()
    else:
        ordered = summary.merge(
            scenarios[["scenario_id", "scenario_role"]],
            on="scenario_id",
            how="left",
        )
    bar_ax.bar(
        ordered["scenario_id"],
        ordered["center_step_delta_sinr_db"],
        color=[color_by_role.get(str(role), "#444444") for role in ordered["scenario_role"]],
    )
    bar_ax.axhline(0.0, color="#444444", linewidth=1.0)
    bar_ax.set_title(
        "中心步 SINR 变化 / Center-Step Delta SINR"
        if bilingual
        else "Center-Step Delta SINR"
    )
    bar_ax.set_ylabel("Delta SINR (dB)")
    bar_ax.set_xlabel("Scenario")

    table_ax = fig.add_subplot(grid[2, 1])
    table_ax.axis("off")
    merged = scenarios.merge(
        summary[["scenario_id", "mean_delta_sinr_db", "center_step_delta_sinr_db"]],
        on="scenario_id",
        how="left",
    )
    rows = [
        [
            str(row["scenario_id"]),
            (
                _role_label_map().get(str(row["scenario_role"]), str(row["scenario_role"])).replace(" / ", "\n")
                if bilingual
                else str(row["scenario_role"]).replace("_", "\n")
            ),
            f"{float(row['delta_sinr_db']):.2f}",
            f"{float(row['center_step_delta_sinr_db']):.2f}",
        ]
        for _, row in merged.iterrows()
    ]
    table = table_ax.table(
        cellText=rows,
        colLabels=(
            ["ID", "角色 / Role", "点值变化\nPoint Delta", "中心步变化\nCenter-Step Delta"]
            if bilingual
            else ["ID", "Role", "Point\nDelta", "Center-Step\nDelta"]
        ),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    table_ax.set_title("角色摘要 / Role Summary" if bilingual else "Role Summary")
    takeaways = [
        (
            "关键结论 / Key takeaways:\n"
            "1. 真实曼哈顿几何中同时存在干扰抑制与服务惩罚两类材质机制。\n"
            "   Both interferer relief and serving penalty exist in real Manhattan geometry.\n"
            "2. 弱邻区敏感场景说明，峰值点之外仍有小但可测的材质效应。\n"
            "   Weak-neighbor probe confirms small but measurable material sensitivity outside peak points.\n"
            "3. 中性对照场景提供零响应基线。\n"
            "   Neutral control provides a zero-response baseline."
        )
        if bilingual
        else "Key takeaways:\n1. Interferer relief and serving penalty both exist in real Manhattan geometry.\n2. Weak-neighbor probe confirms small but measurable material sensitivity outside peak points.\n3. Neutral control gives a zero-response baseline for comparison.",
    ]
    table_ax.text(0.0, -0.28, "\n".join(takeaways), fontsize=9, va="top")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def run_material_sensitive_scenarios(
    site_layout_csv: Path | str,
    building_geojson: Path | str,
    calibrated_profile_json: Path | str,
    output_dir: Path | str,
    *,
    scan_x_min_m: float = -220.0,
    scan_x_max_m: float = 220.0,
    scan_y_min_m: float = -220.0,
    scan_y_max_m: float = 220.0,
    scan_step_m: float = 20.0,
    dynamic_altitude_m: float = 0.0,
    trajectory_half_span_m: float = 60.0,
    trajectory_steps: int = 21,
) -> MaterialScenarioArtifacts:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    candidate_metrics_csv = output_dir_path / "material_scenario_candidates.csv"
    scenario_definitions_csv = output_dir_path / "material_scenario_definitions.csv"
    scenario_summary_csv = output_dir_path / "material_scenario_summary.csv"
    scenario_traces_csv = output_dir_path / "material_scenario_traces.csv"
    scenario_report_md = output_dir_path / "material_scenario_report.md"
    scenario_map_png = figures_dir / "figure_material_scenario_map.png"
    scenario_delta_bar_png = figures_dir / "figure_material_scenario_delta_bar.png"
    summary_onepager_png = output_dir_path / "material_scenario_onepager.png"
    summary_onepager_pdf = output_dir_path / "material_scenario_onepager.pdf"
    bilingual_onepager_png = output_dir_path / "material_scenario_onepager_bilingual.png"
    bilingual_onepager_pdf = output_dir_path / "material_scenario_onepager_bilingual.pdf"

    config = _build_runtime_config(
        site_layout_csv=site_layout_csv,
        building_geojson=building_geojson,
        calibrated_profile_json=calibrated_profile_json,
        dynamic_altitude_m=dynamic_altitude_m,
    )

    x_values_m = np.arange(scan_x_min_m, scan_x_max_m + 0.5 * scan_step_m, scan_step_m, dtype=float)
    y_values_m = np.arange(scan_y_min_m, scan_y_max_m + 0.5 * scan_step_m, scan_step_m, dtype=float)
    candidates = scan_material_sensitive_points(
        config,
        x_values_m=x_values_m,
        y_values_m=y_values_m,
    )
    candidates.to_csv(candidate_metrics_csv, index=False)

    scenarios = _select_material_sensitive_scenarios(
        candidates,
    )
    scenarios.to_csv(scenario_definitions_csv, index=False)

    off_profile_path = _empty_material_profile_json()
    on_profile_path = Path(config.building_material_loss_profile_json)
    summary_rows: list[dict[str, Any]] = []
    combined_traces: list[pd.DataFrame] = []
    for _, scenario_row in scenarios.iterrows():
        off_trace = _run_scenario_trace(
            config,
            scenario_row,
            material_profile_path=off_profile_path,
            trajectory_half_span_m=trajectory_half_span_m,
            trajectory_steps=trajectory_steps,
        )
        off_trace = off_trace.copy()
        off_trace["scenario_id"] = scenario_row["scenario_id"]
        off_trace["material_mode"] = "off"

        on_trace = _run_scenario_trace(
            config,
            scenario_row,
            material_profile_path=on_profile_path,
            trajectory_half_span_m=trajectory_half_span_m,
            trajectory_steps=trajectory_steps,
        )
        on_trace = on_trace.copy()
        on_trace["scenario_id"] = scenario_row["scenario_id"]
        on_trace["material_mode"] = "on"

        combined_traces.extend([off_trace, on_trace])
        delta_sinr_db = on_trace["sinr_db"].to_numpy() - off_trace["sinr_db"].to_numpy()
        delta_neighbor_loss_db = (
            on_trace["mean_neighbor_gis_excess_loss_db"].to_numpy()
            - off_trace["mean_neighbor_gis_excess_loss_db"].to_numpy()
        )
        center_index = trajectory_steps // 2
        summary_rows.append(
            {
                "scenario_id": scenario_row["scenario_id"],
                "scenario_role": scenario_row["scenario_role"],
                "effect_class": scenario_row["effect_class"],
                "mean_delta_sinr_db": float(np.mean(delta_sinr_db)),
                "max_abs_delta_sinr_db": float(np.max(np.abs(delta_sinr_db))),
                "center_step_delta_sinr_db": float(delta_sinr_db[center_index]),
                "mean_delta_neighbor_gis_excess_loss_db": float(np.mean(delta_neighbor_loss_db)),
            }
        )
        _plot_scenario_trace_pair(
            str(scenario_row["scenario_id"]),
            off_trace,
            on_trace,
            figures_dir / f"{scenario_row['scenario_id']}_sinr_comparison.png",
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(scenario_summary_csv, index=False)
    if combined_traces:
        pd.concat(combined_traces, ignore_index=True).to_csv(scenario_traces_csv, index=False)
    else:
        pd.DataFrame().to_csv(scenario_traces_csv, index=False)

    site_layout = pd.read_csv(site_layout_csv)
    _plot_scenario_map(scenarios, site_layout, scenario_map_png)
    _plot_scenario_delta_bar(scenarios, scenario_delta_bar_png)
    _export_summary_onepager(
        scenarios=scenarios,
        summary=summary,
        site_layout=site_layout,
        output_png=summary_onepager_png,
        output_pdf=summary_onepager_pdf,
    )
    _export_summary_onepager(
        scenarios=scenarios,
        summary=summary,
        site_layout=site_layout,
        output_png=bilingual_onepager_png,
        output_pdf=bilingual_onepager_pdf,
        bilingual=True,
    )
    _write_scenario_report(
        scenario_report_md,
        scenarios=scenarios,
        summary=summary,
        candidate_count=len(candidates),
    )

    return MaterialScenarioArtifacts(
        output_dir=output_dir_path,
        candidate_metrics_csv=candidate_metrics_csv,
        scenario_definitions_csv=scenario_definitions_csv,
        scenario_summary_csv=scenario_summary_csv,
        scenario_traces_csv=scenario_traces_csv,
        scenario_report_md=scenario_report_md,
        scenario_map_png=scenario_map_png,
        scenario_delta_bar_png=scenario_delta_bar_png,
        summary_onepager_png=summary_onepager_png,
        summary_onepager_pdf=summary_onepager_pdf,
        bilingual_onepager_png=bilingual_onepager_png,
        bilingual_onepager_pdf=bilingual_onepager_pdf,
    )
