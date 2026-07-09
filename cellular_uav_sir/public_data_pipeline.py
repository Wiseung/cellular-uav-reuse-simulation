from __future__ import annotations

import csv
import gzip
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .calibration import build_parameter_profile, write_parameter_profile
from .config import SimulationConfig
from .experiments import run_all_experiments
from .parameter_profile import apply_parameter_profile, load_parameter_profile
from .plots import generate_all_plots
from tools.prepare_enhanced_site_layout import prepare_enhanced_site_layout
from tools.prepare_overture_3dep_buildings import prepare_overture_3dep_buildings


@dataclass(frozen=True)
class PublicDataPipelineArtifacts:
    output_root: Path
    enhanced_site_layout_csv: Path
    enhanced_buildings_geojson: Path
    initial_profile_json: Path
    calibrated_profile_json: Path
    baseline_results_dir: Path
    enhanced_results_dir: Path
    comparison_report_md: Path
    comparison_metrics_json: Path
    comparison_metrics_csv: Path


def _save_table(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def _run_and_save_results(config: SimulationConfig) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    bundle = run_all_experiments(config)
    _save_table(bundle.sir_vs_reuse, config.results_dir / "table_1_sir_vs_reuse.csv")
    _save_table(bundle.ase_vs_reuse, config.results_dir / "table_2_ase_vs_reuse.csv")
    _save_table(bundle.sinr_vs_height, config.results_dir / "table_3_sir_vs_height.csv")
    _save_table(bundle.cdf_samples, config.results_dir / "table_4_sir_cdf_samples.csv")
    _save_table(bundle.pathloss_sweep, config.results_dir / "table_5_pathloss_sweep.csv")
    _save_table(bundle.dynamic_summary, config.results_dir / "table_6_dynamic_summary.csv")
    _save_table(bundle.dynamic_trace, config.results_dir / "table_7_dynamic_trace.csv")
    _save_table(bundle.dynamic_site_layout, config.results_dir / "table_8_dynamic_site_layout.csv")
    generate_all_plots(
        config=config,
        sir_vs_reuse=bundle.sir_vs_reuse,
        ase_vs_reuse=bundle.ase_vs_reuse,
        sinr_vs_height=bundle.sinr_vs_height,
        cdf_samples=bundle.cdf_samples,
        pathloss_sweep=bundle.pathloss_sweep,
        dynamic_trace=bundle.dynamic_trace,
        dynamic_site_layout=bundle.dynamic_site_layout,
    )


def _deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _count_csv_rows(path: Path) -> int:
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:  # type: ignore[arg-type]
        if path.suffix.lower() == ".gz":
            return sum(1 for line in handle if line.strip())
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _building_material_stats(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    features = payload.get("features", [])
    total_buildings = len(features)
    facade_counts: dict[str, int] = {}
    roof_counts: dict[str, int] = {}
    height_source_counts: dict[str, int] = {}
    material_annotated_count = 0
    for feature in features:
        properties = feature.get("properties") or {}
        facade_material = str(
            properties.get("facade_material")
            or properties.get("building:material")
            or properties.get("material")
            or ""
        ).strip().lower()
        roof_material = str(
            properties.get("roof_material")
            or properties.get("roof:material")
            or ""
        ).strip().lower()
        height_source = str(properties.get("height_source") or "").strip()
        if facade_material:
            facade_counts[facade_material] = facade_counts.get(facade_material, 0) + 1
        if roof_material:
            roof_counts[roof_material] = roof_counts.get(roof_material, 0) + 1
        if height_source:
            height_source_counts[height_source] = height_source_counts.get(height_source, 0) + 1
        if facade_material or roof_material:
            material_annotated_count += 1
    return {
        "building_count": total_buildings,
        "material_annotated_count": material_annotated_count,
        "facade_material_counts": facade_counts,
        "roof_material_counts": roof_counts,
        "height_source_counts": height_source_counts,
    }


def _reuse_metrics(results_dir: Path, default_reuse_factor: int) -> dict[str, float]:
    dataframe = pd.read_csv(results_dir / "table_1_sir_vs_reuse.csv")
    if "reuse_factor" in dataframe.columns and default_reuse_factor in set(dataframe["reuse_factor"]):
        row = dataframe.loc[dataframe["reuse_factor"] == default_reuse_factor].iloc[0]
    else:
        row = dataframe.iloc[0]
    return {
        "reuse_factor": float(row["reuse_factor"]),
        "median_user_sinr_db": float(row["median_user_sinr_db"]),
        "p05_user_sinr_db": float(row["p05_user_sinr_db"]),
        "mean_signal_power_dbm": float(row["mean_signal_power_dbm"]),
        "mean_interference_power_dbm": float(row["mean_interference_power_dbm"]),
    }


def _dynamic_metrics(results_dir: Path) -> dict[str, float]:
    row = pd.read_csv(results_dir / "table_6_dynamic_summary.csv").iloc[0]
    return {
        "handover_count": float(row["handover_count"]),
        "mean_sinr_db": float(row["mean_sinr_db"]),
        "p05_sinr_db": float(row["p05_sinr_db"]),
        "mean_rate_bphz": float(row["mean_rate_bphz"]),
        "outage_probability_at_10db": float(row["outage_probability_at_10db"]),
        "mean_serving_gis_excess_loss_db": float(row["mean_serving_gis_excess_loss_db"]),
        "mean_cochannel_interferer_count": float(row.get("mean_cochannel_interferer_count", 0.0)),
    }


def _height_metrics(results_dir: Path, target_heights_m: tuple[float, ...] = (0.0, 100.0, 300.0)) -> list[dict[str, float]]:
    dataframe = pd.read_csv(results_dir / "table_3_sir_vs_height.csv")
    rows: list[dict[str, float]] = []
    for target_height_m in target_heights_m:
        closest_index = int((dataframe["height_m"] - target_height_m).abs().idxmin())
        row = dataframe.iloc[closest_index]
        rows.append(
            {
                "height_m": float(row["height_m"]),
                "median_sinr_db": float(row["median_sinr_db"]),
                "p05_sinr_db": float(row["p05_sinr_db"]),
                "coverage_probability_at_10db": float(row["coverage_probability_at_10db"]),
            }
        )
    return rows


def _comparison_rows(
    category: str,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, before_value in before_metrics.items():
        if metric_name not in after_metrics:
            continue
        after_value = after_metrics[metric_name]
        rows.append(
            {
                "category": category,
                "metric": metric_name,
                "before": before_value,
                "after": after_value,
                "delta_after_minus_before": after_value - before_value,
            }
        )
    return rows


def _write_markdown_report(
    report_path: Path,
    raw_source_csv: Path,
    source: str,
    building_source_label: str,
    site_layout_csv: Path,
    building_geojson: Path,
    initial_profile_json: Path,
    calibrated_profile_json: Path,
    baseline_results_dir: Path,
    enhanced_results_dir: Path,
    raw_row_count: int,
    site_row_count: int,
    building_stats: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    before_height_metrics: list[dict[str, float]],
    after_height_metrics: list[dict[str, float]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_comparisons: dict[str, list[dict[str, Any]]] = {}
    for row in comparison_rows:
        grouped_comparisons.setdefault(row["category"], []).append(row)

    lines = [
        "# Public Data Gain Report",
        "",
        "This report was generated by the end-to-end public-data pipeline.",
        "",
        "## Scenario Definition",
        "",
        "- Before: enhanced site layout only, initial site-derived profile, no building GIS loss.",
        f"- After: enhanced site layout + {building_source_label} + material-aware excess loss + calibrated profile.",
        "",
        "## Input Summary",
        "",
        f"- Raw source: `{raw_source_csv}` (`{source}`), rows: {raw_row_count}",
        f"- Enhanced site layout: `{site_layout_csv}`, grouped sites: {site_row_count}",
        f"- Enhanced buildings: `{building_geojson}` ({building_source_label}), buildings: {building_stats['building_count']}, material annotated: {building_stats['material_annotated_count']}",
        f"- Initial profile: `{initial_profile_json}`",
        f"- Calibrated profile: `{calibrated_profile_json}`",
        f"- Baseline results: `{baseline_results_dir}`",
        f"- Enhanced results: `{enhanced_results_dir}`",
        "",
    ]

    if building_stats["facade_material_counts"]:
        lines.extend(
            [
                "## Building Material Coverage",
                "",
                "| Material | Count |",
                "| --- | ---: |",
            ]
        )
        for material_name, count in sorted(
            building_stats["facade_material_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"| facade:{material_name} | {count} |")
        for material_name, count in sorted(
            building_stats["roof_material_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            lines.append(f"| roof:{material_name} | {count} |")
        lines.append("")

    for category in ("reuse", "dynamic"):
        rows = grouped_comparisons.get(category, [])
        if not rows:
            continue
        lines.extend(
            [
                f"## {category.title()} Comparison",
                "",
                "| Metric | Before | After | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['metric']} | {row['before']:.4f} | {row['after']:.4f} | {row['delta_after_minus_before']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Height Comparison",
            "",
            "| Height (m) | Before Median SINR (dB) | After Median SINR (dB) | Delta | Before P05 SINR (dB) | After P05 SINR (dB) | Delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for before_row, after_row in zip(before_height_metrics, after_height_metrics, strict=False):
        lines.append(
            "| "
            f"{before_row['height_m']:.0f} | "
            f"{before_row['median_sinr_db']:.4f} | "
            f"{after_row['median_sinr_db']:.4f} | "
            f"{after_row['median_sinr_db'] - before_row['median_sinr_db']:.4f} | "
            f"{before_row['p05_sinr_db']:.4f} | "
            f"{after_row['p05_sinr_db']:.4f} | "
            f"{after_row['p05_sinr_db'] - before_row['p05_sinr_db']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Material-aware excess loss only affects the penetration branch of the GIS blockage model; diffraction remains geometry driven.",
            "- The calibrated profile is bootstrapped from the baseline dynamic trace, so the first-run site-only simulation is part of the data loop.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_public_data_pipeline(
    raw_source_csv: Path | str,
    source: str,
    overture_geojson: Path | str | None,
    output_root: Path | str,
    prepared_building_geojson: Path | str | None = None,
    building_source_label: str = "Overture/3DEP buildings",
    center_latitude_deg: float | None = None,
    center_longitude_deg: float | None = None,
    radius_km: float | None = None,
    limit_sites: int | None = None,
    grouping_decimals: int = 4,
    include_ground_elevation: bool = False,
    default_profile_json: Path | str | None = None,
    simulation_overrides: dict[str, Any] | None = None,
) -> PublicDataPipelineArtifacts:
    output_root_path = Path(output_root)
    inputs_dir = output_root_path / "inputs"
    profiles_dir = output_root_path / "profiles"
    results_dir = output_root_path / "results"
    report_dir = output_root_path / "report"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_source_csv_path = Path(raw_source_csv)
    overture_geojson_path = None if overture_geojson is None else Path(overture_geojson)
    enhanced_site_layout_csv = inputs_dir / "enhanced_site_layout.csv"
    enhanced_buildings_geojson = inputs_dir / "enhanced_buildings.geojson"
    initial_profile_json = profiles_dir / "initial_site_profile.json"
    calibrated_profile_json = profiles_dir / "calibrated_profile.json"
    baseline_results_dir = results_dir / "baseline"
    enhanced_results_dir = results_dir / "enhanced"
    comparison_report_md = report_dir / "public_data_gain_report.md"
    comparison_metrics_json = report_dir / "comparison_metrics.json"
    comparison_metrics_csv = report_dir / "comparison_metrics.csv"

    prepare_enhanced_site_layout(
        input_csv=raw_source_csv_path,
        source=source,
        output_csv=enhanced_site_layout_csv,
        center_latitude_deg=center_latitude_deg,
        center_longitude_deg=center_longitude_deg,
        radius_km=radius_km,
        limit_sites=limit_sites,
        grouping_decimals=grouping_decimals,
        include_ground_elevation=include_ground_elevation,
    )
    if prepared_building_geojson is not None:
        shutil.copyfile(prepared_building_geojson, enhanced_buildings_geojson)
    else:
        if overture_geojson_path is None:
            raise ValueError("Provide either overture_geojson or prepared_building_geojson.")
        prepare_overture_3dep_buildings(
            overture_geojson=overture_geojson_path,
            output_geojson=enhanced_buildings_geojson,
            site_layout_csv=enhanced_site_layout_csv,
            include_ground_elevation=include_ground_elevation,
        )

    initial_profile = build_parameter_profile(site_layout_csv=enhanced_site_layout_csv)
    write_parameter_profile(initial_profile, initial_profile_json)

    config_overrides = dict(simulation_overrides or {})
    base_config = apply_parameter_profile(
        SimulationConfig(),
        initial_profile_json,
    )
    base_config = replace(
        base_config,
        site_layout_csv=enhanced_site_layout_csv,
        dynamic_site_layout_csv=enhanced_site_layout_csv,
        building_footprint_geojson=None,
        parameter_profile_json=initial_profile_json,
        results_dir=baseline_results_dir,
        **config_overrides,
    )
    _run_and_save_results(base_config)

    calibrated_sections = build_parameter_profile(
        site_layout_csv=enhanced_site_layout_csv,
        dynamic_trace_csv=baseline_results_dir / "table_7_dynamic_trace.csv",
    )
    if default_profile_json is None:
        merged_profile = calibrated_sections
    else:
        merged_profile = _deep_merge_dicts(
            load_parameter_profile(default_profile_json),
            calibrated_sections,
        )
    write_parameter_profile(merged_profile, calibrated_profile_json)

    enhanced_config = apply_parameter_profile(
        SimulationConfig(),
        calibrated_profile_json,
    )
    enhanced_config = replace(
        enhanced_config,
        site_layout_csv=enhanced_site_layout_csv,
        dynamic_site_layout_csv=enhanced_site_layout_csv,
        building_footprint_geojson=enhanced_buildings_geojson,
        building_material_loss_profile_json=(
            enhanced_config.building_material_loss_profile_json
        ),
        parameter_profile_json=calibrated_profile_json,
        results_dir=enhanced_results_dir,
        **config_overrides,
    )
    _run_and_save_results(enhanced_config)

    building_stats = _building_material_stats(enhanced_buildings_geojson)
    before_reuse_metrics = _reuse_metrics(baseline_results_dir, base_config.default_reuse_factor)
    after_reuse_metrics = _reuse_metrics(enhanced_results_dir, enhanced_config.default_reuse_factor)
    before_dynamic_metrics = _dynamic_metrics(baseline_results_dir)
    after_dynamic_metrics = _dynamic_metrics(enhanced_results_dir)
    before_height_metrics = _height_metrics(baseline_results_dir)
    after_height_metrics = _height_metrics(enhanced_results_dir)

    comparison_rows = (
        _comparison_rows("reuse", before_reuse_metrics, after_reuse_metrics)
        + _comparison_rows("dynamic", before_dynamic_metrics, after_dynamic_metrics)
    )
    comparison_dataframe = pd.DataFrame(comparison_rows)
    comparison_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_dataframe.to_csv(comparison_metrics_csv, index=False)
    comparison_metrics_json.write_text(
        json.dumps(
            {
                "reuse": comparison_rows[: len(before_reuse_metrics)],
                "dynamic": comparison_rows[len(before_reuse_metrics) :],
                "height_before": before_height_metrics,
                "height_after": after_height_metrics,
                "building_stats": building_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown_report(
        report_path=comparison_report_md,
        raw_source_csv=raw_source_csv_path,
        source=source,
        building_source_label=building_source_label,
        site_layout_csv=enhanced_site_layout_csv,
        building_geojson=enhanced_buildings_geojson,
        initial_profile_json=initial_profile_json,
        calibrated_profile_json=calibrated_profile_json,
        baseline_results_dir=baseline_results_dir,
        enhanced_results_dir=enhanced_results_dir,
        raw_row_count=_count_csv_rows(raw_source_csv_path),
        site_row_count=_count_csv_rows(enhanced_site_layout_csv),
        building_stats=building_stats,
        comparison_rows=comparison_rows,
        before_height_metrics=before_height_metrics,
        after_height_metrics=after_height_metrics,
    )

    return PublicDataPipelineArtifacts(
        output_root=output_root_path,
        enhanced_site_layout_csv=enhanced_site_layout_csv,
        enhanced_buildings_geojson=enhanced_buildings_geojson,
        initial_profile_json=initial_profile_json,
        calibrated_profile_json=calibrated_profile_json,
        baseline_results_dir=baseline_results_dir,
        enhanced_results_dir=enhanced_results_dir,
        comparison_report_md=comparison_report_md,
        comparison_metrics_json=comparison_metrics_json,
        comparison_metrics_csv=comparison_metrics_csv,
    )
