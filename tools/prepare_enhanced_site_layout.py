from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

EARTH_RADIUS_M = 6371000.0
USGS_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
OPENCELLID_EXPORT_COLUMNS = (
    "radio",
    "mcc",
    "net",
    "area",
    "cell",
    "unit",
    "lon",
    "lat",
    "range",
    "samples",
    "changeable",
    "created",
    "updated",
    "averageSignal",
)

SOURCE_FIELD_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "opencellid": {
        "latdec": ("lat", "latitude", "latdec"),
        "londec": ("lon", "longitude", "londec"),
        "radio": ("radio",),
        "mcc": ("mcc",),
        "mnc": ("net", "mnc"),
        "area_code": ("area", "lac", "tac"),
        "cell_id": ("cell", "cid"),
        "sample_count": ("samples",),
        "average_signal_dbm": ("averageSignal", "average_signal"),
        "source_unit": ("unit",),
        "operator": ("operator", "network"),
        "antenna_azimuth_deg": ("azimuth", "antenna_azimuth_deg"),
        "mechanical_downtilt_deg": ("mechanical_downtilt_deg", "mechanical_tilt_deg"),
        "electrical_downtilt_deg": ("electrical_downtilt_deg", "electrical_tilt_deg"),
        "tx_power_dbm": ("tx_power_dbm",),
        "antenna_height_m": ("antenna_height_m", "height_m"),
        "antenna_peak_gain_db": ("antenna_peak_gain_db",),
    },
    "cellmapper": {
        "latdec": ("lat", "latitude", "Latitude"),
        "londec": ("lon", "longitude", "Longitude"),
        "radio": ("radio", "Radio", "RAT"),
        "mcc": ("mcc", "MCC"),
        "mnc": ("mnc", "MNC"),
        "area_code": ("tac", "TAC", "lac", "LAC"),
        "cell_id": ("cid", "CID", "cell", "CellID"),
        "pci": ("pci", "PCI"),
        "arfcn": ("arfcn", "ARFCN", "earfcn", "EARFCN", "nrarfcn", "NRARFCN"),
        "sample_count": ("samples", "Samples"),
        "average_signal_dbm": ("rsrp", "RSRP", "averageSignal", "average_signal"),
        "operator": ("operator", "Operator", "network", "Network"),
        "antenna_azimuth_deg": ("azimuth", "Azimuth", "bearing"),
        "mechanical_downtilt_deg": ("mechanical_downtilt_deg", "mechanical_tilt_deg"),
        "electrical_downtilt_deg": ("electrical_downtilt_deg", "electrical_tilt_deg"),
        "tx_power_dbm": ("tx_power_dbm",),
        "antenna_height_m": ("antenna_height_m", "height_m"),
        "antenna_peak_gain_db": ("antenna_peak_gain_db",),
        "site_hint": ("site", "Site", "enodeb_id", "gnodeb_id"),
    },
}


def _get_first_value(row: dict[str, str], field_names: tuple[str, ...]) -> str | None:
    row_index = {key.lower(): value for key, value in row.items()}
    for field_name in field_names:
        value = row_index.get(field_name.lower())
        if value not in (None, ""):
            return value
    return None


def _parse_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_public_cell_row(source: str, row: dict[str, str]) -> dict[str, Any] | None:
    aliases = SOURCE_FIELD_ALIASES[source]
    latitude_deg = _parse_optional_float(_get_first_value(row, aliases["latdec"]))
    longitude_deg = _parse_optional_float(_get_first_value(row, aliases["londec"]))
    if latitude_deg is None or longitude_deg is None:
        return None

    normalized_row: dict[str, Any] = {
        "source": source,
        "latdec": latitude_deg,
        "londec": longitude_deg,
        "radio": _get_first_value(row, aliases.get("radio", ())),
        "mcc": _get_first_value(row, aliases.get("mcc", ())),
        "mnc": _get_first_value(row, aliases.get("mnc", ())),
        "area_code": _get_first_value(row, aliases.get("area_code", ())),
        "cell_id": _get_first_value(row, aliases.get("cell_id", ())),
        "pci": _get_first_value(row, aliases.get("pci", ())),
        "arfcn": _get_first_value(row, aliases.get("arfcn", ())),
        "operator": _get_first_value(row, aliases.get("operator", ())),
        "source_unit": _get_first_value(row, aliases.get("source_unit", ())),
        "site_hint": _get_first_value(row, aliases.get("site_hint", ())),
        "sample_count": _parse_optional_float(_get_first_value(row, aliases.get("sample_count", ()))),
        "average_signal_dbm": _parse_optional_float(
            _get_first_value(row, aliases.get("average_signal_dbm", ()))
        ),
        "antenna_azimuth_deg": _parse_optional_float(
            _get_first_value(row, aliases.get("antenna_azimuth_deg", ()))
        ),
        "mechanical_downtilt_deg": _parse_optional_float(
            _get_first_value(row, aliases.get("mechanical_downtilt_deg", ()))
        ),
        "electrical_downtilt_deg": _parse_optional_float(
            _get_first_value(row, aliases.get("electrical_downtilt_deg", ()))
        ),
        "tx_power_dbm": _parse_optional_float(
            _get_first_value(row, aliases.get("tx_power_dbm", ()))
        ),
        "antenna_height_m": _parse_optional_float(
            _get_first_value(row, aliases.get("antenna_height_m", ()))
        ),
        "antenna_peak_gain_db": _parse_optional_float(
            _get_first_value(row, aliases.get("antenna_peak_gain_db", ()))
        ),
    }
    return normalized_row


def _open_text_auto(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _iter_normalized_rows(input_csv: Path, source: str) -> list[dict[str, Any]]:
    with _open_text_auto(input_csv) as handle:
        if source == "opencellid":
            sample = handle.read(4096)
            handle.seek(0)
            header_tokens = next(csv.reader([sample.splitlines()[0]]), [])
            has_header = {"lat", "lon", "latitude", "longitude", "latdec", "londec"}.intersection(
                token.strip().lower() for token in header_tokens
            )
            if has_header:
                reader = csv.DictReader(handle)
                rows = [_normalize_public_cell_row(source, row) for row in reader]
            else:
                reader = csv.reader(handle)
                rows = []
                for raw_row in reader:
                    if not raw_row:
                        continue
                    row_dict = {
                        column_name: raw_row[index] if index < len(raw_row) else ""
                        for index, column_name in enumerate(OPENCELLID_EXPORT_COLUMNS)
                    }
                    rows.append(_normalize_public_cell_row(source, row_dict))
        else:
            reader = csv.DictReader(handle)
            rows = [_normalize_public_cell_row(source, row) for row in reader]
    return [row for row in rows if row is not None]


def _haversine_distance_km(
    latitude_a_deg: float,
    longitude_a_deg: float,
    latitude_b_deg: float,
    longitude_b_deg: float,
) -> float:
    latitude_a_rad = math.radians(latitude_a_deg)
    latitude_b_rad = math.radians(latitude_b_deg)
    delta_latitude_rad = latitude_b_rad - latitude_a_rad
    delta_longitude_rad = math.radians(longitude_b_deg - longitude_a_deg)
    arc = (
        math.sin(delta_latitude_rad / 2.0) ** 2
        + math.cos(latitude_a_rad)
        * math.cos(latitude_b_rad)
        * math.sin(delta_longitude_rad / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(max(arc, 0.0))) / 1000.0


def _lonlat_to_local_xy_m(
    latitude_deg: float,
    longitude_deg: float,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
) -> tuple[float, float]:
    origin_latitude_rad = math.radians(origin_latitude_deg)
    delta_latitude_rad = math.radians(latitude_deg - origin_latitude_deg)
    delta_longitude_rad = math.radians(longitude_deg - origin_longitude_deg)
    x_m = EARTH_RADIUS_M * delta_longitude_rad * math.cos(origin_latitude_rad)
    y_m = EARTH_RADIUS_M * delta_latitude_rad
    return x_m, y_m


def _row_group_key(row: dict[str, Any], grouping_decimals: int) -> tuple[str, ...]:
    if row.get("site_hint"):
        return (str(row["source"]), str(row["site_hint"]))
    operator = str(row.get("operator") or "")
    mcc = str(row.get("mcc") or "")
    mnc = str(row.get("mnc") or "")
    latitude_key = f"{float(row['latdec']):.{grouping_decimals}f}"
    longitude_key = f"{float(row['londec']):.{grouping_decimals}f}"
    return (str(row["source"]), operator, mcc, mnc, latitude_key, longitude_key)


def _sorted_join(values: set[str]) -> str:
    return "|".join(sorted(value for value in values if value))


def query_3dep_ground_elevation_m(latitude_deg: float, longitude_deg: float) -> float:
    query = urlencode(
        {
            "x": longitude_deg,
            "y": latitude_deg,
            "units": "Meters",
            "wkid": 4326,
            "includeDate": "false",
        }
    )
    with urlopen(f"{USGS_EPQS_URL}?{query}") as response:
        payload = json.load(response)

    value = payload.get("value")
    if value is None:
        value = payload.get("elevation")
    if value is None and isinstance(payload.get("value"), dict):
        value = payload["value"].get("value")
    if value is None:
        raise ValueError("USGS EPQS response did not include an elevation value.")
    return float(value)


def prepare_enhanced_site_layout(
    input_csv: Path | str,
    source: str,
    output_csv: Path | str,
    center_latitude_deg: float | None = None,
    center_longitude_deg: float | None = None,
    radius_km: float | None = None,
    limit_sites: int | None = None,
    grouping_decimals: int = 4,
    include_ground_elevation: bool = False,
) -> list[dict[str, Any]]:
    normalized_rows = _iter_normalized_rows(Path(input_csv), source)
    if center_latitude_deg is not None and center_longitude_deg is not None and radius_km is not None:
        normalized_rows = [
            row
            for row in normalized_rows
            if _haversine_distance_km(
                center_latitude_deg,
                center_longitude_deg,
                float(row["latdec"]),
                float(row["londec"]),
            )
            <= radius_km
        ]
    if not normalized_rows:
        raise ValueError("No usable public-cell rows remained after normalization/filtering.")

    grouped_rows: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in normalized_rows:
        grouped_rows.setdefault(_row_group_key(row, grouping_decimals), []).append(row)

    grouped_sites: list[dict[str, Any]] = []
    for site_rows in grouped_rows.values():
        latitude_values = np.array([float(row["latdec"]) for row in site_rows], dtype=float)
        longitude_values = np.array([float(row["londec"]) for row in site_rows], dtype=float)
        sample_count = int(sum(int(row["sample_count"] or 0) for row in site_rows))
        average_signal_values = np.array(
            [row["average_signal_dbm"] for row in site_rows if row["average_signal_dbm"] is not None],
            dtype=float,
        )
        azimuth_values = sorted(
            {
                round(float(row["antenna_azimuth_deg"]) % 360.0, 3)
                for row in site_rows
                if row["antenna_azimuth_deg"] is not None
            }
        )
        grouped_sites.append(
            {
                "source": str(site_rows[0]["source"]),
                "operator": str(site_rows[0].get("operator") or ""),
                "radio": str(site_rows[0].get("radio") or ""),
                "mcc": str(site_rows[0].get("mcc") or ""),
                "mnc": str(site_rows[0].get("mnc") or ""),
                "area_code": str(site_rows[0].get("area_code") or ""),
                "latdec": float(np.mean(latitude_values)),
                "londec": float(np.mean(longitude_values)),
                "cell_count": len({str(row.get("cell_id") or "") for row in site_rows if row.get("cell_id")}),
                "cell_ids": _sorted_join(
                    {str(row.get("cell_id") or "") for row in site_rows if row.get("cell_id")}
                ),
                "pci_list": _sorted_join(
                    {str(row.get("pci") or "") for row in site_rows if row.get("pci")}
                ),
                "arfcn_list": _sorted_join(
                    {str(row.get("arfcn") or "") for row in site_rows if row.get("arfcn")}
                ),
                "source_unit_list": _sorted_join(
                    {str(row.get("source_unit") or "") for row in site_rows if row.get("source_unit")}
                ),
                "sector_azimuths_deg": "|".join(str(value) for value in azimuth_values),
                "antenna_azimuth_deg": azimuth_values[0] if len(azimuth_values) == 1 else "",
                "mechanical_downtilt_deg": "",
                "electrical_downtilt_deg": "",
                "tx_power_dbm": "",
                "antenna_height_m": "",
                "antenna_peak_gain_db": "",
                "sample_count": sample_count,
                "average_signal_dbm": (
                    float(np.mean(average_signal_values)) if average_signal_values.size else ""
                ),
            }
        )
        mechanical_tilts_deg = [
            float(row["mechanical_downtilt_deg"])
            for row in site_rows
            if row["mechanical_downtilt_deg"] is not None
        ]
        if mechanical_tilts_deg:
            grouped_sites[-1]["mechanical_downtilt_deg"] = float(np.median(mechanical_tilts_deg))
        electrical_tilts_deg = [
            float(row["electrical_downtilt_deg"])
            for row in site_rows
            if row["electrical_downtilt_deg"] is not None
        ]
        if electrical_tilts_deg:
            grouped_sites[-1]["electrical_downtilt_deg"] = float(np.median(electrical_tilts_deg))
        tx_power_values_dbm = [
            float(row["tx_power_dbm"])
            for row in site_rows
            if row["tx_power_dbm"] is not None
        ]
        if tx_power_values_dbm:
            grouped_sites[-1]["tx_power_dbm"] = float(np.median(tx_power_values_dbm))
        antenna_height_values_m = [
            float(row["antenna_height_m"])
            for row in site_rows
            if row["antenna_height_m"] is not None
        ]
        if antenna_height_values_m:
            grouped_sites[-1]["antenna_height_m"] = float(np.median(antenna_height_values_m))
        peak_gain_values_db = [
            float(row["antenna_peak_gain_db"])
            for row in site_rows
            if row["antenna_peak_gain_db"] is not None
        ]
        if peak_gain_values_db:
            grouped_sites[-1]["antenna_peak_gain_db"] = float(np.median(peak_gain_values_db))

    if center_latitude_deg is not None and center_longitude_deg is not None:
        grouped_sites.sort(
            key=lambda row: _haversine_distance_km(
                center_latitude_deg,
                center_longitude_deg,
                float(row["latdec"]),
                float(row["londec"]),
            )
        )
    if limit_sites is not None:
        grouped_sites = grouped_sites[:limit_sites]
    if not grouped_sites:
        raise ValueError("No grouped sites remained after limiting.")

    if center_latitude_deg is None or center_longitude_deg is None:
        center_latitude_deg = float(np.mean([float(row["latdec"]) for row in grouped_sites]))
        center_longitude_deg = float(np.mean([float(row["londec"]) for row in grouped_sites]))

    origin_site = min(
        grouped_sites,
        key=lambda row: _haversine_distance_km(
            center_latitude_deg,
            center_longitude_deg,
            float(row["latdec"]),
            float(row["londec"]),
        ),
    )
    origin_latitude_deg = float(origin_site["latdec"])
    origin_longitude_deg = float(origin_site["londec"])

    for site_index, row in enumerate(grouped_sites):
        x_m, y_m = _lonlat_to_local_xy_m(
            latitude_deg=float(row["latdec"]),
            longitude_deg=float(row["londec"]),
            origin_latitude_deg=origin_latitude_deg,
            origin_longitude_deg=origin_longitude_deg,
        )
        row["site_id"] = f"{source}_{site_index:03d}"
        row["x_m"] = round(float(x_m), 3)
        row["y_m"] = round(float(y_m), 3)
        if include_ground_elevation:
            row["ground_elevation_m"] = round(
                query_3dep_ground_elevation_m(float(row["latdec"]), float(row["londec"])),
                3,
            )
        else:
            row["ground_elevation_m"] = ""

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "site_id",
        "x_m",
        "y_m",
        "latdec",
        "londec",
        "ground_elevation_m",
        "source",
        "operator",
        "radio",
        "mcc",
        "mnc",
        "area_code",
        "cell_count",
        "cell_ids",
        "pci_list",
        "arfcn_list",
        "source_unit_list",
        "sector_azimuths_deg",
        "antenna_azimuth_deg",
        "mechanical_downtilt_deg",
        "electrical_downtilt_deg",
        "tx_power_dbm",
        "antenna_height_m",
        "antenna_peak_gain_db",
        "sample_count",
        "average_signal_dbm",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(grouped_sites)
    return grouped_sites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize OpenCelliD/CellMapper public cell exports into an enhanced site-layout CSV."
    )
    parser.add_argument("--source", choices=sorted(SOURCE_FIELD_ALIASES), required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--center-lat", type=float)
    parser.add_argument("--center-lon", type=float)
    parser.add_argument("--radius-km", type=float)
    parser.add_argument("--limit-sites", type=int)
    parser.add_argument("--grouping-decimals", type=int, default=4)
    parser.add_argument(
        "--include-ground-elevation",
        action="store_true",
        help="Query USGS 3DEP EPQS for each grouped site and write ground_elevation_m.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_enhanced_site_layout(
        input_csv=args.input_csv,
        source=args.source,
        output_csv=args.output,
        center_latitude_deg=args.center_lat,
        center_longitude_deg=args.center_lon,
        radius_km=args.radius_km,
        limit_sites=args.limit_sites,
        grouping_decimals=args.grouping_decimals,
        include_ground_elevation=args.include_ground_elevation,
    )
    print(f"Wrote enhanced site layout to {args.output}")


if __name__ == "__main__":
    main()
