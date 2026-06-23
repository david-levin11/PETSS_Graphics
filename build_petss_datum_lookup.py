#!/usr/bin/env python3
"""
Build a PETSS station datum lookup table using NOAA CO-OPS/NOS MDAPI.

Formula:
    value_ft_MHHW = value_ft_MLLW - (MHHW_ft - MLLW_ft)

Typical output field to use in the TWL plotter:
    mhhw_minus_mllw_ft

Example:
    python build_petss_datum_lookup.py --station-map petss_station_map.py --output-csv petss_station_datums.csv

For PETSS-only IDs like ber####, optional cautious name matching:
    python build_petss_datum_lookup.py --station-map petss_station_map.py --allow-name-match
"""

from __future__ import annotations

import argparse
import csv
import difflib
import importlib.util
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MDAPI_BASE_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
USER_AGENT = "petss-datum-lookup-builder/1.0"

COMMON_DATUM_KEYS = [
    "MHHW", "MHW", "MTL", "MSL", "MLW", "MLLW", "NAVD88", "STND",
    "GT", "MN", "DHQ", "DLQ", "HWI", "LWI", "LAT", "HAT",
]


@dataclass
class StationInput:
    station_id: str
    station_name: str
    aliases: list[str]


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[, \-]+ak$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[, \-]+alaska$", "", text, flags=re.IGNORECASE)
    text = text.replace("&", " and ")
    text = re.sub(r"[#'`’]", "", text)
    text = re.sub(r"[/_.:,;()\[\]{}\-]+", " ", text)

    replacements = {
        r"\bst\b": "st",
        r"\bpt\b": "point",
        r"\bent\b": "entrance",
        r"\brv\b": "river",
        r"\br\b": "river",
        r"\bsd\b": "sound",
        r"\bi\b": "island",
        r"\bis\b": "island",
        r"\bn\b": "north",
        r"\bs\b": "south",
        r"\be\b": "east",
        r"\bw\b": "west",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    return re.sub(r"\s+", " ", text).strip()


def request_json(url: str, *, params: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    if params:
        url = f"{url}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def get_station_datums(station_id: str, *, units: str = "english", timeout: int = 30) -> dict[str, Any]:
    url = f"{MDAPI_BASE_URL}/stations/{station_id}/datums.json"
    return request_json(url, params={"units": units}, timeout=timeout)


def get_station_details(station_id: str, *, timeout: int = 30) -> dict[str, Any] | None:
    try:
        return request_json(f"{MDAPI_BASE_URL}/stations/{station_id}.json", timeout=timeout)
    except Exception:
        return None


def get_all_datum_stations(*, timeout: int = 60) -> list[dict[str, Any]]:
    payload = request_json(
        f"{MDAPI_BASE_URL}/stations.json",
        params={"type": "datums", "units": "english"},
        timeout=timeout,
    )
    for key in ("stations", "Stations", "stationList", "StationList", "Station"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Could not find station list in MDAPI response keys: {list(payload.keys())}")


def datum_list_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("datums", "datumList", "DatumList", "Datum"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"No datum list found in response keys: {list(payload.keys())}")


def parse_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def datums_to_dict(payload: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for item in datum_list_from_payload(payload):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().upper()
        if name:
            out[name] = parse_float(item.get("value"))
    return out


def load_station_inputs_from_py(path: Path) -> list[StationInput]:
    spec = importlib.util.spec_from_file_location("petss_station_map_dynamic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import station map file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metadata = getattr(module, "STATION_METADATA", None)
    default_map = getattr(module, "DEFAULT_STATION_MAP", {})

    if isinstance(metadata, dict) and metadata:
        stations = []
        for station_id, item in sorted(metadata.items()):
            if not isinstance(item, dict):
                continue
            stations.append(
                StationInput(
                    station_id=str(station_id),
                    station_name=str(item.get("station_name") or station_id),
                    aliases=[str(a) for a in item.get("aliases", []) if a],
                )
            )
        return stations

    reverse: dict[str, set[str]] = {}
    for alias, sid in default_map.items():
        reverse.setdefault(str(sid), set()).add(str(alias))
    return [
        StationInput(station_id=sid, station_name=sorted(aliases)[0], aliases=sorted(aliases))
        for sid, aliases in sorted(reverse.items())
    ]


def load_station_inputs_from_json(path: Path) -> list[StationInput]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("station_metadata", payload.get("STATION_METADATA", {}))
    if not isinstance(metadata, dict):
        raise ValueError(f"No station_metadata found in {path}")
    return [
        StationInput(
            station_id=str(station_id),
            station_name=str(item.get("station_name") or station_id),
            aliases=[str(a) for a in item.get("aliases", []) if a] if isinstance(item, dict) else [],
        )
        for station_id, item in sorted(metadata.items())
        if isinstance(item, dict)
    ]


def load_station_inputs_from_csv(path: Path) -> list[StationInput]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        if "station_id" not in cols:
            raise ValueError("CSV must include station_id column.")
        name_col = cols.get("station_name")
        for row in reader:
            sid = str(row[cols["station_id"]]).strip()
            name = str(row[name_col]).strip() if name_col else sid
            rows.append(StationInput(station_id=sid, station_name=name, aliases=[normalize_name(name)]))
    return rows


def load_station_inputs(path: Path) -> list[StationInput]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return load_station_inputs_from_py(path)
    if suffix == ".json":
        return load_station_inputs_from_json(path)
    if suffix == ".csv":
        return load_station_inputs_from_csv(path)
    raise ValueError("Station map must be .py, .json, or .csv")


def build_name_index(coops_stations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index = {}
    for stn in coops_stations:
        name = stn.get("name") or stn.get("stationName") or ""
        sid = stn.get("id") or stn.get("station_id") or ""
        if name and sid:
            index[normalize_name(str(name))] = stn
    return index


def match_coops_by_name(
    station: StationInput,
    name_index: dict[str, dict[str, Any]],
    *,
    cutoff: float = 0.88,
) -> tuple[str | None, str | None, float | None, str]:
    candidates = [station.station_name] + list(station.aliases)
    norm_candidates = [normalize_name(c) for c in candidates if normalize_name(c)]

    for cand in norm_candidates:
        if cand in name_index:
            stn = name_index[cand]
            return str(stn.get("id")), str(stn.get("name")), 1.0, "name_exact"

    all_names = list(name_index.keys())
    best_name = None
    best_score = 0.0
    for cand in norm_candidates:
        matches = difflib.get_close_matches(cand, all_names, n=1, cutoff=cutoff)
        if matches:
            score = difflib.SequenceMatcher(None, cand, matches[0]).ratio()
            if score > best_score:
                best_score = score
                best_name = matches[0]

    if best_name is None:
        return None, None, None, "none"

    stn = name_index[best_name]
    return str(stn.get("id")), str(stn.get("name")), best_score, "name_fuzzy"


def build_ok_row(
    station: StationInput,
    coops_station_id: str,
    coops_station_name: str | None,
    payload: dict[str, Any],
    *,
    match_method: str,
    match_score: float | None,
) -> dict[str, Any]:
    datums = datums_to_dict(payload)
    mllw = datums.get("MLLW")
    mhhw = datums.get("MHHW")
    diff = (mhhw - mllw) if (mhhw is not None and mllw is not None) else None

    row = {
        "station_id": station.station_id,
        "station_name": station.station_name,
        "coops_station_id": coops_station_id,
        "coops_station_name": coops_station_name or payload.get("name") or "",
        "match_method": match_method,
        "match_score": match_score,
        "datum_epoch": payload.get("epoch", ""),
        "datum_units": payload.get("units", ""),
        "orthometric_datum": payload.get("OrthometricDatum", ""),
        "mhhw_minus_mllw_ft": diff,
        "mllw_to_mhhw_offset_ft": -diff if diff is not None else None,
        "status": "ok" if diff is not None else "missing_required_datums",
        "message": "" if diff is not None else "Could not compute MHHW - MLLW from returned datums.",
    }
    for key in COMMON_DATUM_KEYS:
        row[f"{key.lower()}_ft"] = datums.get(key)
    return row


def missing_row(station: StationInput, status: str, message: str) -> dict[str, Any]:
    row = {
        "station_id": station.station_id,
        "station_name": station.station_name,
        "coops_station_id": "",
        "coops_station_name": "",
        "match_method": "",
        "match_score": None,
        "datum_epoch": "",
        "datum_units": "",
        "orthometric_datum": "",
        "mhhw_minus_mllw_ft": None,
        "mllw_to_mhhw_offset_ft": None,
        "status": status,
        "message": message,
    }
    for key in COMMON_DATUM_KEYS:
        row[f"{key.lower()}_ft"] = None
    return row


def build_lookup(
    stations: list[StationInput],
    *,
    units: str,
    allow_name_match: bool,
    match_cutoff: float,
    timeout: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    rows = []
    name_index = None
    if allow_name_match:
        print("Downloading CO-OPS station list for optional name matching...")
        name_index = build_name_index(get_all_datum_stations(timeout=timeout))
        print(f"Loaded {len(name_index)} CO-OPS datum station names.")

    for i, station in enumerate(stations, start=1):
        print(f"[{i}/{len(stations)}] {station.station_id} {station.station_name}")

        try:
            payload = get_station_datums(station.station_id, units=units, timeout=timeout)
            details = get_station_details(station.station_id, timeout=timeout)
            coops_name = None
            if details:
                coops_name = details.get("name") or details.get("stationName")
            rows.append(
                build_ok_row(
                    station,
                    station.station_id,
                    coops_name,
                    payload,
                    match_method="station_id",
                    match_score=None,
                )
            )
        except Exception as direct_exc:
            if not allow_name_match or name_index is None:
                rows.append(
                    missing_row(
                        station,
                        "not_found",
                        f"Direct CO-OPS datum lookup failed: {direct_exc}",
                    )
                )
            else:
                match_id, match_name, score, method = match_coops_by_name(
                    station,
                    name_index,
                    cutoff=match_cutoff,
                )
                if not match_id:
                    rows.append(
                        missing_row(
                            station,
                            "not_found",
                            f"Direct lookup failed and no name match found. Direct error: {direct_exc}",
                        )
                    )
                else:
                    try:
                        payload = get_station_datums(match_id, units=units, timeout=timeout)
                        rows.append(
                            build_ok_row(
                                station,
                                match_id,
                                match_name,
                                payload,
                                match_method=method,
                                match_score=score,
                            )
                        )
                    except Exception as match_exc:
                        rows.append(
                            missing_row(
                                station,
                                "name_match_failed",
                                f"Matched to {match_id} {match_name}, but datum lookup failed: {match_exc}. Direct error: {direct_exc}",
                            )
                        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "station_id", "station_name", "coops_station_id", "coops_station_name",
        "match_method", "match_score", "datum_epoch", "datum_units",
        "orthometric_datum", "mllw_ft", "mhhw_ft", "mhhw_minus_mllw_ft",
        "mllw_to_mhhw_offset_ft", "mhw_ft", "mtl_ft", "msl_ft", "mlw_ft",
        "navd88_ft", "stnd_ft", "gt_ft", "mn_ft", "dhq_ft", "dlq_ft",
        "hwi_ft", "lwi_ft", "lat_ft", "hat_ft", "status", "message",
    ]
    extras = sorted({k for row in rows for k in row.keys()} - set(fieldnames))
    fieldnames += extras
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PETSS station datum lookup table from NOAA CO-OPS/NOS MDAPI."
    )
    parser.add_argument("--station-map", type=Path, default=Path("petss_station_map.py"))
    parser.add_argument("--output-csv", type=Path, default=Path("petss_station_datums.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("petss_station_datums.json"))
    parser.add_argument("--write-json", action="store_true")
    parser.add_argument("--units", choices=["english", "metric"], default="english")
    parser.add_argument("--allow-name-match", action="store_true")
    parser.add_argument("--match-cutoff", type=float, default=0.88)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stations = load_station_inputs(args.station_map)
        if not stations:
            raise RuntimeError(f"No stations loaded from {args.station_map}")

        rows = build_lookup(
            stations,
            units=args.units,
            allow_name_match=args.allow_name_match,
            match_cutoff=args.match_cutoff,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )

        write_csv(args.output_csv, rows)
        print(f"\nWrote datum lookup CSV: {args.output_csv}")

        if args.write_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Wrote datum lookup JSON: {args.output_json}")

        ok = sum(1 for row in rows if row.get("status") == "ok")
        print(f"\nSummary: {ok} stations with MHHW/MLLW offsets, {len(rows) - ok} missing or incomplete.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
