#!/usr/bin/env python3
"""
Build an updated PETSS station map from PETSS storm tide text bulletins.

This script downloads one or more PETSS storm tide text files from NOMADS,
extracts station_id / station_name pairs, and writes a Python config snippet
containing:

    DEFAULT_STATION_MAP
    AMBIGUOUS_STATION_MAP
    STATION_METADATA

The intended use is to refresh the station-name lookup used by a separate
PETSS CSV/TWL plotting script.

Examples:
    python build_petss_station_map.py --date 20260622 --cycle 06 --regions nwak
    python build_petss_station_map.py --regions nwak --latest
    python build_petss_station_map.py --input-file petss.t00z.mean.stormtide.nwak.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


NOMADS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/petss/prod"
DEFAULT_REGIONS = ["nwak"]

# Matches PETSS station header lines like:
#   9468756 NOME NORTON SOUND, AK                                                               -10
#   ber0001 POINT BARROW, AK                                                                      4
# The final integer is the first time-series value, not part of the station name.
STATION_HEADER_RE = re.compile(
    r"^\s*"
    r"(?P<station_id>(?:\d{7}|[A-Za-z]{3}\d{4}))"
    r"\s+"
    r"(?P<station_name>.*?[A-Za-z].*?)"
    r"\s+"
    r"(?P<first_value>-?\d+)"
    r"\s*$"
)


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    station_name: str
    source_region: str | None = None
    source_url: str | None = None
    source_file: str | None = None


def normalize_station_key(name: str) -> str:
    """
    Normalize a PETSS station name into a consistent lookup key.

    Examples:
        "NOME NORTON SOUND, AK" -> "nome norton sound"
        "Point-Hope-AK" -> "point hope"
    """
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()

    # Remove trailing state/location tokens common in PETSS station labels.
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
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    return re.sub(r"\s+", " ", text).strip()


def make_aliases(station_name: str) -> set[str]:
    """
    Build conservative station-name aliases for lookup.

    Generated aliases include the full normalized name, the name before the comma,
    and a simplified version with common descriptors removed.
    """
    aliases: set[str] = set()

    original = station_name.strip()
    normalized = normalize_station_key(original)
    if normalized:
        aliases.add(normalized)

    before_comma = normalize_station_key(original.split(",")[0])
    if before_comma:
        aliases.add(before_comma)

    generic_words = [
        "norton sound", "nushagak bay", "inlet", "dock", "entrance", "river",
        "mouth", "channel", "bay", "cove", "island", "creek", "pass", "cape",
    ]
    simplified = normalized
    for word in generic_words:
        simplified = re.sub(rf"\b{re.escape(word)}\b", " ", simplified)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    if simplified and len(simplified) >= 3:
        aliases.add(simplified)

    # A few practical Alaska variants that users are likely to type.
    manual_alias_replacements = {
        "point barrow": ["barrow", "utqiagvik"],
        "tooksook nelson": ["toksook", "toksook bay", "toksook nelson"],
        "st lrnce niyrakpak": ["st lawrence", "st lawrence island", "niyrakpak"],
        "nome norton sound": ["nome"],
        "prudhoe bay": ["prudhoe"],
        "red dog dock": ["red dog"],
        "port clarence": ["clarence"],
        "goodhope": ["goodhope bay", "good hope", "good hope bay"],
    }

    for key, extra_aliases in manual_alias_replacements.items():
        if key in normalized:
            aliases.update(extra_aliases)

    return {normalize_station_key(a) for a in aliases if normalize_station_key(a)}


def parse_station_records_from_text(
    text: str,
    *,
    source_region: str | None = None,
    source_url: str | None = None,
    source_file: str | None = None,
) -> list[StationRecord]:
    """Extract station IDs and station names from one PETSS storm tide text bulletin."""
    records: list[StationRecord] = []
    seen: set[tuple[str, str]] = set()

    for line in text.splitlines():
        match = STATION_HEADER_RE.match(line)
        if not match:
            continue

        station_id = match.group("station_id").strip()
        station_name = re.sub(r"\s+", " ", match.group("station_name").strip())
        key = (station_id, station_name)
        if key in seen:
            continue

        seen.add(key)
        records.append(
            StationRecord(
                station_id=station_id,
                station_name=station_name,
                source_region=source_region,
                source_url=source_url,
                source_file=source_file,
            )
        )

    return records


def petss_url(date_yyyymmdd: str, cycle_hh: str, region: str, statistic: str = "mean") -> str:
    cycle_hh = cycle_hh.lower().replace("z", "").zfill(2)
    region = region.lower()
    statistic = statistic.lower()
    return (
        f"{NOMADS_BASE_URL}/petss.{date_yyyymmdd}/"
        f"petss.t{cycle_hh}z.{statistic}.stormtide.{region}.txt"
    )


def http_get_text(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": "petss-station-map-builder/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def url_exists(url: str, timeout: int = 15) -> bool:
    req = Request(url, method="HEAD", headers={"User-Agent": "petss-station-map-builder/1.0"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return 200 <= response.status < 400
    except HTTPError as exc:
        # Some servers do not like HEAD. Try a tiny GET fallback for 403/405.
        if exc.code not in (403, 405):
            return False
    except URLError:
        return False

    try:
        req = Request(url, headers={"User-Agent": "petss-station-map-builder/1.0"})
        with urlopen(req, timeout=timeout) as response:
            response.read(256)
            return 200 <= response.status < 400
    except Exception:
        return False


def candidate_recent_cycles(*, now_utc: datetime | None = None, lookback_days: int = 3) -> Iterable[tuple[str, str]]:
    """Yield recent date/cycle candidates from newest to oldest."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    current_cycle = (now_utc.hour // 6) * 6
    start = now_utc.replace(hour=current_cycle, minute=0, second=0, microsecond=0)
    for i in range(lookback_days * 4 + 4):
        dt = start - timedelta(hours=6 * i)
        yield dt.strftime("%Y%m%d"), f"{dt.hour:02d}"


def download_station_records(
    *,
    regions: list[str],
    date: str | None,
    cycle: str | None,
    latest: bool,
    statistic: str,
    cache_dir: Path,
    lookback_days: int,
    timeout: int,
) -> tuple[list[StationRecord], list[str]]:
    """Download PETSS text files and parse station records."""
    records: list[StationRecord] = []
    used_sources: list[str] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    if latest:
        candidates = list(candidate_recent_cycles(lookback_days=lookback_days))
    else:
        if not date or not cycle:
            raise ValueError("Provide --date and --cycle, or use --latest.")
        candidates = [(date, cycle.lower().replace("z", "").zfill(2))]

    for region in regions:
        region_found = False
        for date_candidate, cycle_candidate in candidates:
            url = petss_url(date_candidate, cycle_candidate, region, statistic=statistic)
            filename = f"petss.{date_candidate}.t{cycle_candidate}z.{statistic}.stormtide.{region}.txt"
            local_path = cache_dir / filename

            if local_path.exists() and local_path.stat().st_size > 0:
                text = local_path.read_text(encoding="utf-8", errors="replace")
            else:
                if latest and not url_exists(url, timeout=timeout):
                    continue
                try:
                    text = http_get_text(url, timeout=timeout)
                except HTTPError as exc:
                    if latest:
                        continue
                    raise RuntimeError(f"HTTP error downloading {url}: {exc}") from exc
                except URLError as exc:
                    if latest:
                        continue
                    raise RuntimeError(f"URL error downloading {url}: {exc}") from exc
                local_path.write_text(text, encoding="utf-8")

            parsed = parse_station_records_from_text(
                text,
                source_region=region,
                source_url=url,
                source_file=str(local_path),
            )
            if not parsed:
                if latest:
                    continue
                raise RuntimeError(f"No station records parsed from {url}")

            records.extend(parsed)
            used_sources.append(url)
            region_found = True
            break

        if not region_found:
            raise RuntimeError(
                f"No available {statistic} stormtide file found for region={region!r} "
                f"within lookback_days={lookback_days}."
            )

    return records, used_sources


def read_local_station_records(paths: list[Path]) -> tuple[list[StationRecord], list[str]]:
    records: list[StationRecord] = []
    used_sources: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        region = None
        m = re.search(r"\.stormtide\.([a-z0-9]+)\.txt$", path.name, flags=re.IGNORECASE)
        if m:
            region = m.group(1).lower()
        parsed = parse_station_records_from_text(text, source_region=region, source_file=str(path))
        records.extend(parsed)
        used_sources.append(str(path))
    return records, used_sources


def build_maps(records: list[StationRecord]) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict]]:
    """
    Build station lookup maps.

    Returns:
      default_map: unambiguous alias/name -> station_id
      ambiguous_map: alias/name -> sorted list of station_ids
      metadata: station_id -> station metadata
    """
    by_id: dict[str, dict] = {}

    for rec in records:
        item = by_id.setdefault(
            rec.station_id,
            {
                "station_id": rec.station_id,
                "station_name": rec.station_name,
                "aliases": set(),
                "regions": set(),
                "source_files": set(),
                "source_urls": set(),
            },
        )
        item["aliases"].update(make_aliases(rec.station_name))
        if rec.source_region:
            item["regions"].add(rec.source_region)
        if rec.source_file:
            item["source_files"].add(rec.source_file)
        if rec.source_url:
            item["source_urls"].add(rec.source_url)

    alias_to_ids: dict[str, set[str]] = defaultdict(set)
    for station_id, item in by_id.items():
        for alias in item["aliases"]:
            alias_to_ids[alias].add(station_id)

    default_map = {alias: sorted(ids)[0] for alias, ids in alias_to_ids.items() if len(ids) == 1}
    ambiguous_map = {alias: sorted(ids) for alias, ids in alias_to_ids.items() if len(ids) > 1}

    metadata = {}
    for station_id, item in sorted(by_id.items(), key=lambda pair: pair[0]):
        metadata[station_id] = {
            "station_id": station_id,
            "station_name": item["station_name"],
            "aliases": sorted(item["aliases"]),
            "regions": sorted(item["regions"]),
            "source_files": sorted(item["source_files"]),
            "source_urls": sorted(item["source_urls"]),
        }

    return dict(sorted(default_map.items())), dict(sorted(ambiguous_map.items())), metadata


def write_python_config(
    path: Path,
    *,
    default_map: dict[str, str],
    ambiguous_map: dict[str, list[str]],
    metadata: dict[str, dict],
    sources: list[str],
) -> None:
    source_lines = "\n".join(f"    - {src}" for src in sources)
    content = (
        '"""\n'
        "Auto-generated PETSS station map.\n\n"
        "Generated by build_petss_station_map.py.\n\n"
        "Import into your PETSS plotting config with:\n\n"
        "    from petss_station_map import DEFAULT_STATION_MAP, AMBIGUOUS_STATION_MAP\n\n"
        "Sources:\n"
        f"{source_lines}\n"
        '"""\n\n'
        f"DEFAULT_STATION_MAP = {json.dumps(default_map, indent=4, sort_keys=True)}\n\n"
        f"AMBIGUOUS_STATION_MAP = {json.dumps(ambiguous_map, indent=4, sort_keys=True)}\n\n"
        f"STATION_METADATA = {json.dumps(metadata, indent=4, sort_keys=True)}\n"
    )
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, *, default_map, ambiguous_map, metadata, sources) -> None:
    payload = {
        "default_station_map": default_map,
        "ambiguous_station_map": ambiguous_map,
        "station_metadata": metadata,
        "sources": sources,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, metadata: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["station_id", "station_name", "regions", "aliases"])
        writer.writeheader()
        for station_id, item in metadata.items():
            writer.writerow(
                {
                    "station_id": station_id,
                    "station_name": item["station_name"],
                    "regions": ";".join(item["regions"]),
                    "aliases": ";".join(item["aliases"]),
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PETSS DEFAULT_STATION_MAP from PETSS storm tide text output.")

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--latest", action="store_true", help="Find and use the latest available PETSS cycle from NOMADS.")
    source_group.add_argument(
        "--input-file",
        action="append",
        type=Path,
        help="Parse one local PETSS storm tide text file instead of downloading. May be repeated.",
    )

    parser.add_argument("--date", help="PETSS cycle date, YYYYMMDD.")
    parser.add_argument("--cycle", help="PETSS cycle hour: 00, 06, 12, or 18. Accepts either '06' or '06z'.")
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS, help="PETSS regions to parse. Default: nwak.")
    parser.add_argument("--statistic", default="mean", choices=["mean", "min", "max"], help="Stormtide statistic file to use. Default: mean.")
    parser.add_argument("--cache-dir", type=Path, default=Path("petss_station_map_cache"), help="Directory for downloaded storm tide text files.")
    parser.add_argument("--output-py", type=Path, default=Path("petss_station_map.py"), help="Output Python config path.")
    parser.add_argument("--output-json", type=Path, default=Path("petss_station_map.json"), help="Output JSON path, used with --write-json.")
    parser.add_argument("--output-csv", type=Path, default=Path("petss_station_metadata.csv"), help="Output CSV path, used with --write-csv.")
    parser.add_argument("--write-json", action="store_true", help="Also write a JSON version of the station maps.")
    parser.add_argument("--write-csv", action="store_true", help="Also write station metadata as CSV.")
    parser.add_argument("--lookback-days", type=int, default=3, help="When using --latest, how many days back to search. Default: 3.")
    parser.add_argument("--timeout", type=int, default=30, help="Network timeout in seconds. Default: 30.")

    args = parser.parse_args(argv)
    if not args.input_file and not args.latest and not (args.date and args.cycle):
        parser.error("Use --latest, --input-file, or provide both --date and --cycle.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.input_file:
            records, sources = read_local_station_records(args.input_file)
        else:
            records, sources = download_station_records(
                regions=[region.lower() for region in args.regions],
                date=args.date,
                cycle=args.cycle,
                latest=args.latest,
                statistic=args.statistic,
                cache_dir=args.cache_dir,
                lookback_days=args.lookback_days,
                timeout=args.timeout,
            )

        if not records:
            raise RuntimeError("No station records were parsed.")

        default_map, ambiguous_map, metadata = build_maps(records)
        write_python_config(args.output_py, default_map=default_map, ambiguous_map=ambiguous_map, metadata=metadata, sources=sources)

        if args.write_json:
            write_json(args.output_json, default_map=default_map, ambiguous_map=ambiguous_map, metadata=metadata, sources=sources)
        if args.write_csv:
            write_csv(args.output_csv, metadata)

        print(f"Parsed station records: {len(records)}")
        print(f"Unique stations: {len(metadata)}")
        print(f"Unambiguous lookup keys: {len(default_map)}")
        print(f"Ambiguous lookup keys: {len(ambiguous_map)}")
        print(f"Wrote Python config: {args.output_py}")

        if ambiguous_map:
            print("\nAmbiguous names/aliases. Use station_id for these:")
            for alias, ids in ambiguous_map.items():
                print(f"  {alias}: {', '.join(ids)}")

        if args.write_json:
            print(f"Wrote JSON: {args.output_json}")
        if args.write_csv:
            print(f"Wrote CSV: {args.output_csv}")

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
