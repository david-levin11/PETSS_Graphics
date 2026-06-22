"""
Download and parse PETSS storm tide station text files from NOMADS.

Example:
    python petss_stormtide_downloader.py --region nwak --location "Nome" --outdir petss_data
    python petss_stormtide_downloader.py --date 20260621 --cycle 00 --station-id 9468756

Outputs:
    - raw downloaded text files under <outdir>/raw/YYYYMMDD/HH/
    - combined long-form CSV under <outdir>/parsed/
    - optional selected-station CSV and PNG plot
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import requests

BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/petss/prod"
STATS = ("min", "mean", "max")
CYCLES = ("00", "06", "12", "18")

# Station header examples:
#  ber0001 POINT BARROW, AK                                                                      4
#  9497645 PRUDHOE BAY, AK                                                                       3
HEADER_RE = re.compile(
    r"^\s*(?P<station_id>(?:[A-Za-z]{3}\d{4}|\d{7}))\s+"
    r"(?P<station_name>.*?)\s+(?P<first_value>-?\d+)\s*$"
)


@dataclass(frozen=True)
class PetssCycle:
    date: str  # YYYYMMDD
    cycle: str  # HH, one of 00/06/12/18

    @property
    def run_time(self) -> pd.Timestamp:
        return pd.Timestamp(
            datetime.strptime(f"{self.date}{self.cycle}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
        )

    @property
    def url_dir(self) -> str:
        return f"{BASE_URL}/petss.{self.date}"


def file_name(stat: str, cycle: str, region: str = "nwak") -> str:
    return f"petss.t{cycle}z.{stat}.stormtide.{region}.txt"


def file_url(cycle: PetssCycle, stat: str, region: str = "nwak") -> str:
    return f"{cycle.url_dir}/{file_name(stat, cycle.cycle, region)}"


def url_exists(url: str, timeout: int = 15) -> bool:
    """Return True if a NOMADS file appears to exist."""
    try:
        # HEAD is usually fine, but GET fallback is more reliable for simple web indexes.
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return True
        if r.status_code in (403, 405):
            r = requests.get(url, timeout=timeout, stream=True)
            r.close()
            return r.status_code == 200
        return False
    except requests.RequestException:
        return False


def candidate_cycles(now_utc: datetime | None = None, days_back: int = 3) -> Iterable[PetssCycle]:
    """Yield recent cycles newest-first, accounting for possible production latency."""
    now_utc = now_utc or datetime.now(timezone.utc)
    today = now_utc.date()

    candidates: list[PetssCycle] = []
    for d in range(days_back + 1):
        date = today - timedelta(days=d)
        yyyymmdd = date.strftime("%Y%m%d")
        for cyc in CYCLES:
            run_dt = datetime.strptime(f"{yyyymmdd}{cyc}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
            if run_dt <= now_utc:
                candidates.append(PetssCycle(yyyymmdd, cyc))

    return sorted(candidates, key=lambda c: c.run_time, reverse=True)


def find_latest_available_cycle(region: str = "nwak", required_stats: Iterable[str] = STATS) -> PetssCycle:
    """Find the newest cycle where all required stat files are present."""
    for cyc in candidate_cycles():
        urls = [file_url(cyc, stat, region) for stat in required_stats]
        if all(url_exists(u) for u in urls):
            return cyc
    raise RuntimeError("No complete PETSS cycle found in the recent NOMADS directories.")


def download_text(url: str, dest: Path, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return dest

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_text(r.text, encoding="utf-8")
    return dest


def parse_petss_text(
    text: str,
    run_time: pd.Timestamp,
    statistic: str,
    source_file: str | None = None,
) -> pd.DataFrame:
    """
    Parse a PETSS station text file into tidy hourly rows.

    Values in the file are tenths of feet, so both raw tenths and feet are returned.
    The first value is printed on the station header line, followed by wrapped rows.
    """
    stations: list[tuple[dict[str, str], list[int]]] = []
    current_station: dict[str, str] | None = None
    values: list[int] = []

    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if m:
            if current_station is not None:
                stations.append((current_station, values))
            current_station = {
                "station_id": m.group("station_id").strip(),
                "station_name": m.group("station_name").strip(),
            }
            values = [int(m.group("first_value"))]
            continue

        if current_station is not None:
            # Handles normal spacing and compressed negatives like -55-270.
            values.extend(int(x) for x in re.findall(r"-?\d+", line))

    if current_station is not None:
        stations.append((current_station, values))

    rows = []
    for station, vals in stations:
        for forecast_hour, value_tenths_ft in enumerate(vals, start=1):
            rows.append(
                {
                    **station,
                    "run_time": run_time,
                    "valid_time": run_time + pd.Timedelta(hours=forecast_hour),
                    "forecast_hour": forecast_hour,
                    "statistic": statistic,
                    "storm_tide_tenths_ft": value_tenths_ft,
                    "storm_tide_ft": value_tenths_ft / 10.0,
                    "source_file": source_file,
                }
            )

    return pd.DataFrame(rows)


def download_and_parse_cycle(
    cycle: PetssCycle,
    region: str = "nwak",
    outdir: Path = Path("petss_data"),
    overwrite: bool = False,
) -> pd.DataFrame:
    frames = []
    for stat in STATS:
        fname = file_name(stat, cycle.cycle, region)
        url = file_url(cycle, stat, region)
        raw_path = outdir / "raw" / cycle.date / cycle.cycle / fname
        download_text(url, raw_path, overwrite=overwrite)
        text = raw_path.read_text(encoding="utf-8", errors="replace")
        frames.append(parse_petss_text(text, cycle.run_time, stat, source_file=str(raw_path)))

    df = pd.concat(frames, ignore_index=True)

    parsed_dir = outdir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = parsed_dir / f"petss_{cycle.date}_t{cycle.cycle}z_stormtide_{region}_long.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote combined CSV: {csv_path}")
    return df


def select_station(df: pd.DataFrame, station_id: str | None = None, location: str | None = None) -> pd.DataFrame:
    if station_id:
        out = df[df["station_id"].str.lower() == station_id.lower()].copy()
    elif location:
        out = df[df["station_name"].str.contains(location, case=False, na=False)].copy()
    else:
        return pd.DataFrame()

    if out.empty:
        available = df[["station_id", "station_name"]].drop_duplicates().sort_values("station_name")
        print("No matching station found. First 25 available stations:")
        print(available.head(25).to_string(index=False))
    return out


def station_wide(df_station: pd.DataFrame) -> pd.DataFrame:
    """Convert selected station to one row per valid_time with min/mean/max columns."""
    id_cols = ["station_id", "station_name", "run_time", "valid_time", "forecast_hour"]
    return (
        df_station.pivot_table(
            index=id_cols,
            columns="statistic",
            values="storm_tide_ft",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values("valid_time")
    )


def plot_station(wide: pd.DataFrame, out_png: Path) -> None:
    if wide.empty:
        return

    title_station = f"{wide['station_name'].iloc[0]} ({wide['station_id'].iloc[0]})"
    run_time = pd.Timestamp(wide["run_time"].iloc[0]).strftime("%Y-%m-%d %HZ")

    fig, ax = plt.subplots(figsize=(11, 5))
    if {"min", "max"}.issubset(wide.columns):
        ax.fill_between(wide["valid_time"], wide["min"], wide["max"], alpha=0.25, label="Min–Max")
    if "mean" in wide.columns:
        ax.plot(wide["valid_time"], wide["mean"], linewidth=2, label="Mean")

    ax.axhline(0, linewidth=0.8)
    ax.set_title(f"PETSS Storm Tide: {title_station}\nRun: {run_time}")
    ax.set_ylabel("Storm tide (ft)")
    ax.set_xlabel("Valid time (UTC)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and parse PETSS storm tide text files.")
    parser.add_argument("--date", help="Cycle date in YYYYMMDD. If omitted, newest complete cycle is used.")
    parser.add_argument("--cycle", choices=CYCLES, help="Cycle hour: 00, 06, 12, or 18.")
    parser.add_argument("--region", default="nwak", help="PETSS region suffix, e.g., nwak, goak, goam, east, west.")
    parser.add_argument("--outdir", type=Path, default=Path("petss_data"), help="Output directory.")
    parser.add_argument("--station-id", help="Exact station ID to subset and plot, e.g., 9468756.")
    parser.add_argument("--location", help="Case-insensitive station-name search, e.g., Nome.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they already exist.")
    args = parser.parse_args()

    if args.date and args.cycle:
        cycle = PetssCycle(args.date, args.cycle)
    elif args.date or args.cycle:
        parser.error("Use both --date and --cycle together, or omit both for latest available.")
    else:
        cycle = find_latest_available_cycle(region=args.region)
        print(f"Using latest complete cycle: {cycle.date} t{cycle.cycle}z")

    df = download_and_parse_cycle(cycle, region=args.region, outdir=args.outdir, overwrite=args.overwrite)
    print(f"Parsed {df['station_id'].nunique()} stations and {len(df):,} rows.")

    selected = select_station(df, station_id=args.station_id, location=args.location)
    if not selected.empty:
        wide = station_wide(selected)
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", wide["station_name"].iloc[0]).strip("_")
        selected_csv = args.outdir / "parsed" / f"petss_{cycle.date}_t{cycle.cycle}z_{safe_name}_{args.region}_wide.csv"
        wide.to_csv(selected_csv, index=False)
        print(f"Wrote selected-station CSV: {selected_csv}")

        out_png = args.outdir / "plots" / f"petss_{cycle.date}_t{cycle.cycle}z_{safe_name}_{args.region}.png"
        plot_station(wide, out_png)

    return 0


if __name__ == "__main__":
    sys.exit(main())
