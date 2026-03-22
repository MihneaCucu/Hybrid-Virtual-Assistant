from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile

SOURCE_REGISTRY_DEFAULT = "data/kb_sources_bucharest.json"
OUTPUT_DEFAULT = "kb/structured/transit.jsonl"

ROUTE_TYPE_LABELS = {
    "0": "tram",
    "1": "subway",
    "2": "rail",
    "3": "bus",
    "4": "ferry",
    "5": "cable_tram",
    "6": "aerial_lift",
    "7": "funicular",
    "11": "trolleybus",
    "12": "monorail",
}


def read_source_registry(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_structured_source(registry: dict, source_id: str) -> dict | None:
    for source in registry.get("structured_sources", []):
        if source.get("id") == source_id:
            return source
    return None


def discover_gtfs_zip_url(entry_url: str, timeout: int = 20) -> str:
    with urllib.request.urlopen(entry_url, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="ignore")

    links = re.findall(r'href=["\']([^"\']+\.zip[^"\']*)["\']', html, flags=re.IGNORECASE)
    if not links:
        raise RuntimeError(f"Could not find a .zip link on {entry_url}")

    # Prefer region-wide feeds (typically include metro), then "all", then agency-specific zips.
    def rank(link: str) -> tuple[int, int]:
        lower = link.lower()
        if "bucharest-region" in lower or "region" in lower:
            bucket = 0
        elif "all" in lower:
            bucket = 1
        elif "stb" in lower:
            bucket = 3
        else:
            bucket = 2
        return (bucket, len(link))

    prioritized = sorted(links, key=rank)
    return urllib.parse.urljoin(entry_url, prioritized[0])


def download_file(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "HybridVA-QA/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def read_gtfs_table(zf: zipfile.ZipFile, table_name: str) -> list[dict]:
    try:
        with zf.open(table_name) as f:
            text = f.read().decode("utf-8-sig", errors="replace")
    except KeyError:
        return []

    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def iter_gtfs_rows(zf: zipfile.ZipFile, table_name: str):
    try:
        binary = zf.open(table_name)
    except KeyError:
        return
    with binary:
        with io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace", newline="") as text_stream:
            reader = csv.DictReader(text_stream)
            for row in reader:
                yield row


def derive_subway_stop_ids(zf: zipfile.ZipFile, routes: list[dict]) -> set[str]:
    subway_route_ids = {
        str(route.get("route_id", "")).strip()
        for route in routes
        if str(route.get("route_type", "")).strip() == "1"
    }
    if not subway_route_ids:
        return set()

    subway_trip_ids: set[str] = set()
    for row in iter_gtfs_rows(zf, "trips.txt") or []:
        route_id = str(row.get("route_id", "")).strip()
        trip_id = str(row.get("trip_id", "")).strip()
        if route_id in subway_route_ids and trip_id:
            subway_trip_ids.add(trip_id)

    if not subway_trip_ids:
        return set()

    subway_stop_ids: set[str] = set()
    for row in iter_gtfs_rows(zf, "stop_times.txt") or []:
        trip_id = str(row.get("trip_id", "")).strip()
        stop_id = str(row.get("stop_id", "")).strip()
        if trip_id in subway_trip_ids and stop_id:
            subway_stop_ids.add(stop_id)
    return subway_stop_ids


def normalize_gtfs(
    agencies: list[dict],
    routes: list[dict],
    stops: list[dict],
    subway_stop_ids: set[str],
    max_routes: int,
    max_stops: int,
) -> list[dict]:
    records: list[dict] = []

    records.append(
        {
            "record_id": "tpbi_system_summary",
            "record_type": "system_summary",
            "source": "tpbi_gtfs",
            "agency_count": len(agencies),
            "route_count": len(routes),
            "stop_count": len(stops),
            "subway_stop_count": len(subway_stop_ids),
            "city": "Bucharest-Ilfov",
        }
    )

    for idx, agency in enumerate(agencies, start=1):
        agency_id = agency.get("agency_id") or f"agency_{idx}"
        records.append(
            {
                "record_id": f"agency_{agency_id}",
                "record_type": "agency",
                "source": "tpbi_gtfs",
                "agency_id": agency_id,
                "agency_name": agency.get("agency_name", ""),
                "agency_url": agency.get("agency_url", ""),
                "agency_timezone": agency.get("agency_timezone", ""),
                "agency_lang": agency.get("agency_lang", ""),
            }
        )

    sorted_routes = sorted(routes, key=lambda row: (row.get("route_short_name", ""), row.get("route_id", "")))
    route_limit = len(sorted_routes) if max_routes <= 0 else min(max_routes, len(sorted_routes))
    for route in sorted_routes[:route_limit]:
        route_id = route.get("route_id", "")
        route_type = route.get("route_type", "")
        records.append(
            {
                "record_id": f"route_{route_id}",
                "record_type": "route",
                "source": "tpbi_gtfs",
                "route_id": route_id,
                "route_short_name": route.get("route_short_name", ""),
                "route_long_name": route.get("route_long_name", ""),
                "route_desc": route.get("route_desc", ""),
                "route_type": route_type,
                "route_type_label": ROUTE_TYPE_LABELS.get(str(route_type), "unknown"),
                "agency_id": route.get("agency_id", ""),
            }
        )

    sorted_stops = sorted(stops, key=lambda row: (row.get("stop_name", ""), row.get("stop_id", "")))
    stop_limit = len(sorted_stops) if max_stops <= 0 else min(max_stops, len(sorted_stops))
    for stop in sorted_stops[:stop_limit]:
        stop_id = stop.get("stop_id", "")
        records.append(
            {
                "record_id": f"stop_{stop_id}",
                "record_type": "stop",
                "source": "tpbi_gtfs",
                "stop_id": stop_id,
                "stop_name": stop.get("stop_name", ""),
                "stop_desc": stop.get("stop_desc", ""),
                "stop_lat": stop.get("stop_lat", ""),
                "stop_lon": stop.get("stop_lon", ""),
                "zone_id": stop.get("zone_id", ""),
                "parent_station": stop.get("parent_station", ""),
                "is_subway_stop": str(stop_id).strip() in subway_stop_ids,
            }
        )

    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and normalize TPBI GTFS feed.")
    parser.add_argument("--source-registry", default=SOURCE_REGISTRY_DEFAULT, help=f"Path to source registry JSON (default: {SOURCE_REGISTRY_DEFAULT})")
    parser.add_argument("--entry-url", default="", help="GTFS index page URL. If empty, read from source registry.")
    parser.add_argument("--gtfs-url", default="", help="Direct GTFS zip URL.")
    parser.add_argument("--gtfs-zip", default="", help="Local GTFS .zip path (overrides URL download).")
    parser.add_argument("--out", default=OUTPUT_DEFAULT, help=f"Output JSONL path (default: {OUTPUT_DEFAULT})")
    parser.add_argument("--max-routes", type=int, default=300, help="Max number of routes to keep.")
    parser.add_argument("--max-stops", type=int, default=800, help="Max number of stops to keep.")
    parser.add_argument("--timeout", type=int, default=30, help="Network timeout in seconds.")
    args = parser.parse_args()

    gtfs_zip_bytes: bytes

    if args.gtfs_zip:
        with open(args.gtfs_zip, "rb") as f:
            gtfs_zip_bytes = f.read()
        print(f"Loaded local GTFS zip: {args.gtfs_zip}")
    else:
        gtfs_url = args.gtfs_url.strip()
        entry_url = args.entry_url.strip()
        if not gtfs_url:
            if not entry_url:
                registry = read_source_registry(args.source_registry)
                source = find_structured_source(registry, "tpbi_gtfs")
                if source is None:
                    raise RuntimeError("tpbi_gtfs source not found in source registry.")
                entry_url = source.get("entry_url", "").strip()
            if not entry_url:
                raise RuntimeError("Missing GTFS entry URL. Provide --entry-url or source registry value.")
            gtfs_url = discover_gtfs_zip_url(entry_url, timeout=args.timeout)
        print(f"Downloading GTFS zip from: {gtfs_url}")
        gtfs_zip_bytes = download_file(gtfs_url, timeout=args.timeout)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(gtfs_zip_bytes)
        tmp_path = tmp.name

    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            agencies = read_gtfs_table(zf, "agency.txt")
            routes = read_gtfs_table(zf, "routes.txt")
            stops = read_gtfs_table(zf, "stops.txt")
            subway_stop_ids = derive_subway_stop_ids(zf, routes)

        records = normalize_gtfs(
            agencies=agencies,
            routes=routes,
            stops=stops,
            subway_stop_ids=subway_stop_ids,
            max_routes=args.max_routes,
            max_stops=args.max_stops,
        )
        write_jsonl(args.out, records)
    finally:
        os.unlink(tmp_path)

    route_records = sum(1 for row in records if row.get("record_type") == "route")
    stop_records = sum(1 for row in records if row.get("record_type") == "stop")
    subway_stop_records = sum(1 for row in records if row.get("record_type") == "stop" and row.get("is_subway_stop"))
    print(f"Wrote {len(records)} records -> {args.out}")
    print(f"- routes: {route_records}")
    print(f"- stops: {stop_records}")
    print(f"- subway stops: {subway_stop_records}")


if __name__ == "__main__":
    main()
