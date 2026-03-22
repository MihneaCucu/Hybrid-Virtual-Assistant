from __future__ import annotations

import argparse
import json
import math
import os

INPUTS_DEFAULT = "kb/structured/museums.jsonl,kb/structured/osm_places.jsonl"
TRANSIT_DEFAULT = "kb/structured/transit.jsonl"
OUTPUT_DEFAULT = "kb/structured/museum_metro_links.jsonl"


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def load_subway_stops(path: str) -> list[dict]:
    stops = []
    for row in iter_jsonl(path):
        if row.get("record_type") != "stop":
            continue
        if not row.get("is_subway_stop"):
            continue
        lat = to_float(row.get("stop_lat"))
        lon = to_float(row.get("stop_lon"))
        if lat is None or lon is None:
            continue
        stops.append(
            {
                "stop_id": row.get("stop_id", ""),
                "stop_name": row.get("stop_name", ""),
                "lat": lat,
                "lon": lon,
            }
        )
    return stops


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def normalize_place(row: dict) -> dict | None:
    record_type = row.get("record_type", "")
    record_id = row.get("record_id", "")

    if record_type == "museum":
        name = row.get("name", "")
        name_en = row.get("name_en", "")
        aliases = unique_strings(
            [name, name_en] + list(row.get("name_aliases", []) or [])
        )
        lat = to_float(row.get("lat"))
        lon = to_float(row.get("lon"))
        if lat is None or lon is None:
            return None
        return {
            "record_id": record_id,
            "record_type": record_type,
            "name": name or name_en,
            "name_en": name_en,
            "aliases": aliases,
            "address": row.get("address", ""),
            "city": row.get("city", ""),
            "lat": lat,
            "lon": lon,
        }

    if record_type == "osm_place":
        title = row.get("title", "")
        name = row.get("name", "")
        slug_words = str(row.get("slug", "")).replace("_", " ").strip()
        aliases = unique_strings(
            [title, name, slug_words] + list(row.get("name_aliases", []) or [])
        )
        lat = to_float(row.get("lat"))
        lon = to_float(row.get("lon"))
        if lat is None or lon is None:
            return None
        return {
            "record_id": record_id,
            "record_type": record_type,
            "name": title or name,
            "name_en": "",
            "aliases": aliases,
            "address": row.get("address", ""),
            "city": "Bucharest",
            "lat": lat,
            "lon": lon,
        }

    return None


def load_places(paths: list[str]) -> list[dict]:
    places: list[dict] = []
    seen_record_ids: set[str] = set()
    for path in paths:
        if not path:
            continue
        if not os.path.exists(path):
            continue
        for row in iter_jsonl(path):
            place = normalize_place(row)
            if place is None:
                continue
            record_id = str(place.get("record_id", "")).strip()
            if not record_id:
                continue
            if record_id in seen_record_ids:
                continue
            seen_record_ids.add(record_id)
            places.append(place)
    return places


def main() -> None:
    parser = argparse.ArgumentParser(description="Link structured places to nearest subway stations.")
    parser.add_argument(
        "--inputs",
        default=INPUTS_DEFAULT,
        help=f"Comma-separated JSONL inputs with place coordinates (default: {INPUTS_DEFAULT})",
    )
    parser.add_argument("--transit", default=TRANSIT_DEFAULT, help=f"Transit JSONL path (default: {TRANSIT_DEFAULT})")
    parser.add_argument("--out", default=OUTPUT_DEFAULT, help=f"Output JSONL path (default: {OUTPUT_DEFAULT})")
    parser.add_argument("--max-distance-m", type=float, default=5000.0, help="Maximum accepted place->station distance in meters.")
    args = parser.parse_args()

    subway_stops = load_subway_stops(args.transit)
    if not subway_stops:
        raise SystemExit("No subway stops found in transit data. Rebuild transit with subway-stop enrichment first.")

    input_paths = [token.strip() for token in args.inputs.split(",") if token.strip()]
    places = load_places(input_paths)
    if not places:
        raise SystemExit("No valid place records with coordinates found in --inputs.")

    links: list[dict] = []

    for place in places:
        lat = place["lat"]
        lon = place["lon"]

        best_stop = None
        best_distance = float("inf")
        for stop in subway_stops:
            dist = haversine_m(lat, lon, stop["lat"], stop["lon"])
            if dist < best_distance:
                best_distance = dist
                best_stop = stop

        if best_stop is None:
            continue
        if best_distance > args.max_distance_m:
            continue

        place_id = place.get("record_id", "")
        links.append(
            {
                "record_id": f"place_metro_{place_id}",
                "record_type": "place_metro_link",
                "source": "place_metro_linker",
                "place_record_id": place_id,
                "place_record_type": place.get("record_type", ""),
                "place_name": place.get("name", ""),
                "place_name_en": place.get("name_en", ""),
                "place_name_aliases": place.get("aliases", []),
                "place_address": place.get("address", ""),
                "place_city": place.get("city", ""),
                "place_lat": lat,
                "place_lon": lon,
                "metro_stop_id": best_stop["stop_id"],
                "metro_stop_name": best_stop["stop_name"],
                "distance_m": round(best_distance, 1),
            }
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in links:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(links)} place-metro links -> {args.out}")
    print(f"- place inputs: {', '.join(input_paths)}")
    print(f"- places considered: {len(places)}")
    print(f"- subway stops considered: {len(subway_stops)}")


if __name__ == "__main__":
    main()
