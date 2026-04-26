from __future__ import annotations

import argparse
import json
import os
import re

STRUCTURED_DEFAULT = "kb/structured"
OUTPUT_DIR_DEFAULT = "kb/clean"
DOC_PREFIX = "structured_"
PROFILE_INCLUDE_TYPES = {
    "full": "system_summary,agency,route,museum,restaurant,coffee_shop,museum_metro_link,place_metro_link,osm_place",
    "final_demo": "system_summary,agency,route,museum,museum_metro_link,place_metro_link,osm_place",
}


def sanitize_slug(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "item"


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_allowlist(path: str) -> set[str]:
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {str(item).strip() for item in payload if str(item).strip()}
    if isinstance(payload, dict):
        values = payload.get("record_ids", [])
        if isinstance(values, list):
            return {str(item).strip() for item in values if str(item).strip()}
    raise ValueError(f"Unsupported allowlist format in {path}; expected a list or object with record_ids.")


def record_matches_allowlist(record: dict, allowlist: set[str]) -> bool:
    if not allowlist:
        return True
    candidates = {
        str(record.get("record_id", "")).strip(),
        str(record.get("place_record_id", "")).strip(),
        str(record.get("slug", "")).strip(),
        str(record.get("name", "")).strip(),
        str(record.get("place_name", "")).strip(),
        str(record.get("title", "")).strip(),
    }
    return any(candidate and candidate in allowlist for candidate in candidates)


def should_include_record(
    record: dict,
    profile: str,
    restaurant_allowlist: set[str],
    coffee_shop_allowlist: set[str],
) -> bool:
    record_type = record.get("record_type", "")
    if record_type == "restaurant":
        return bool(restaurant_allowlist) and record_matches_allowlist(record, restaurant_allowlist)
    if record_type == "coffee_shop":
        return bool(coffee_shop_allowlist) and record_matches_allowlist(record, coffee_shop_allowlist)
    if record_type == "place_metro_link":
        place_type = str(record.get("place_record_type", "")).strip()
        if place_type == "restaurant":
            return bool(restaurant_allowlist) and record_matches_allowlist(record, restaurant_allowlist)
        if place_type == "coffee_shop":
            return bool(coffee_shop_allowlist) and record_matches_allowlist(record, coffee_shop_allowlist)
        if profile == "final_demo" and place_type not in {"museum", "osm_place"}:
            return False
    return True


def text_for_record(record: dict) -> str:
    record_type = record.get("record_type", "")

    if record_type == "system_summary":
        return (
            f"Transit system summary for {record.get('city', 'the city')}. "
            f"The feed includes {record.get('route_count', 0)} routes and "
            f"{record.get('stop_count', 0)} stops."
        )

    if record_type == "agency":
        return (
            f"Transit agency {record.get('agency_name', 'unknown')} operates in the city. "
            f"Timezone is {record.get('agency_timezone', 'unknown')}. "
            f"Official website: {record.get('agency_url', 'not provided')}."
        )

    if record_type == "route":
        short_name = record.get("route_short_name", "")
        long_name = record.get("route_long_name", "")
        route_type = record.get("route_type_label", "unknown")
        desc = record.get("route_desc", "")
        return (
            f"Transit route {short_name} is a {route_type} line in Bucharest-Ilfov. "
            f"Long name: {long_name}. Description: {desc}."
        )

    if record_type == "stop":
        return (
            f"Transit stop {record.get('stop_name', 'unknown')} has stop id {record.get('stop_id', '')}. "
            f"Coordinates: latitude {record.get('stop_lat', '')}, longitude {record.get('stop_lon', '')}. "
            f"Zone id: {record.get('zone_id', 'not specified')}."
        )

    if record_type == "museum":
        aliases = record.get("name_aliases", []) or []
        aliases_text = f" Also known as: {', '.join(aliases)}." if aliases else ""
        name_en = record.get("name_en", "")
        name_en_text = f" English name: {name_en}." if name_en else ""
        coordinates_text = ""
        if record.get("lat") is not None and record.get("lon") is not None:
            coordinates_text = f" Coordinates: latitude {record.get('lat')}, longitude {record.get('lon')}."
        return (
            f"{record.get('name', 'This museum')} is a museum in {record.get('city', 'Romania')}. "
            f"County: {record.get('county', 'unknown')}. "
            f"Address: {record.get('address', 'not provided')}. "
            f"Category: {record.get('category', 'not specified')}. "
            f"Description: {record.get('description', 'not provided')}. "
            f"Website: {record.get('website', 'not provided')}. "
            f"Phone: {record.get('phone', 'not provided')}."
            f"{name_en_text}{aliases_text}{coordinates_text}"
        )

    if record_type == "restaurant":
        aliases = record.get("name_aliases", []) or []
        aliases_text = f" Also known as: {', '.join(aliases)}." if aliases else ""
        name_en = record.get("name_en", "")
        name_en_text = f" English name: {name_en}." if name_en else ""
        return (
            f"{record.get('name', 'This restaurant')} is a restaurant in {record.get('city', 'Bucharest')}. "
            f"Address: {record.get('address', 'not provided')}. "
            f"Cuisine: {record.get('cuisine', 'not specified')}. "
            f"Website: {record.get('website', 'not provided')}. "
            f"Phone: {record.get('phone', 'not provided')}. "
            f"Coordinates: latitude {record.get('lat', '')}, longitude {record.get('lon', '')}."
            f"{name_en_text}{aliases_text}"
        )

    if record_type == "coffee_shop":
        aliases = record.get("name_aliases", []) or []
        aliases_text = f" Also known as: {', '.join(aliases)}." if aliases else ""
        name_en = record.get("name_en", "")
        name_en_text = f" English name: {name_en}." if name_en else ""
        return (
            f"{record.get('name', 'This coffee shop')} is a coffee shop in {record.get('city', 'Bucharest')}. "
            f"Address: {record.get('address', 'not provided')}. "
            f"Cuisine or specialty tags: {record.get('cuisine', 'not specified')}. "
            f"Website: {record.get('website', 'not provided')}. "
            f"Phone: {record.get('phone', 'not provided')}. "
            f"Coordinates: latitude {record.get('lat', '')}, longitude {record.get('lon', '')}."
            f"{name_en_text}{aliases_text}"
        )

    if record_type in {"museum_metro_link", "place_metro_link"}:
        name = record.get("place_name") or record.get("museum_name") or "This place"
        address = record.get("place_address") or record.get("museum_address") or "unknown address"
        city = record.get("place_city") or record.get("museum_city") or "Bucharest"
        aliases = record.get("place_name_aliases", []) or record.get("museum_name_aliases", []) or []
        aliases_text = f" Also known as: {', '.join(aliases)}." if aliases else ""
        english_name = record.get("place_name_en") or record.get("museum_name_en")
        english_text = f" English name: {english_name}." if english_name else ""
        return (
            f"{name} is at {address} in {city}. "
            f"The nearest metro station is {record.get('metro_stop_name', 'unknown')} "
            f"(about {record.get('distance_m', 'unknown')} meters away)."
            f"{english_text}{aliases_text}"
        )

    if record_type == "osm_place":
        aliases = record.get("name_aliases", []) or []
        aliases_text = f" Also known as: {', '.join(aliases)}." if aliases else ""
        return (
            f"{record.get('title', 'This place')} is a place in Bucharest. "
            f"Address: {record.get('address', 'not provided')}. "
            f"Full location: {record.get('display_name', 'not provided')}. "
            f"Coordinates: latitude {record.get('lat', '')}, longitude {record.get('lon', '')}. "
            f"OpenStreetMap class/type: {record.get('class', 'unknown')}/{record.get('type', 'unknown')}."
            f"{aliases_text}"
        )

    return json.dumps(record, ensure_ascii=False)


def write_doc(output_dir: str, doc_id: str, text: str) -> str:
    filename = f"{doc_id}.txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QA text docs from structured JSONL records.")
    parser.add_argument("--structured-dir", default=STRUCTURED_DEFAULT, help=f"Input directory for structured JSONL files (default: {STRUCTURED_DEFAULT})")
    parser.add_argument("--out-dir", default=OUTPUT_DIR_DEFAULT, help=f"Output directory for .txt docs (default: {OUTPUT_DIR_DEFAULT})")
    parser.add_argument("--clear-existing", action="store_true", help=f"Delete existing {DOC_PREFIX}*.txt docs before generating.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_INCLUDE_TYPES),
        default="full",
        help="Predefined structured-doc profile. Use final_demo for a smaller report/demo KB.",
    )
    parser.add_argument(
        "--include-types",
        default="",
        help="Comma-separated record types to convert. Overrides --profile when provided.",
    )
    parser.add_argument(
        "--restaurant-allowlist",
        default="",
        help="Optional JSON allowlist for restaurant records to include.",
    )
    parser.add_argument(
        "--coffee-shop-allowlist",
        default="",
        help="Optional JSON allowlist for coffee-shop records to include.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    include_text = args.include_types or PROFILE_INCLUDE_TYPES[args.profile]
    include_types = {token.strip() for token in include_text.split(",") if token.strip()}
    restaurant_allowlist = load_allowlist(args.restaurant_allowlist)
    coffee_shop_allowlist = load_allowlist(args.coffee_shop_allowlist)

    if args.clear_existing:
        for fname in os.listdir(args.out_dir):
            if fname.startswith(DOC_PREFIX) and fname.endswith(".txt"):
                os.remove(os.path.join(args.out_dir, fname))
        print(f"Cleared existing {DOC_PREFIX}*.txt docs from {args.out_dir}")

    input_files = sorted(
        os.path.join(args.structured_dir, fname)
        for fname in os.listdir(args.structured_dir)
        if fname.endswith(".jsonl")
    ) if os.path.isdir(args.structured_dir) else []

    if not input_files:
        raise SystemExit(f"No JSONL files found in {args.structured_dir}.")

    written = 0
    per_source: dict[str, int] = {}
    skipped_by_allowlist = 0
    for path in input_files:
        for record in iter_jsonl(path):
            record_type = record.get("record_type", "")
            if include_types and record_type not in include_types:
                continue
            if not should_include_record(record, args.profile, restaurant_allowlist, coffee_shop_allowlist):
                skipped_by_allowlist += 1
                continue
            source = record.get("source", "unknown")
            per_source[source] = per_source.get(source, 0) + 1
            rid = record.get("record_id", f"record_{written+1}")
            doc_id = f"{DOC_PREFIX}{sanitize_slug(rid)}"
            text = text_for_record(record)
            write_doc(args.out_dir, doc_id, text)
            written += 1

    print(f"Wrote {written} structured QA docs -> {args.out_dir}")
    print(f"Profile: {args.profile}")
    print(f"Included record types: {', '.join(sorted(include_types))}")
    if skipped_by_allowlist:
        print(f"Skipped by allowlist: {skipped_by_allowlist}")
    for source, count in sorted(per_source.items()):
        print(f"- {source}: {count}")


if __name__ == "__main__":
    main()
