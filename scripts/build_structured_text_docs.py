from __future__ import annotations

import argparse
import json
import os
import re

STRUCTURED_DEFAULT = "kb/structured"
OUTPUT_DIR_DEFAULT = "kb/clean"
DOC_PREFIX = "structured_"


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
        "--include-types",
        default="system_summary,agency,route,museum,restaurant,museum_metro_link,place_metro_link,osm_place",
        help="Comma-separated record types to convert (default excludes stop-level records).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    include_types = {token.strip() for token in args.include_types.split(",") if token.strip()}

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
    for path in input_files:
        for record in iter_jsonl(path):
            if include_types and record.get("record_type", "") not in include_types:
                continue
            source = record.get("source", "unknown")
            per_source[source] = per_source.get(source, 0) + 1
            rid = record.get("record_id", f"record_{written+1}")
            doc_id = f"{DOC_PREFIX}{sanitize_slug(rid)}"
            text = text_for_record(record)
            write_doc(args.out_dir, doc_id, text)
            written += 1

    print(f"Wrote {written} structured QA docs -> {args.out_dir}")
    print(f"Included record types: {', '.join(sorted(include_types))}")
    for source, count in sorted(per_source.items()):
        print(f"- {source}: {count}")


if __name__ == "__main__":
    main()
