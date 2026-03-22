from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

SOURCE_REGISTRY_DEFAULT = "data/kb_sources_bucharest.json"
OUTPUT_DEFAULT = "kb/structured/osm_places.jsonl"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "HybridVirtualAssistantQA/1.0 (student project)"


def read_source_registry(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_targets(registry: dict) -> list[dict]:
    targets = registry.get("wikipedia_targets", [])
    result: list[dict] = []
    for row in targets:
        title = str(row.get("title", "")).strip()
        slug = str(row.get("slug", "")).strip()
        if title and slug:
            result.append({"title": title, "slug": slug})
    return result


def fetch_nominatim(query: str, timeout: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "limit": "3",
            "addressdetails": "1",
            "namedetails": "1",
            "accept-language": "en",
        }
    )
    url = f"{NOMINATIM_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read().decode("utf-8")
    payload = json.loads(data)
    if not isinstance(payload, list):
        return []
    return payload


def build_query_variants(target: dict, city: str) -> list[str]:
    title = str(target.get("title", "")).strip()
    slug_words = str(target.get("slug", "")).replace("_", " ").strip()
    seeds = [title, slug_words]
    variants: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        if not seed:
            continue
        query = f"{seed}, {city}".strip()
        key = query.lower()
        if key not in seen:
            variants.append(query)
            seen.add(key)
    return variants


def choose_best_result(results: list[dict]) -> dict | None:
    if not results:
        return None

    def score(row: dict) -> tuple[float, int]:
        importance = float(row.get("importance") or 0.0)
        cls = str(row.get("category", "") or row.get("class", "")).lower()
        typ = str(row.get("type", "")).lower()
        bonus = 0
        if cls in {"tourism", "historic", "amenity", "railway", "building", "leisure"}:
            bonus += 2
        if typ in {"museum", "attraction", "monument", "station", "train_station", "subway_entrance"}:
            bonus += 2
        display_name = str(row.get("display_name", "")).lower()
        if "bucharest" in display_name or "bucurești" in display_name or "bucuresti" in display_name:
            bonus += 2
        return (importance, bonus)

    return max(results, key=score)


def format_address(address: dict) -> str:
    road = (
        address.get("road")
        or address.get("pedestrian")
        or address.get("footway")
        or address.get("square")
        or address.get("residential")
    )
    house_number = address.get("house_number")
    city = address.get("city") or address.get("town") or address.get("municipality") or address.get("county")

    first = ""
    if road and house_number:
        first = f"{road} {house_number}"
    elif road:
        first = str(road)

    if first and city:
        return f"{first}, {city}"
    if first:
        return first
    if city:
        return str(city)
    return ""


def parse_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def to_record(target: dict, result: dict) -> dict:
    address_obj = result.get("address") if isinstance(result.get("address"), dict) else {}
    display_name = str(result.get("display_name", "")).strip()
    concise_address = format_address(address_obj)
    address = concise_address or display_name
    title = target["title"]
    slug = target["slug"]
    name = str(result.get("name", "")).strip()
    aliases = [title, slug.replace("_", " ")]
    if name and name not in aliases:
        aliases.append(name)

    return {
        "record_id": f"osm_place_{slug}",
        "record_type": "osm_place",
        "source": "osm_nominatim",
        "title": title,
        "slug": slug,
        "name": name or title,
        "name_aliases": aliases,
        "address": address,
        "display_name": display_name,
        "lat": parse_float(result.get("lat")),
        "lon": parse_float(result.get("lon")),
        "osm_id": str(result.get("osm_id", "")),
        "osm_type": str(result.get("osm_type", "")),
        "class": str(result.get("category", "") or result.get("class", "")),
        "type": str(result.get("type", "")),
        "importance": float(result.get("importance") or 0.0),
    }


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch place addresses from OpenStreetMap Nominatim.")
    parser.add_argument("--source-registry", default=SOURCE_REGISTRY_DEFAULT, help=f"Path to source registry JSON (default: {SOURCE_REGISTRY_DEFAULT})")
    parser.add_argument("--out", default=OUTPUT_DEFAULT, help=f"Output JSONL path (default: {OUTPUT_DEFAULT})")
    parser.add_argument("--city", default="Bucharest, Romania", help="City/country suffix for geocoding queries.")
    parser.add_argument("--timeout", type=int, default=25, help="HTTP timeout in seconds.")
    parser.add_argument("--delay-sec", type=float, default=1.1, help="Delay between API requests to respect rate limits.")
    parser.add_argument("--max-targets", type=int, default=0, help="Max targets to process (0 = all).")
    args = parser.parse_args()

    registry = read_source_registry(args.source_registry)
    targets = load_targets(registry)
    if args.max_targets > 0:
        targets = targets[: args.max_targets]
    if not targets:
        raise SystemExit("No targets found in source registry.")

    output: list[dict] = []
    failures = 0
    for i, target in enumerate(targets, start=1):
        query_variants = build_query_variants(target, args.city)
        print(f"[{i:02d}/{len(targets)}] {target['title']}")
        try:
            best = None
            for query in query_variants:
                results = fetch_nominatim(query, timeout=args.timeout)
                best = choose_best_result(results)
                if best is not None:
                    break
            if best is None:
                failures += 1
                print("  - no result")
            else:
                record = to_record(target, best)
                output.append(record)
                print(f"  - ok: {record['address']}")
        except Exception as exc:
            failures += 1
            print(f"  - fail: {exc}")

        if i < len(targets):
            time.sleep(max(args.delay_sec, 0.0))

    write_jsonl(args.out, output)
    print(f"\nWrote {len(output)} records -> {args.out}")
    print(f"Failed targets: {failures}")


if __name__ == "__main__":
    main()
