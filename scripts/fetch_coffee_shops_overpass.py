from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request

OVERPASS_URL_DEFAULT = "https://overpass-api.de/api/interpreter"
OUTPUT_DEFAULT = "kb/structured/coffee_shops.jsonl"


def build_query(city_names: list[str], timeout_s: int) -> str:
    escaped = [re.escape(name.strip()) for name in city_names if name.strip()]
    city_pattern = "|".join(escaped) if escaped else "București|Bucharest"
    return f"""
[out:json][timeout:{int(timeout_s)}];
area["name"~"^({city_pattern})$"]["boundary"="administrative"]->.searchArea;
(
  node(area.searchArea)["amenity"="cafe"]["name"];
  way(area.searchArea)["amenity"="cafe"]["name"];
  relation(area.searchArea)["amenity"="cafe"]["name"];
);
out center tags;
""".strip()


def fetch_overpass_json(overpass_url: str, query: str, timeout_s: int) -> dict:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        overpass_url,
        data=body,
        headers={
            "User-Agent": "HybridVirtualAssistantQA/1.0 (student project)",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        preview = payload[:240].replace("\n", " ")
        raise RuntimeError(f"Overpass returned a non-JSON response: {preview}") from exc


def split_semicolon(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


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


def format_address(tags: dict, default_city: str) -> str:
    full = str(tags.get("addr:full", "")).strip()
    if full:
        return full

    street = str(tags.get("addr:street", "")).strip()
    number = str(tags.get("addr:housenumber", "")).strip()
    city = str(tags.get("addr:city", "")).strip() or default_city

    line = ""
    if street and number:
        line = f"{street} {number}"
    elif street:
        line = street
    elif number:
        line = number

    if line and city:
        return f"{line}, {city}"
    if line:
        return line
    if city:
        return city
    return ""


def get_lat_lon(element: dict) -> tuple[float | None, float | None]:
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    center = element.get("center", {})
    if isinstance(center, dict) and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])
    return None, None


def to_record(element: dict, default_city: str) -> dict | None:
    tags = element.get("tags", {})
    if not isinstance(tags, dict):
        return None

    name = str(tags.get("name", "")).strip()
    if not name:
        return None

    lat, lon = get_lat_lon(element)
    if lat is None or lon is None:
        return None

    name_en = str(tags.get("name:en", "")).strip()
    aliases = unique_strings(
        split_semicolon(str(tags.get("alt_name", "")))
        + split_semicolon(str(tags.get("official_name", "")))
        + split_semicolon(str(tags.get("short_name", "")))
        + ([name_en] if name_en else [])
    )
    cuisine = ", ".join(split_semicolon(str(tags.get("cuisine", ""))))
    phone = str(tags.get("phone", "") or tags.get("contact:phone", "")).strip()
    website = str(tags.get("website", "") or tags.get("contact:website", "")).strip()
    address = format_address(tags, default_city=default_city)
    has_specific_address = bool(address and address.lower() != default_city.lower())
    has_useful_metadata = bool(cuisine or phone or website or has_specific_address)
    if not has_useful_metadata:
        return None

    osm_type = str(element.get("type", "")).strip()
    osm_id = str(element.get("id", "")).strip()
    if not osm_type or not osm_id:
        return None

    return {
        "record_id": f"coffee_shop_osm_{osm_type}_{osm_id}",
        "record_type": "coffee_shop",
        "source": "osm_overpass_coffee_shops",
        "name": name,
        "name_en": name_en,
        "name_aliases": aliases,
        "cuisine": cuisine,
        "address": address,
        "city": str(tags.get("addr:city", "")).strip() or default_city,
        "website": website,
        "phone": phone,
        "lat": round(lat, 7),
        "lon": round(lon, 7),
        "osm_type": osm_type,
        "osm_id": osm_id,
    }


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Bucharest coffee shops from OSM Overpass.")
    parser.add_argument("--overpass-url", default=OVERPASS_URL_DEFAULT, help=f"Overpass endpoint (default: {OVERPASS_URL_DEFAULT})")
    parser.add_argument("--city-names", default="București,Bucharest", help="Comma-separated city names used in Overpass area lookup.")
    parser.add_argument("--default-city", default="Bucharest", help="Default city label for records without addr:city.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP/Overpass timeout in seconds.")
    parser.add_argument("--max-records", type=int, default=0, help="Max output records (0 = all).")
    parser.add_argument("--out", default=OUTPUT_DEFAULT, help=f"Output JSONL path (default: {OUTPUT_DEFAULT})")
    args = parser.parse_args()

    city_names = [part.strip() for part in args.city_names.split(",") if part.strip()]
    query = build_query(city_names=city_names, timeout_s=args.timeout)
    payload = fetch_overpass_json(args.overpass_url, query, timeout_s=args.timeout)

    elements = payload.get("elements", []) if isinstance(payload, dict) else []
    rows: list[dict] = []
    seen: set[str] = set()
    for element in elements:
        if not isinstance(element, dict):
            continue
        record = to_record(element, default_city=args.default_city)
        if record is None:
            continue
        rid = record["record_id"]
        if rid in seen:
            continue
        seen.add(rid)
        rows.append(record)

    rows.sort(key=lambda row: row.get("name", "").lower())
    if args.max_records > 0:
        rows = rows[: args.max_records]

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} coffee-shop records -> {args.out}")
    if rows:
        print(f"Sample: {rows[0]['name']} | {rows[0].get('cuisine', '')} | {rows[0].get('address', '')}")


if __name__ == "__main__":
    main()
