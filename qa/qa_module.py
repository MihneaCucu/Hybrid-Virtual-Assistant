"""
qa_module.py — Public API for the Knowledge/QA module.

Usage:
    from qa.qa_module import load_qa_system, answer_question

    load_qa_system()   # call ONCE at startup
    result = answer_question("What are the opening hours of the Louvre?")
    # result = {
    #     "status": "answered",
    #     "reason_code": None,
    #     "answer": "The Louvre is open from 9am to 6pm.",
    #     "source_doc": "louvre_museum",
    #     "sources": [{"doc_id": "louvre_museum", "chunk_id": "louvre_museum_001"}],
    #     "confidence": 0.83,
    #     "fallback": False,
    # }
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import re
import unicodedata
from urllib.parse import urlparse

from qa.retrieval import load_index, retrieve
from qa.reader import load_reader, extract_answer
from qa.fallback import (
    make_answer_response,
    make_fallback_response,
    make_handoff_response,
    should_fallback_reader,
    should_fallback_retrieval,
)

_system_loaded = False
_chunks = []
_bm25 = None
_READER_TOP_N = 6
_museum_metro_aliases: list[tuple[str, list[dict]]] = []
_museum_address_aliases: list[tuple[str, list[dict]]] = []
_place_address_aliases: list[tuple[str, list[dict]]] = []
_transit_agencies: list[dict] = []
_transit_stops: list[dict] = []
_domain_config: dict = {}
_LEET_REPLACEMENTS = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
_MIN_ALIAS_CHARS = 3
_MAX_DISAMBIG_OPTIONS = 4

DEFAULT_DOMAIN_CONFIG = {
    "nlu": {
        "question_markers": ["what", "when", "where", "who", "why", "how", "which", "tell me", "is", "are", "can"],
        "command_markers": ["set", "book", "schedule", "create", "add", "cancel", "remind"],
    },
    "qa": {
        "metro_station_markers": ["metro station", "subway station", "station", "metro line", "subway line"],
        "metro_relation_markers": ["at", "near", "nearest", "closest", "for", "to"],
        "metro_target_markers": ["museum", "muzeu", "park", "square", "monastery", "palace", "athenaeum", "landmark"],
        "transport_nearby_markers": ["near", "nearby", "around", "vicinity", "close to", "in vicinity"],
        "transport_stop_markers": ["transport", "station", "stations", "stop", "stops", "stb", "metro", "bus", "tram", "trolleybus"],
        "address_markers": ["address", "street", "number", "located at", "where exactly", "where is", "located"],
        "museum_markers": ["museum", "muzeu"],
        "price_query_markers": ["how much", "price", "cost", "fare", "ticket"],
        "transit_markers": ["metro", "subway", "bus", "tram", "transport", "ticket", "fare", "stb", "metrorex"],
        "parking_markers": ["parking", "parcare", "park fee", "parking fee"],
        "symbolic_query_markers": ["symbolize", "symbolise", "commemorate", "historical event"],
        "exact_location_query_markers": ["exact address", "street", "number", "where exactly", "exact location"],
        "symbolic_answer_markers": ["world war", "victory", "coronation", "king", "historical"],
        "location_answer_markers": ["strada", "street", "soseaua", "boulevard", "sector", "piata", "nr"],
        "price_answer_markers": ["lei", "ron", "euro", "eur", "$", "usd", "€"],
        "nearby_transport_radius_m": 800,
        "nearby_transport_max_stops": 6,
        "nearby_transport_max_metro_stops": 2,
        "nearby_transport_max_surface_stops": 4,
        "nearby_transport_max_lines_per_stop": 5,
    },
    "links": {
        "pmb_url": "https://www.pmb.ro",
        "transit_official_links": [
            {"name": "STB SA", "url": "http://stbsa.ro"},
            {"name": "METROREX SA", "url": "http://www.metrorex.ro"},
        ],
    },
}


def _looks_like_question(text: str) -> bool:
    lowered = text.lower().strip()
    if "?" in lowered:
        return True
    return _contains_any_word(lowered, _cfg_list("nlu", "question_markers"))


def _looks_like_command(text: str) -> bool:
    lowered = text.lower().strip()
    return _contains_any_word(lowered, _cfg_list("nlu", "command_markers"))


def _normalize_for_match(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    lowered = folded.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _deleet_text(text: str) -> str:
    return text.translate(_LEET_REPLACEMENTS)


def _name_variants(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    variants: list[str] = [raw]
    deleeted = _deleet_text(raw)
    if deleeted.lower() != raw.lower():
        variants.append(deleeted)

    normalized = _normalize_for_match(raw)
    tokens = normalized.split()
    if tokens:
        first = tokens[0]
        if len(first) >= 5 or (any(ch.isdigit() for ch in first) and len(first) >= 3):
            variants.append(first)
            first_deleeted = _deleet_text(first)
            if first_deleeted != first and len(first_deleeted) >= _MIN_ALIAS_CHARS:
                variants.append(first_deleeted)

    unique: list[str] = []
    seen: set[str] = set()
    for item in variants:
        norm = _normalize_for_match(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique.append(item)
    return unique


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_domain_config(path: str) -> dict:
    config = deepcopy(DEFAULT_DOMAIN_CONFIG)
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
    except FileNotFoundError:
        return config
    except json.JSONDecodeError:
        return config
    if not isinstance(loaded, dict):
        return config
    return _deep_merge(config, loaded)


def _cfg_list(section: str, key: str) -> list[str]:
    section_obj = _domain_config.get(section, {})
    if not isinstance(section_obj, dict):
        return []
    values = section_obj.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(item).lower() for item in values if str(item).strip()]


def _cfg_links() -> list[dict]:
    links_obj = _domain_config.get("links", {})
    if not isinstance(links_obj, dict):
        return []
    values = links_obj.get("transit_official_links", [])
    if not isinstance(values, list):
        return []
    result = []
    for item in values:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        if name and url:
            result.append({"name": name, "url": url})
    return result


def _cfg_text(section: str, key: str, default: str = "") -> str:
    section_obj = _domain_config.get(section, {})
    if not isinstance(section_obj, dict):
        return default
    value = section_obj.get(key, default)
    return str(value).strip() if value is not None else default


def _cfg_int(section: str, key: str, default: int) -> int:
    section_obj = _domain_config.get(section, {})
    if not isinstance(section_obj, dict):
        return default
    value = section_obj.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _contains_any(lowered_text: str, markers: list[str]) -> bool:
    return any(marker in lowered_text for marker in markers)


def _contains_any_word(lowered_text: str, markers: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered_text) is not None for marker in markers)


def _to_float(value) -> float | None:
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


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _alias_in_query(alias: str, normalized_query: str) -> bool:
    if len(alias) < _MIN_ALIAS_CHARS:
        return False
    return re.search(rf"\b{re.escape(alias)}\b", normalized_query) is not None


def _row_place_name(row: dict) -> str:
    return (
        row.get("place_name_en")
        or row.get("place_name")
        or row.get("name_en")
        or row.get("name")
        or row.get("title")
        or row.get("museum_name_en")
        or row.get("museum_name")
        or "this place"
    )


def _row_place_address(row: dict) -> str:
    return (
        row.get("place_address")
        or row.get("address")
        or row.get("display_name")
        or row.get("place_city")
        or row.get("city")
        or ""
    )


def _build_ambiguous_place_fallback(rows: list[dict]) -> dict:
    place_name = _row_place_name(rows[0]) if rows else "this place"
    options: list[str] = []
    seen: set[str] = set()

    for row in rows:
        name = _row_place_name(row)
        address = _row_place_address(row)
        option = f"{name} ({address})" if address else name
        key = option.lower()
        if key in seen:
            continue
        seen.add(key)
        options.append(option)
        if len(options) >= _MAX_DISAMBIG_OPTIONS:
            break

    if not options:
        return make_fallback_response(reason_code="AMBIGUOUS_PLACE_NAME")

    options_text = "; ".join(options)
    return make_fallback_response(
        reason_code="AMBIGUOUS_PLACE_NAME",
        answer=f"I found multiple locations for {place_name}. Please specify one: {options_text}.",
    )


def _distinct_place_rows(rows: list[dict]) -> list[dict]:
    selected: dict[str, dict] = {}
    for row in rows:
        lat, lon = _extract_place_coords(row)
        if lat is not None and lon is not None:
            key = f"geo:{round(lat, 3)}:{round(lon, 3)}"
        else:
            key = f"text:{_normalize_for_match(_row_place_name(row))}:{_normalize_for_match(_row_place_address(row))}"
        if key not in selected:
            selected[key] = row
    return list(selected.values())


def _load_museum_metro_aliases(path: str) -> list[tuple[str, list[dict]]]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []

    alias_to_rows: dict[str, dict[str, tuple[float, dict]]] = {}
    for row in rows:
        if row.get("record_type") not in {"museum_metro_link", "place_metro_link"}:
            continue

        names = [
            row.get("place_name", ""),
            row.get("place_name_en", ""),
            row.get("museum_name", ""),
            row.get("museum_name_en", ""),
        ]
        names.extend(row.get("place_name_aliases", []) or [])
        names.extend(row.get("museum_name_aliases", []) or [])
        try:
            distance = float(row.get("distance_m", 1e9))
        except (TypeError, ValueError):
            distance = 1e9

        expanded_names: list[str] = []
        for name in names:
            expanded_names.extend(_name_variants(str(name)))

        for name in expanded_names:
            norm = _normalize_for_match(str(name))
            if not norm:
                continue
            by_record = alias_to_rows.setdefault(norm, {})
            record_id = str(row.get("record_id", "")).strip()
            if not record_id:
                continue
            current = by_record.get(record_id)
            if current is None or distance < current[0]:
                by_record[record_id] = (distance, row)

    aliases: list[tuple[str, list[dict]]] = []
    for alias, by_record in alias_to_rows.items():
        rows_sorted = [item[1] for item in sorted(by_record.values(), key=lambda pair: pair[0])]
        if rows_sorted:
            aliases.append((alias, rows_sorted))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return aliases


def _load_museum_address_aliases(path: str) -> list[tuple[str, list[dict]]]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []

    alias_to_rows: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row.get("record_type") != "museum":
            continue
        names = [row.get("name", ""), row.get("name_en", "")]
        names.extend(row.get("name_aliases", []) or [])
        expanded_names: list[str] = []
        for name in names:
            expanded_names.extend(_name_variants(str(name)))

        for name in expanded_names:
            norm = _normalize_for_match(str(name))
            if not norm:
                continue
            record_id = str(row.get("record_id", "")).strip()
            if not record_id:
                continue
            alias_to_rows.setdefault(norm, {})[record_id] = row

    aliases: list[tuple[str, list[dict]]] = []
    for alias, by_record in alias_to_rows.items():
        rows_for_alias = list(by_record.values())
        if rows_for_alias:
            aliases.append((alias, rows_for_alias))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return aliases


def _load_place_address_aliases(path: str) -> list[tuple[str, list[dict]]]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []

    alias_to_rows: dict[str, dict[str, dict]] = {}
    for row in rows:
        record_type = row.get("record_type")
        if record_type not in {"osm_place", "restaurant"}:
            continue
        if record_type == "osm_place" and row.get("slug") == "bucharest_city":
            # Avoid broad aliases that can over-trigger on many city-level questions.
            continue
        names = [row.get("title", ""), row.get("name", ""), row.get("name_en", "")]
        names.extend(row.get("name_aliases", []) or [])
        slug = str(row.get("slug", "")).replace("_", " ").strip()
        if slug:
            names.append(slug)
        expanded_names: list[str] = []
        for name in names:
            expanded_names.extend(_name_variants(str(name)))

        for name in expanded_names:
            norm = _normalize_for_match(str(name))
            if not norm:
                continue
            record_id = str(row.get("record_id", "")).strip()
            if not record_id:
                continue
            alias_to_rows.setdefault(norm, {})[record_id] = row

    aliases: list[tuple[str, list[dict]]] = []
    for alias, by_record in alias_to_rows.items():
        rows_for_alias = list(by_record.values())
        if rows_for_alias:
            aliases.append((alias, rows_for_alias))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return aliases


def _load_transit_agencies(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []

    agencies = []
    for row in rows:
        if row.get("record_type") != "agency":
            continue
        name = str(row.get("agency_name", "")).strip()
        url = str(row.get("agency_url", "")).strip()
        if not name or not url:
            continue
        agencies.append({"name": name, "url": url})
    return agencies


def _list_strings(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _load_transit_stops(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []

    stops: list[dict] = []
    for row in rows:
        if row.get("record_type") != "stop":
            continue
        lat = _to_float(row.get("stop_lat"))
        lon = _to_float(row.get("stop_lon"))
        if lat is None or lon is None:
            continue
        stops.append(
            {
                "stop_id": str(row.get("stop_id", "")).strip(),
                "stop_name": str(row.get("stop_name", "")).strip() or "Unknown stop",
                "lat": lat,
                "lon": lon,
                "is_subway_stop": bool(row.get("is_subway_stop")),
                "route_short_names": _list_strings(row.get("route_short_names")),
                "route_type_labels": _list_strings(row.get("route_type_labels")),
            }
        )
    return stops


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    return clean


def _build_price_fallback_answer() -> str:
    return _build_price_fallback_answer_for_query("")


def _is_transit_query(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, _cfg_list("qa", "transit_markers"))


def _is_parking_query(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, _cfg_list("qa", "parking_markers"))


def _build_price_fallback_answer_for_query(query: str) -> str:
    if _is_parking_query(query):
        pmb_url = _cfg_text("links", "pmb_url", "https://www.pmb.ro")
        return (
            "I can't confirm current parking prices from the knowledge base. "
            f"Please check the official Bucharest City Hall website: {pmb_url}."
        )

    if not _is_transit_query(query):
        return (
            "I can't confirm current prices for that request from the knowledge base. "
            "Please check an official source for up-to-date pricing."
        )

    preferred = []
    others = []
    for agency in _transit_agencies:
        name = agency["name"]
        lowered = name.lower()
        item = f"{name}: {_normalize_url(agency['url'])}"
        if "metro" in lowered or "stb" in lowered:
            preferred.append(item)
        else:
            others.append(item)

    selected = preferred[:2] if preferred else others[:2]
    if not selected:
        fallback_links = _cfg_links()
        selected = [f"{item['name']}: {_normalize_url(item['url'])}" for item in fallback_links[:2]]
    if not selected:
        return (
            "I can't confirm ticket prices from the current knowledge base. "
            "Please check official transit operator websites."
        )
    links = " | ".join(selected)
    return (
        "I can't confirm current ticket prices from the knowledge base. "
        f"Please check official sources: {links}."
    )


def _is_museum_metro_query(text: str) -> bool:
    lowered = text.lower()
    has_station = _contains_any(lowered, _cfg_list("qa", "metro_station_markers"))
    has_relation = _contains_any_word(lowered, _cfg_list("qa", "metro_relation_markers"))
    has_target_hint = _contains_any(lowered, _cfg_list("qa", "metro_target_markers"))
    return has_station and (has_relation or has_target_hint)


def _is_museum_address_query(text: str) -> bool:
    lowered = text.lower()
    has_address = _contains_any(lowered, _cfg_list("qa", "address_markers"))
    has_target = _contains_any(lowered, _cfg_list("qa", "museum_markers"))
    return has_address and has_target


def _is_place_address_query(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, _cfg_list("qa", "address_markers"))


def _is_nearby_transport_query(text: str) -> bool:
    lowered = text.lower()
    has_nearby = _contains_any(lowered, _cfg_list("qa", "transport_nearby_markers"))
    has_transport = _contains_any(lowered, _cfg_list("qa", "transport_stop_markers"))
    return has_nearby and has_transport


def _extract_place_coords(row: dict) -> tuple[float | None, float | None]:
    lat = _to_float(row.get("place_lat"))
    lon = _to_float(row.get("place_lon"))
    if lat is not None and lon is not None:
        return lat, lon
    lat = _to_float(row.get("lat"))
    lon = _to_float(row.get("lon"))
    return lat, lon


def _format_stop_summary(stop: dict, distance_m: float, max_lines: int) -> str:
    mode = "metro" if stop.get("is_subway_stop") else "STB"
    lines = [line for line in stop.get("route_short_names", []) if line][:max_lines]
    if lines:
        return f"{stop['stop_name']} ({distance_m:.0f} m, {mode}, lines {', '.join(lines)})"
    return f"{stop['stop_name']} ({distance_m:.0f} m, {mode})"


def _find_place_rows_in_query(normalized_query: str) -> tuple[str, list[dict]] | None:
    sources = [_museum_metro_aliases, _place_address_aliases, _museum_address_aliases]
    for alias_source in sources:
        for alias, rows in alias_source:
            if _alias_in_query(alias, normalized_query):
                return alias, rows
    return None


def _answer_nearby_transport_query(query: str) -> dict | None:
    if not _is_nearby_transport_query(query):
        return None
    if not _transit_stops:
        return make_fallback_response(reason_code="NO_TRANSIT_DATA")

    normalized_query = _normalize_for_match(query)
    place_match = _find_place_rows_in_query(normalized_query)
    if place_match is None:
        return None

    _, rows = place_match
    distinct_rows = _distinct_place_rows(rows)
    if len(distinct_rows) > 1:
        return _build_ambiguous_place_fallback(distinct_rows)

    place_row = distinct_rows[0]
    place_name = _row_place_name(place_row)
    lat, lon = _extract_place_coords(place_row)
    if lat is None or lon is None:
        return make_fallback_response(reason_code="NO_PLACE_COORDINATES")

    radius_m = max(200, _cfg_int("qa", "nearby_transport_radius_m", 800))
    max_total = max(1, _cfg_int("qa", "nearby_transport_max_stops", 6))
    max_metro = max(1, _cfg_int("qa", "nearby_transport_max_metro_stops", 2))
    max_surface = max(1, _cfg_int("qa", "nearby_transport_max_surface_stops", 4))
    max_lines = max(1, _cfg_int("qa", "nearby_transport_max_lines_per_stop", 5))

    nearby: list[tuple[float, dict]] = []
    for stop in _transit_stops:
        distance = _haversine_m(lat, lon, stop["lat"], stop["lon"])
        if distance <= radius_m:
            nearby.append((distance, stop))

    if not nearby:
        return make_fallback_response(
            reason_code="NO_NEARBY_TRANSIT_STOPS",
            answer=f"I couldn't find transport stops within {radius_m} meters of {place_name}.",
        )

    nearby.sort(key=lambda item: item[0])
    metro = [(d, s) for d, s in nearby if s.get("is_subway_stop")]
    surface = [(d, s) for d, s in nearby if not s.get("is_subway_stop")]

    chosen: list[tuple[float, dict]] = []
    chosen.extend(metro[:max_metro])
    chosen.extend(surface[:max_surface])
    chosen.sort(key=lambda item: item[0])
    chosen = chosen[:max_total]

    deduped: list[tuple[float, dict]] = []
    seen_keys: set[tuple[str, bool]] = set()
    for dist, stop in chosen:
        key = (str(stop.get("stop_name", "")).strip().lower(), bool(stop.get("is_subway_stop")))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((dist, stop))
    chosen = deduped

    metro_summaries = [
        _format_stop_summary(stop, dist, max_lines)
        for dist, stop in chosen
        if stop.get("is_subway_stop")
    ]
    surface_summaries = [
        _format_stop_summary(stop, dist, max_lines)
        for dist, stop in chosen
        if not stop.get("is_subway_stop")
    ]
    if len(surface_summaries) > 2:
        surface_summaries = surface_summaries[:1]

    parts = [f"Nearby transport for {place_name}:"]
    if metro_summaries:
        parts.append("Metro: " + "; ".join(metro_summaries) + ".")
    if surface_summaries:
        parts.append("STB: " + "; ".join(surface_summaries) + ".")
    answer = " ".join(parts)

    source_doc = f"structured_{place_row.get('record_id', 'transit')}"
    sources = [{"doc_id": source_doc, "chunk_id": None}]
    for dist, stop in chosen[:3]:
        _ = dist
        sources.append({"doc_id": f"structured_stop_{stop.get('stop_id', '')}", "chunk_id": None})

    return {
        "status": "answered",
        "reason_code": "RULE_BASED_NEARBY_TRANSPORT_MATCH",
        "answer": answer,
        "source_doc": source_doc,
        "sources": sources,
        "confidence": 0.98,
        "fallback": False,
    }


def _answer_museum_metro_query(query: str) -> dict | None:
    if not _museum_metro_aliases:
        return None
    if not _is_museum_metro_query(query):
        return None

    normalized_query = _normalize_for_match(query)
    for alias, rows in _museum_metro_aliases:
        if _alias_in_query(alias, normalized_query):
            distinct_rows = _distinct_place_rows(rows)
            if len(distinct_rows) > 1:
                return _build_ambiguous_place_fallback(distinct_rows)

            row = distinct_rows[0]
            place_name = (
                row.get("place_name_en")
                or row.get("place_name")
                or row.get("museum_name_en")
                or row.get("museum_name")
                or "this place"
            )
            station = row.get("metro_stop_name", "unknown station")
            distance = row.get("distance_m", "unknown")
            source_doc = f"structured_{row.get('record_id', 'place_metro_link')}"
            answer = f"The nearest metro station to {place_name} is {station} (about {distance} meters)."
            return {
                "status": "answered",
                "reason_code": "RULE_BASED_LINK_MATCH",
                "answer": answer,
                "source_doc": source_doc,
                "sources": [{"doc_id": source_doc, "chunk_id": None}],
                "confidence": 0.99,
                "fallback": False,
            }
    return None


def _answer_museum_address_query(query: str) -> dict | None:
    if not _museum_address_aliases:
        return None
    if not _is_museum_address_query(query):
        return None

    normalized_query = _normalize_for_match(query)
    for alias, rows in _museum_address_aliases:
        if _alias_in_query(alias, normalized_query):
            distinct_rows = _distinct_place_rows(rows)
            if len(distinct_rows) > 1:
                return _build_ambiguous_place_fallback(distinct_rows)

            row = distinct_rows[0]
            museum_name = row.get("name_en") or row.get("name") or "this museum"
            address = row.get("address", "")
            if not address:
                return None
            answer = f"The exact address of {museum_name} is {address}."
            source_doc = f"structured_{row.get('record_id', 'museum_record')}"
            return {
                "status": "answered",
                "reason_code": "RULE_BASED_ADDRESS_MATCH",
                "answer": answer,
                "source_doc": source_doc,
                "sources": [{"doc_id": source_doc, "chunk_id": None}],
                "confidence": 0.99,
                "fallback": False,
            }
    return None


def _answer_place_address_query(query: str) -> dict | None:
    if not _place_address_aliases:
        return None
    if not _is_place_address_query(query):
        return None

    normalized_query = _normalize_for_match(query)
    for alias, rows in _place_address_aliases:
        if _alias_in_query(alias, normalized_query):
            distinct_rows = _distinct_place_rows(rows)
            if len(distinct_rows) > 1:
                return _build_ambiguous_place_fallback(distinct_rows)

            row = distinct_rows[0]
            place_name = row.get("title") or row.get("name") or "this place"
            address = row.get("address") or row.get("display_name")
            if not address:
                return None
            answer = f"The exact address of {place_name} is {address}."
            source_doc = f"structured_{row.get('record_id', 'osm_place')}"
            return {
                "status": "answered",
                "reason_code": "RULE_BASED_PLACE_ADDRESS_MATCH",
                "answer": answer,
                "source_doc": source_doc,
                "sources": [{"doc_id": source_doc, "chunk_id": None}],
                "confidence": 0.99,
                "fallback": False,
            }
    return None


def _is_price_query(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, _cfg_list("qa", "price_query_markers"))


def _is_symbolic_query(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, _cfg_list("qa", "symbolic_query_markers"))


def _is_exact_location_query(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, _cfg_list("qa", "exact_location_query_markers"))


def _answer_looks_like_price(text: str) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, _cfg_list("qa", "price_answer_markers"))


def load_qa_system(
    kb_path: str = "kb/chunks.jsonl",
    index_path: str = "kb/bm25_index.pkl",
    museum_metro_links_path: str = "kb/structured/museum_metro_links.jsonl",
    museums_path: str = "kb/structured/museums.jsonl",
    places_path: str = "kb/structured/osm_places.jsonl",
    restaurants_path: str = "kb/structured/restaurants.jsonl",
    transit_path: str = "kb/structured/transit.jsonl",
    domain_config_path: str = "data/domain_config_bucharest.json",
) -> None:
    """
    Load the knowledge base index and QA model into memory.
    Must be called ONCE before any calls to answer_question().

    Args:
        kb_path:    Path to the chunked knowledge base (JSONL).
        index_path: Path to the serialized BM25 index (pickle).
    """
    global _system_loaded, _chunks, _bm25, _museum_metro_aliases, _museum_address_aliases, _place_address_aliases, _transit_agencies, _transit_stops, _domain_config
    if _system_loaded:
        return

    _domain_config = _load_domain_config(domain_config_path)
    _chunks, _bm25 = load_index(chunks_path=kb_path, index_path=index_path)
    _museum_metro_aliases = _load_museum_metro_aliases(museum_metro_links_path)
    _museum_address_aliases = _load_museum_address_aliases(museums_path)
    _place_address_aliases = _load_place_address_aliases(places_path)
    _place_address_aliases.extend(_load_place_address_aliases(restaurants_path))
    _place_address_aliases.sort(key=lambda item: len(item[0]), reverse=True)
    _transit_agencies = _load_transit_agencies(transit_path)
    _transit_stops = _load_transit_stops(transit_path)
    load_reader()
    _system_loaded = True


def answer_question(query: str) -> dict:
    """
    Answer a factual question using the loaded knowledge base.

    Args:
        query: The user's natural-language question.

    Returns:
        A dict with the following keys:
            status      (str)         - answered | fallback | handoff
            reason_code (str | None)  - machine-readable fallback/handoff reason
            answer      (str | None)  - extracted answer span, or None on fallback
            source_doc  (str | None)  - document slug the answer came from
            sources     (list[dict])  - evidence chunks used
            confidence  (float)       - confidence score in [0, 1]
            fallback    (bool)        - True if no reliable answer was found
    """
    if not _system_loaded:
        raise RuntimeError("Must call load_qa_system() before answering questions.")

    query = query.strip()
    if not query:
        return make_fallback_response(reason_code="EMPTY_QUERY")

    if _looks_like_question(query) and _looks_like_command(query):
        return make_handoff_response(reason_code="MIXED_COMMAND_QUERY")

    nearby_transport_answer = _answer_nearby_transport_query(query)
    if nearby_transport_answer is not None:
        return nearby_transport_answer

    linked_answer = _answer_museum_metro_query(query)
    if linked_answer is not None:
        return linked_answer

    address_answer = _answer_museum_address_query(query)
    if address_answer is not None:
        return address_answer

    place_address_answer = _answer_place_address_query(query)
    if place_address_answer is not None:
        return place_address_answer

    # 1. Retrieve top passages
    results = retrieve(query, _bm25, _chunks, top_k=5)

    if not results:
        if _is_price_query(query):
            return make_fallback_response(
                reason_code="UNSUPPORTED_PRICE_QUERY",
                answer=_build_price_fallback_answer_for_query(query),
            )
        return make_fallback_response(reason_code="NO_RELEVANT_DOC")

    best_result = results[0]
    best_chunk = best_result["chunk"]
    bm25_score = best_result["score"]
    second_score = results[1]["score"] if len(results) > 1 else None

    # 2. Check retrieval fallback threshold
    if should_fallback_retrieval(bm25_score, second_score):
        if _is_price_query(query):
            return make_fallback_response(
                reason_code="UNSUPPORTED_PRICE_QUERY",
                answer=_build_price_fallback_answer_for_query(query),
            )
        return make_fallback_response(reason_code="LOW_RETRIEVAL_CONFIDENCE")

    # 3. Extract answer spans from top-N retrieved chunks and choose best by score.
    candidates = []
    for item in results[:_READER_TOP_N]:
        chunk = item["chunk"]
        extraction = extract_answer(query, chunk["text"])
        if extraction["answer"] is not None:
            candidates.append((extraction, chunk))

    if not candidates:
        if _is_price_query(query):
            return make_fallback_response(
                reason_code="UNSUPPORTED_PRICE_QUERY",
                answer=_build_price_fallback_answer_for_query(query),
            )
        return make_fallback_response(reason_code="LOW_READER_CONFIDENCE")

    prioritized = candidates
    if _is_symbolic_query(query):
        symbolic = []
        symbolic_markers = _cfg_list("qa", "symbolic_answer_markers")
        for extraction, chunk in candidates:
            answer_l = extraction["answer"].lower()
            if _contains_any(answer_l, symbolic_markers):
                symbolic.append((extraction, chunk))
        if symbolic:
            prioritized = symbolic

    if _is_exact_location_query(query):
        location_like = []
        location_markers = _cfg_list("qa", "location_answer_markers")
        for extraction, chunk in candidates:
            answer_l = extraction["answer"].lower()
            has_address_token = _contains_any(answer_l, location_markers)
            has_number = re.search(r"\d", answer_l) is not None
            looks_phone = re.search(r"\b\d{2,4}[/.:-]\d", answer_l) is not None
            if has_address_token and has_number and not looks_phone:
                location_like.append((extraction, chunk))
        if location_like:
            prioritized = location_like

    best_extraction, best_chunk_for_answer = max(prioritized, key=lambda pair: pair[0]["score"])

    # 4. Check reader fallback threshold
    if should_fallback_reader(best_extraction["score"]):
        if _is_price_query(query):
            return make_fallback_response(
                reason_code="UNSUPPORTED_PRICE_QUERY",
                answer=_build_price_fallback_answer_for_query(query),
            )
        return make_fallback_response(reason_code="LOW_READER_CONFIDENCE")

    if _is_price_query(query) and not _answer_looks_like_price(best_extraction["answer"]):
        return make_fallback_response(
            reason_code="UNSUPPORTED_PRICE_QUERY",
            answer=_build_price_fallback_answer_for_query(query),
        )

    return make_answer_response(
        answer=best_extraction["answer"],
        chunk=best_chunk_for_answer,
        reader_score=best_extraction["score"],
    )
