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
_museum_metro_aliases: list[tuple[str, dict]] = []
_museum_address_aliases: list[tuple[str, dict]] = []
_place_address_aliases: list[tuple[str, dict]] = []
_transit_agencies: list[dict] = []
_domain_config: dict = {}

DEFAULT_DOMAIN_CONFIG = {
    "nlu": {
        "question_markers": ["what", "when", "where", "who", "why", "how", "which", "tell me", "is", "are", "can"],
        "command_markers": ["set", "book", "schedule", "create", "add", "cancel", "remind"],
    },
    "qa": {
        "metro_station_markers": ["metro station", "subway station", "station", "metro line", "subway line"],
        "metro_relation_markers": ["at", "near", "nearest", "closest", "for", "to"],
        "metro_target_markers": ["museum", "muzeu", "park", "square", "monastery", "palace", "athenaeum", "landmark"],
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


def _contains_any(lowered_text: str, markers: list[str]) -> bool:
    return any(marker in lowered_text for marker in markers)


def _contains_any_word(lowered_text: str, markers: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered_text) is not None for marker in markers)


def _load_museum_metro_aliases(path: str) -> list[tuple[str, dict]]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []

    # Deduplicate by alias and keep the closest linked station when duplicates exist.
    best_by_alias: dict[str, tuple[float, dict]] = {}
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

        for name in names:
            norm = _normalize_for_match(str(name))
            if norm:
                current = best_by_alias.get(norm)
                if current is None or distance < current[0]:
                    best_by_alias[norm] = (distance, row)

    aliases: list[tuple[str, dict]] = [(alias, row) for alias, (_, row) in best_by_alias.items()]
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return aliases


def _load_museum_address_aliases(path: str) -> list[tuple[str, dict]]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []

    aliases: list[tuple[str, dict]] = []
    for row in rows:
        if row.get("record_type") != "museum":
            continue
        names = [row.get("name", ""), row.get("name_en", "")]
        names.extend(row.get("name_aliases", []) or [])
        for name in names:
            norm = _normalize_for_match(str(name))
            if norm:
                aliases.append((norm, row))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return aliases


def _load_place_address_aliases(path: str) -> list[tuple[str, dict]]:
    try:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []

    aliases: list[tuple[str, dict]] = []
    for row in rows:
        if row.get("record_type") != "osm_place":
            continue
        if row.get("slug") == "bucharest_city":
            # Avoid broad aliases that can over-trigger on many city-level questions.
            continue
        names = [row.get("title", ""), row.get("name", "")]
        names.extend(row.get("name_aliases", []) or [])
        slug = str(row.get("slug", "")).replace("_", " ").strip()
        if slug:
            names.append(slug)
        for name in names:
            norm = _normalize_for_match(str(name))
            if norm:
                aliases.append((norm, row))
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


def _answer_museum_metro_query(query: str) -> dict | None:
    if not _museum_metro_aliases:
        return None
    if not _is_museum_metro_query(query):
        return None

    normalized_query = _normalize_for_match(query)
    for alias, row in _museum_metro_aliases:
        if alias in normalized_query:
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
    for alias, row in _museum_address_aliases:
        if alias in normalized_query:
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
    for alias, row in _place_address_aliases:
        if alias in normalized_query:
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
    global _system_loaded, _chunks, _bm25, _museum_metro_aliases, _museum_address_aliases, _place_address_aliases, _transit_agencies, _domain_config
    if _system_loaded:
        return

    _domain_config = _load_domain_config(domain_config_path)
    _chunks, _bm25 = load_index(chunks_path=kb_path, index_path=index_path)
    _museum_metro_aliases = _load_museum_metro_aliases(museum_metro_links_path)
    _museum_address_aliases = _load_museum_address_aliases(museums_path)
    _place_address_aliases = _load_place_address_aliases(places_path)
    _transit_agencies = _load_transit_agencies(transit_path)
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
