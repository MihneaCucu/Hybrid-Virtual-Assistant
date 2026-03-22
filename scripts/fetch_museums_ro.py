from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from urllib.error import HTTPError

SOURCE_REGISTRY_DEFAULT = "data/kb_sources_bucharest.json"
OUTPUT_DEFAULT = "kb/structured/museums.jsonl"


def normalize_key(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", folded.lower())


def normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(folded.lower().split())


def read_source_registry(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_structured_source(registry: dict, source_id: str) -> dict | None:
    for source in registry.get("structured_sources", []):
        if source.get("id") == source_id:
            return source
    return None


def discover_csv_urls(entry_url: str, timeout: int = 20) -> list[str]:
    with urllib.request.urlopen(entry_url, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="ignore")

    links = re.findall(r'href=["\']([^"\']+\.csv[^"\']*)["\']', html, flags=re.IGNORECASE)
    if not links:
        raise RuntimeError(f"Could not find a CSV link on {entry_url}")

    seen: set[str] = set()
    urls: list[str] = []
    for link in links:
        absolute = urllib.parse.urljoin(entry_url, link)
        if absolute not in seen:
            urls.append(absolute)
            seen.add(absolute)
    return urls


def candidate_url_variants(url: str) -> list[str]:
    variants = [url]
    if url.startswith("http://"):
        variants.append("https://" + url[len("http://"):])
    return variants


def download_bytes(url: str, timeout: int = 30, retries: int = 2) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "HybridVA-QA/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(1 + attempt)
                continue
            raise
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected download failure")


def decode_csv_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def pick_value(row: dict, key_map: dict[str, str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in key_map:
            value = row.get(key_map[candidate], "")
            if value:
                return str(value).strip()
    return ""


def parse_float(text: str) -> float | None:
    raw = (text or "").strip()
    if not raw:
        return None
    normalized = raw.replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def make_reader(text: str) -> csv.DictReader:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="|;,\t")
        delimiter = dialect.delimiter
    except csv.Error:
        first_line = text.splitlines()[0] if text.splitlines() else ""
        delimiter = "|" if "|" in first_line else ","
    return csv.DictReader(io.StringIO(text), delimiter=delimiter)


def parse_csv_records(text: str) -> list[dict]:
    reader = make_reader(text)
    rows = list(reader)
    if not rows:
        return []

    first = rows[0]
    key_map = {normalize_key(key): key for key in first.keys() if key is not None}

    records: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        name = pick_value(
            row,
            key_map,
            ["denumirearomana", "denumire", "nume", "name", "titlu", "muzeu"],
        )
        name_en = pick_value(
            row,
            key_map,
            ["denumireaengleza", "nameenglish", "englishname"],
        )
        locality = pick_value(
            row,
            key_map,
            ["localitatea", "localitate", "oras", "municipiu", "city", "town"],
        )
        county = pick_value(row, key_map, ["judetul", "judet", "county", "region"])
        address = pick_value(row, key_map, ["adresa", "address", "strada"])
        description = pick_value(
            row,
            key_map,
            [
                "descriereasumararomana",
                "descrierearomana",
                "istoricromana",
                "profilulprincipalromana",
                "description",
                "specific",
            ],
        )
        website = pick_value(row, key_map, ["url", "website", "web", "site"])
        phone = pick_value(row, key_map, ["telefon", "phone", "tel"])
        category = pick_value(
            row,
            key_map,
            ["categoriaromana", "profilulprincipalromana", "tip", "category", "categorie", "clasificare"],
        )
        lat_text = pick_value(row, key_map, ["latitudine", "latitude", "lat"])
        lon_text = pick_value(row, key_map, ["longitudine", "longitude", "lon", "lng"])
        lat = parse_float(lat_text)
        lon = parse_float(lon_text)
        name_aliases = []
        if name_en:
            name_aliases.append(name_en)
        normalized_name = normalize_text(name)
        if "muzeul municipiului bucuresti" in normalized_name and "The Museum of Bucharest" not in name_aliases:
            name_aliases.append("The Museum of Bucharest")
        if "grigore antipa" in normalized_name and "Antipa Museum" not in name_aliases:
            name_aliases.append("Antipa Museum")

        records.append(
            {
                "record_id": f"museum_{idx:05d}",
                "record_type": "museum",
                "source": "romania_museums",
                "name": name,
                "name_en": name_en,
                "name_aliases": name_aliases,
                "city": locality,
                "county": county,
                "address": address,
                "description": description,
                "website": website,
                "phone": phone,
                "category": category,
                "lat": lat,
                "lon": lon,
            }
        )
    return records


def filter_bucharest(records: list[dict], city_keywords: list[str]) -> list[dict]:
    normalized_keywords = [normalize_text(keyword) for keyword in city_keywords if keyword.strip()]
    result: list[dict] = []

    for record in records:
        haystack = " ".join(
            [
                record.get("city", ""),
                record.get("county", ""),
                record.get("address", ""),
                record.get("description", ""),
            ]
        )
        haystack_norm = normalize_text(haystack)
        if any(keyword in haystack_norm for keyword in normalized_keywords):
            result.append(record)

    return result


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and normalize Romanian museum records.")
    parser.add_argument("--source-registry", default=SOURCE_REGISTRY_DEFAULT, help=f"Path to source registry JSON (default: {SOURCE_REGISTRY_DEFAULT})")
    parser.add_argument("--entry-url", default="", help="Dataset page URL. If empty, read from source registry.")
    parser.add_argument("--csv-url", default="", help="Direct CSV URL.")
    parser.add_argument("--csv-path", default="", help="Local CSV path (overrides URL download).")
    parser.add_argument("--out", default=OUTPUT_DEFAULT, help=f"Output JSONL path (default: {OUTPUT_DEFAULT})")
    parser.add_argument("--city-keywords", default="bucharest,bucuresti,bucurești", help="Comma-separated city filter keywords.")
    parser.add_argument("--max-records", type=int, default=0, help="Max output records after filtering (0 = all).")
    parser.add_argument("--timeout", type=int, default=30, help="Network timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Download retries for temporary HTTP failures.")
    args = parser.parse_args()

    if args.csv_path:
        with open(args.csv_path, "rb") as f:
            raw = f.read()
        print(f"Loaded local CSV: {args.csv_path}")
    else:
        csv_url = args.csv_url.strip()
        entry_url = args.entry_url.strip()
        candidate_urls: list[str] = []
        if csv_url:
            candidate_urls.extend(candidate_url_variants(csv_url))
        else:
            if not entry_url:
                registry = read_source_registry(args.source_registry)
                source = find_structured_source(registry, "romania_museums")
                if source is None:
                    raise RuntimeError("romania_museums source not found in source registry.")
                entry_url = source.get("entry_url", "").strip()
            if not entry_url:
                raise RuntimeError("Missing museums entry URL. Provide --entry-url or source registry value.")
            discovered_urls = discover_csv_urls(entry_url, timeout=args.timeout)
            for url in discovered_urls:
                candidate_urls.extend(candidate_url_variants(url))

        if not candidate_urls:
            raise RuntimeError("No candidate CSV URLs found.")

        raw = b""
        last_error: Exception | None = None
        for url in candidate_urls:
            try:
                print(f"Downloading museums CSV from: {url}")
                raw = download_bytes(url, timeout=args.timeout, retries=max(args.retries, 0))
                break
            except Exception as exc:
                last_error = exc
                print(f"- failed: {exc}")
        if not raw:
            raise RuntimeError(f"Failed to download any museums CSV URL. Last error: {last_error}")

    csv_text = decode_csv_bytes(raw)
    parsed = parse_csv_records(csv_text)
    keywords = [token.strip() for token in args.city_keywords.split(",")]
    filtered = filter_bucharest(parsed, keywords)
    if args.max_records > 0:
        filtered = filtered[: args.max_records]

    write_jsonl(args.out, filtered)
    print(f"Wrote {len(filtered)} records -> {args.out}")
    print(f"(parsed total rows: {len(parsed)})")


if __name__ == "__main__":
    main()
