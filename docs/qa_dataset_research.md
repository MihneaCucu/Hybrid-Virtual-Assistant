# QA Dataset Research Notes (March 2026)

This project should use a bounded city knowledge base for runtime QA, not an open-domain 100k benchmark.

## Recommended Runtime Data (Bucharest)

1. TPBI GTFS (Bucharest-Ilfov transport)
- Entry: https://gtfs.tpbi.ro/regional/
- Why useful: routes, stops, service metadata for factual transport QA.
- Notes: open-data terms are provided by TPBI. Check usage constraints before redistribution.

2. Romania Museums dataset
- Entry: https://data.gov.ro/dataset/ghidul-muzeelor-din-romania
- Why useful: official museum records; filter to Bucharest entities.
- License: OGL-ROU-1.0 is linked from the dataset page.

3. OSM Overpass API
- Docs: https://wiki.openstreetmap.org/wiki/Overpass_API
- Why useful: attractions and place-of-interest entities with coordinates.
- License: ODbL attribution/share-alike obligations apply to OSM data.

4. Wikidata Query Service
- Entry: https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service
- Why useful: entity-level enrichment and normalization.
- License: Wikidata structured data is CC0.

## Methodology References (Not Runtime KB)

1. SQuAD Explorer
- https://rajpurkar.github.io/SQuAD-explorer/
- Use for: metric style (EM/F1), unanswerable setup inspiration (SQuAD 2.0), reader model background.

2. SQuAD paper entry
- https://aclanthology.org/D16-1264/
- Use for: report citation and benchmark framing.

## Practical Decision
- Runtime QA: custom Bucharest KB.
- Benchmark references: SQuAD in report/evaluation methodology.
- Architecture: BM25 retrieval + extractive reader + strict fallback.
