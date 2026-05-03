# Knowledge/Q&A Confidence And Fallback Policy

## Goal

The Q&A module should answer only when it has local evidence from the Bucharest tourist-guide knowledge base or from deterministic structured records. It should not invent live facts, prices, bookings, or open-domain answers.

## Response Statuses

| Status | Meaning | Router behavior |
| --- | --- | --- |
| `answered` | The module found a grounded answer. | Speak/display `answer`. |
| `fallback` | The module cannot answer reliably from the KB. | Use fallback text or generic unsupported response. |
| `handoff` | The request mixes Q&A with a command. | Split or reroute to command/dialogue manager. |

## Confidence Thresholds

| Threshold | Value | Purpose |
| --- | ---: | --- |
| `BM25_THRESHOLD` | 2.0 | Reject very weak retrieval matches. |
| `BM25_MARGIN_THRESHOLD` | 0.10 | Treat nearly tied top retrieval scores as weak evidence. |
| `READER_THRESHOLD` | 0.25 | Reject low-confidence extractive answer spans. |

These thresholds are intentionally simple and explainable. They are not calibrated probabilities; they are engineering cutoffs for a bounded semester-project KB.

## Main Reason Codes

| Reason code | Status | Meaning | Example |
| --- | --- | --- | --- |
| `EMPTY_QUERY` | `fallback` | The user sent no usable text. | empty input |
| `MIXED_COMMAND_QUERY` | `handoff` | Command and factual request are mixed. | "Set a reminder and tell me where University Square is." |
| `UNSUPPORTED_PRICE_QUERY` | `fallback` | The user asks for current/live price information. | "How much is a metro ticket?" |
| `LOW_RETRIEVAL_CONFIDENCE` | `fallback` | BM25 evidence is too weak or ambiguous. | unrelated open-domain query |
| `LOW_READER_CONFIDENCE` | `fallback` | Reader did not find a reliable span. | weakly related question |
| `NO_RELEVANT_DOC` | `fallback` | Retrieval returned no candidates. | empty or impossible query |
| `AMBIGUOUS_PLACE_NAME` | `fallback` | Multiple structured places match the same alias. | ambiguous museum/place name |
| `NO_PLACE_MATCH_ON_LOCATION` | `fallback` | Listing query has no matching structured place. | restaurants on an unknown street |
| `NO_TRANSIT_DATA` | `fallback` | Transit data is unavailable at runtime. | nearby transport query without GTFS data |
| `NO_PLACE_COORDINATES` | `fallback` | Matched place lacks coordinates. | nearby transport for a place without lat/lon |
| `NO_NEARBY_TRANSIT_STOPS` | `fallback` | No stops found within the configured radius. | isolated place query |
| `RULE_BASED_LINK_MATCH` | `answered` | Nearest metro station answered from structured links. | "What metro station is Antipa Museum at?" |
| `RULE_BASED_ADDRESS_MATCH` | `answered` | Exact museum address answered from structured data. | "What is the exact address of Antipa Museum?" |
| `RULE_BASED_PLACE_ADDRESS_MATCH` | `answered` | Exact selected place address answered from structured data. | "Where is Stavropoleos Monastery?" |
| `RULE_BASED_NEARBY_TRANSPORT_MATCH` | `answered` | Nearby metro/STB answer built from coordinates and stops. | "Show metro and STB near Romanian Athenaeum." |
| `RULE_BASED_TRAVEL_GUIDANCE` | `answered` | Curated travel guidance answer. | "How many days should I stay in Bucharest?" |

## Safety Rules

- No open-web answering at runtime.
- No live prices, live weather, live opening hours, availability, or bookings.
- No broad open-domain knowledge claims.
- Every `answered` response should include `source_doc` and `sources`.
- Mixed command plus Q&A requests should be handled by the router/dialogue manager, not forced through the Q&A module.

## Report Note

The confidence score is a local ranking and thresholding signal, not a calibrated probability. The report should describe it as an engineering confidence score used to decide whether to answer or fallback.
