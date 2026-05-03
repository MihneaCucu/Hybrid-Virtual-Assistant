# QA Scope: Bucharest Tourist Guide

## Role
The Knowledge/QA module handles the information branch of the assistant. It answers factual tourist-guide questions from a bounded Bucharest knowledge base and refuses or hands off requests that are outside that scope.

## Supported Questions
- Landmark and city facts: definitions, historical facts, location context, symbolic meaning.
- Museum facts: type/category, address, website/phone if present in the local data.
- Exact place addresses for selected Bucharest landmarks and places.
- Nearest metro station for selected landmarks, museums, restaurants, and coffee shops when a structured link exists.
- Nearby transport stops around selected places when coordinates are available.
- Curated travel guidance: best season, recommended trip length, and broad daily budget ranges.

## Unsupported Questions
- Live weather, live prices, live availability, live opening hours, reservations, or booking execution.
- General world knowledge outside the Bucharest guide.
- Personal advice that requires current external data or subjective ranking beyond the curated facts.
- Commands such as reminders, booking, alarms, or calendar actions.

## Runtime Behavior
- `answered`: the module found a grounded answer in the KB or a deterministic structured rule.
- `fallback`: the question is unsupported, outside the KB, ambiguous, or below confidence thresholds.
- `handoff`: the request mixes a command with a factual question and should be handled by the router/dialogue manager.

## Final Architecture
The final QA design is closed-domain retrieval QA:

`question -> rule handlers for structured cases -> BM25 retrieval -> extractive reader -> confidence/fallback`

The module does not use open-web search at runtime and does not generate unsupported facts.
