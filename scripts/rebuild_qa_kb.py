from __future__ import annotations

import argparse
import subprocess
import sys


def run_step(cmd: list[str]) -> None:
    printable = " ".join(cmd)
    print(f"\n>> {printable}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild QA KB artifacts in one command (fetch + link + docs + index)."
    )
    parser.add_argument(
        "--no-network-fetch",
        action="store_true",
        help="Skip network data fetch steps and only run link/docs/index on existing structured files.",
    )
    parser.add_argument(
        "--tpbi-entry-url",
        default="https://gtfs.tpbi.ro/regional/",
        help="TPBI GTFS entry page URL.",
    )
    parser.add_argument(
        "--museums-entry-url",
        default="https://data.gov.ro/dataset/ghidul-muzeelor-din-romania",
        help="Romania museums dataset page URL.",
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip OSM place/address fetch even when network fetch is enabled.",
    )
    parser.add_argument(
        "--skip-restaurants",
        action="store_true",
        help="Skip OSM restaurant fetch even when network fetch is enabled.",
    )
    parser.add_argument(
        "--skip-coffee-shops",
        action="store_true",
        help="Skip OSM coffee-shop fetch even when network fetch is enabled.",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "final_demo"],
        default="full",
        help="Structured text profile passed to build_structured_text_docs.py.",
    )
    parser.add_argument(
        "--final-demo",
        action="store_true",
        help="Shortcut for --profile final_demo and skipping restaurant/coffee fetches.",
    )
    args = parser.parse_args()
    if args.final_demo:
        args.profile = "final_demo"
        args.skip_restaurants = True
        args.skip_coffee_shops = True

    py = sys.executable
    steps: list[list[str]] = []

    if not args.no_network_fetch:
        steps.append(
            [
                py,
                "scripts/fetch_tpbi_gtfs.py",
                "--entry-url",
                args.tpbi_entry_url,
                "--max-routes",
                "0",
                "--max-stops",
                "0",
            ]
        )
        steps.append(
            [
                py,
                "scripts/fetch_museums_ro.py",
                "--entry-url",
                args.museums_entry_url,
            ]
        )
        if not args.skip_osm:
            steps.append([py, "scripts/fetch_osm_places.py"])
        if not args.skip_restaurants:
            steps.append([py, "scripts/fetch_restaurants_overpass.py"])
        if not args.skip_coffee_shops:
            steps.append([py, "scripts/fetch_coffee_shops_overpass.py"])

    steps.extend(
        [
            [py, "scripts/link_places_to_metro.py"],
            [py, "scripts/build_structured_text_docs.py", "--clear-existing", "--profile", args.profile],
            [py, "scripts/build_index.py"],
        ]
    )

    for cmd in steps:
        run_step(cmd)

    print("\nDone. QA KB rebuilt successfully.")


if __name__ == "__main__":
    main()
