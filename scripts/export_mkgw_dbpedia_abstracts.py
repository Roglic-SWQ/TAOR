from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "https://dbpedia.org/sparql"
DEFAULT_USER_AGENT = "Knowledge-Bridger-MKGW-abstract-export/1.0"
FIELDS = [
    "entity_id",
    "entity_key",
    "wikidata_uri",
    "dbpedia_uri",
    "abstract_status",
    "abstract",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MKG-W DBpedia English abstracts aligned to entity2id order."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-dbpedia",
        action="store_true",
        help="Skip DBpedia SPARQL and use only the configured fallback source.",
    )
    parser.add_argument(
        "--fallback-source",
        choices=("none", "wikipedia-summary"),
        default="none",
        help="Optional fallback for DBpedia URIs that have no abstract from SPARQL.",
    )
    parser.add_argument(
        "--cache-json",
        default=None,
        help="Optional JSON cache mapping DBpedia URI to abstract. Existing entries are reused.",
    )
    return parser.parse_args()


def export_abstract_rows(
    *,
    dataset_dir: Path,
    output_path: Path,
    abstracts_by_dbpedia_uri: dict[str, str],
    limit: int | None = None,
) -> dict[str, Any]:
    entity_rows = load_entity_rows(dataset_dir)
    ent_links = load_ent_links(dataset_dir)
    if limit is not None:
        entity_rows = entity_rows[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for entity_row in entity_rows:
        wikidata_uri = entity_row["wikidata_uri"]
        dbpedia_uri = ent_links[wikidata_uri]
        abstract = normalize_abstract(abstracts_by_dbpedia_uri.get(dbpedia_uri, ""))
        rows.append(
            {
                "entity_id": str(entity_row["entity_id"]),
                "entity_key": entity_row["entity_key"],
                "wikidata_uri": wikidata_uri,
                "dbpedia_uri": dbpedia_uri,
                "abstract_status": "found" if abstract else "missing",
                "abstract": abstract,
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with_abstract = sum(1 for row in rows if row["abstract_status"] == "found")
    return {
        "ok": True,
        "entities": len(rows),
        "with_abstract": with_abstract,
        "missing_abstract": len(rows) - with_abstract,
        "output_path": str(output_path),
    }


def load_entity_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "entity2id.txt"
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(f"expected two columns in {path}:{line_number}")
        wikidata_uri, raw_entity_id = parts
        entity_id = int(raw_entity_id)
        rows.append(
            {
                "entity_id": entity_id,
                "entity_key": wikidata_uri.rstrip("/").rsplit("/", maxsplit=1)[-1],
                "wikidata_uri": wikidata_uri,
            }
        )
    return sorted(rows, key=lambda row: row["entity_id"])


def load_ent_links(dataset_dir: Path) -> dict[str, str]:
    path = dataset_dir / "ent_links"
    links: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2:
            raise ValueError(f"expected two tab-separated columns in {path}:{line_number}")
        dbpedia_uri, wikidata_uri = parts
        if wikidata_uri in links:
            raise ValueError(f"duplicate Wikidata URI in {path}:{line_number}: {wikidata_uri}")
        links[wikidata_uri] = dbpedia_uri
    return links


def fetch_dbpedia_abstracts(
    dbpedia_uris: list[str],
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    batch_size: int = 50,
    timeout_seconds: float = 30.0,
    sleep_seconds: float = 0.2,
) -> dict[str, str]:
    abstracts: dict[str, str] = {}
    for start in range(0, len(dbpedia_uris), batch_size):
        batch = dbpedia_uris[start : start + batch_size]
        try:
            abstracts.update(fetch_abstract_batch(batch, endpoint=endpoint, timeout_seconds=timeout_seconds))
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"warning: DBpedia abstract batch failed at offset {start}: {exc}")
        if sleep_seconds and start + batch_size < len(dbpedia_uris):
            time.sleep(sleep_seconds)
    return abstracts


def fetch_abstract_batch(
    dbpedia_uris: list[str],
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout_seconds: float = 30.0,
) -> dict[str, str]:
    values = " ".join(f"<{uri}>" for uri in dbpedia_uris)
    query = f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT ?resource ?abstract WHERE {{
  VALUES ?resource {{ {values} }}
  ?resource dbo:abstract ?abstract .
  FILTER(lang(?abstract) = "en")
}}
"""
    payload = urllib.parse.urlencode(
        {
            "query": query,
            "format": "application/sparql-results+json",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": DEFAULT_USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload_obj = json.loads(response.read().decode("utf-8"))

    rows: dict[str, str] = {}
    for binding in payload_obj.get("results", {}).get("bindings", []):
        resource = binding.get("resource", {}).get("value")
        abstract = normalize_abstract(binding.get("abstract", {}).get("value", ""))
        if resource and abstract:
            rows[resource] = abstract
    return rows


def fetch_wikipedia_summaries(
    dbpedia_uris: list[str],
    *,
    batch_size: int = 50,
    timeout_seconds: float = 30.0,
    sleep_seconds: float = 0.2,
    max_retries: int = 3,
    retry_sleep_seconds: float = 10.0,
    cache_path: Path | None = None,
    abstracts_by_dbpedia_uri: dict[str, str] | None = None,
) -> dict[str, str]:
    abstracts: dict[str, str] = {} if abstracts_by_dbpedia_uri is None else abstracts_by_dbpedia_uri
    for start in range(0, len(dbpedia_uris), batch_size):
        batch = dbpedia_uris[start : start + batch_size]
        titles_by_uri = {uri: dbpedia_uri_to_wikipedia_title(uri) for uri in batch}
        summaries = fetch_wikipedia_summary_batch(
            list(titles_by_uri.values()),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        for uri, title in titles_by_uri.items():
            summary = normalize_abstract(summaries.get(title, ""))
            if summary:
                abstracts[uri] = summary
        write_cache(cache_path, abstracts)
        print(
            f"wikipedia fallback progress: {min(start + batch_size, len(dbpedia_uris))}/"
            f"{len(dbpedia_uris)} batch_items, cached={len(abstracts)}"
        )
        if sleep_seconds and start + batch_size < len(dbpedia_uris):
            time.sleep(sleep_seconds)
    return abstracts


def fetch_wikipedia_summary_batch(
    titles: list[str],
    *,
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    retry_sleep_seconds: float = 10.0,
) -> dict[str, str]:
    if not titles:
        return {}
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "redirects": "1",
        "titles": "|".join(titles),
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )
    payload_obj = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload_obj = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            if exc.code not in {429, 503} or attempt >= max_retries:
                print(f"warning: Wikipedia summary batch failed: {exc}")
                return {}
            retry_after = parse_retry_after(exc.headers.get("Retry-After"))
            sleep_for = retry_after if retry_after is not None else retry_sleep_seconds * (attempt + 1)
            print(f"warning: Wikipedia summary batch got HTTP {exc.code}; retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
        except (URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                print(f"warning: Wikipedia summary batch failed: {exc}")
                return {}
            sleep_for = retry_sleep_seconds * (attempt + 1)
            print(f"warning: Wikipedia summary batch failed: {exc}; retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    if payload_obj is None:
        return {}

    query = payload_obj.get("query", {})
    title_aliases = {title: title for title in titles}
    for item in query.get("normalized", []):
        if "from" in item and "to" in item:
            title_aliases[item["from"]] = item["to"]
    for item in query.get("redirects", []):
        if "from" in item and "to" in item:
            title_aliases[item["from"]] = item["to"]

    extracts_by_page_title = {}
    for page in query.get("pages", {}).values():
        title = page.get("title")
        extract = normalize_abstract(page.get("extract", ""))
        if title and extract:
            extracts_by_page_title[title] = extract

    summaries = {}
    for original_title in titles:
        resolved_title = resolve_title_alias(original_title, title_aliases)
        summary = extracts_by_page_title.get(resolved_title, "")
        if summary:
            summaries[original_title] = summary
    return summaries


def parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return max(parsed, 0.0)


def resolve_title_alias(title: str, aliases: dict[str, str]) -> str:
    seen = set()
    current = title
    while current in aliases and current not in seen:
        seen.add(current)
        next_title = aliases[current]
        if next_title == current:
            break
        current = next_title
    return current


def dbpedia_uri_to_wikipedia_title(dbpedia_uri: str) -> str:
    return urllib.parse.unquote(dbpedia_uri.rstrip("/").rsplit("/", maxsplit=1)[-1])


def normalize_abstract(value: object) -> str:
    return " ".join(str(value).split())


def load_cache(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object cache: {path}")
    return {str(key): str(item) for key, item in value.items() if item}


def write_cache(path: Path | None, abstracts_by_dbpedia_uri: dict[str, str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(abstracts_by_dbpedia_uri, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_path)
    cache_path = Path(args.cache_json) if args.cache_json else None

    entity_rows = load_entity_rows(dataset_dir)
    if args.limit is not None:
        entity_rows = entity_rows[: args.limit]
    ent_links = load_ent_links(dataset_dir)
    dbpedia_uris = [ent_links[row["wikidata_uri"]] for row in entity_rows]

    abstracts = load_cache(cache_path)
    missing_uris = [uri for uri in dbpedia_uris if uri not in abstracts]
    if missing_uris and not args.skip_dbpedia:
        abstracts.update(
            fetch_dbpedia_abstracts(
                missing_uris,
                endpoint=args.endpoint,
                batch_size=args.batch_size,
                timeout_seconds=args.timeout_seconds,
                sleep_seconds=args.sleep_seconds,
            )
        )
    still_missing_uris = [uri for uri in dbpedia_uris if uri not in abstracts]
    if args.fallback_source == "wikipedia-summary" and still_missing_uris:
        abstracts.update(
                fetch_wikipedia_summaries(
                    still_missing_uris,
                    batch_size=args.batch_size,
                    timeout_seconds=args.timeout_seconds,
                    sleep_seconds=args.sleep_seconds,
                    max_retries=args.max_retries,
                    retry_sleep_seconds=args.retry_sleep_seconds,
                    cache_path=cache_path,
                    abstracts_by_dbpedia_uri=abstracts,
                )
            )
    write_cache(cache_path, abstracts)

    summary = export_abstract_rows(
        dataset_dir=dataset_dir,
        output_path=output_path,
        abstracts_by_dbpedia_uri=abstracts,
        limit=args.limit,
    )
    summary["cache_json"] = "" if cache_path is None else str(cache_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
