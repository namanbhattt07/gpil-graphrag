"""
validate_index.py

WHAT THIS SCRIPT DOES (plain language)
Reads a completed GraphRAG index's output/ folder (entities.parquet,
relationships.parquet, text_units.parquet, communities.parquet,
community_reports.parquet, documents.parquet) and checks it for the
specific data-quality problems a GPT-4o-mini pilot run was found to
produce: a document whose State entity never got extracted, Franchise
entities that got extracted with inconsistent underscore/space spelling
so the same real-world franchise ends up as two nodes, and Category
entities that are misspelled or have a state name incorrectly baked into
them. It also checks more generic structural invariants (every id a
table references actually exists, every community has a report, etc.)
so a differently-shaped future bug doesn't slip through silently.

This script never writes anything -- it only reads --input-dir (the
GraphRAG output), --source-dir (the original Phase 4 .txt documents,
used to verify a document's actual State: line without trusting the
manifest alone), and --manifest (graphrag_docs_manifest.csv), and prints
a report. canonicalize_index.py imports the check functions here
directly and uses the same violation list as its worklist, so the two
scripts can never quietly disagree about what counts as a problem.

WHY IT LOOKS THE WAY IT DOES
Every check is deterministic -- no LLM call, no fuzzy matching. The
Category/Franchise checks resolve extracted titles against this
project's own existing canonical vocabulary (INDIAN_STATES,
CATEGORY_FRANCHISES in src/data_gen/generate_synthetic_data.py, the
same source of truth compute_kpis.py already imports from) rather than
inventing a new schema file or guessing that two similar-looking titles
must be the same thing. An extracted title that doesn't resolve against
that vocabulary is reported as unresolved, never silently merged.

The missing-State check does not trust graphrag_docs_manifest.csv by
itself: it cross-checks the state name implied by the document's own
filename against the state name written in the actual source document's
first "State: ..." line (a fixed, deterministic format Phase 4 always
writes -- see build_documents.py). Only when those two agree is the
manifest consulted; a manifest that disagrees with the two directly
verifiable sources is reported as its own violation
(manifest_source_mismatch), never blamed on GraphRAG extraction.

Checks run in a fixed order (see CHECK_ORDER below / validate()):
  1. schema / required-column validation
  2. manifest-source consistency
  3. referential integrity (list-column linkage)
  4. canonical vocabulary validation (Category, Franchise, forbidden types)
  5. duplicate entity detection (Category, Franchise)
  6. required State entity validation
  7. relationship integrity (relationships.parquet source/target)
  8. community/report consistency
  9. final summary

Run with (from the project root, venv activated):

    python -m src.graph.validate_index \\
        --input-dir data/graphrag_index/pilot_run_mini/output \\
        --source-dir data/graphrag_index/input_pilot \\
        --manifest data/graphrag_docs_manifest.csv

Exits non-zero if any violation is found (pass --exit-zero to always
exit 0, for inspection-only use that should never block a script).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.data_gen.generate_synthetic_data import CATEGORY_FRANCHISES, INDIAN_STATES

OUTPUT_TABLES = [
    "entities",
    "relationships",
    "text_units",
    "communities",
    "community_reports",
    "documents",
]

# The column sets actually observed in a completed run's output. Checks
# below assume whichever tables ARE present have at least these columns.
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "entities": {
        "id",
        "human_readable_id",
        "title",
        "type",
        "description",
        "text_unit_ids",
        "frequency",
    },
    "relationships": {
        "id",
        "human_readable_id",
        "source",
        "target",
        "description",
        "text_unit_ids",
        "weight",
    },
    "text_units": {"id", "document_id", "entity_ids", "relationship_ids"},
    "communities": {"id", "community", "level", "entity_ids", "relationship_ids"},
    "community_reports": {"id", "community", "full_content"},
    "documents": {"id", "title"},
}

FORBIDDEN_ENTITY_TYPES = {"ZONE", "CHANNEL"}

# Phase 4 (build_documents.py) always names files "{State_with_underscores}
# _{YYYY-MM}.txt" and always opens the document with a literal "State: name"
# line -- both are fixed, deterministic formats we can parse without an LLM.
FILENAME_RE = re.compile(r"^(?P<state>.+)_(?P<year>\d{4})-(?P<month>\d{2})\.txt$")
CONTENT_STATE_RE = re.compile(r"^State:\s*(.+?)\s*$", re.MULTILINE)

def _normalize_title(title: str) -> str:
    """Uppercase, fold runs of underscores/whitespace to a single space.
    Used only to MATCH a raw extracted title against a closed vocabulary
    -- never used to invent a new canonical name that isn't already in
    that vocabulary."""
    return re.sub(r"[_\s]+", " ", str(title).strip()).upper()


def _canonical_category_by_normalized() -> dict[str, str]:
    return {_normalize_title(c): c for c in CATEGORY_FRANCHISES}


# Known typos/variants that fold to a canonical Category despite not being
# an exact or normalized match. Deliberately a short, explicit, auditable
# list -- never guessed or fuzzy-matched. Empty entries are fine; this is
# where a human adds an alias after seeing an unknown_franchise_entity /
# invalid_category_entity violation they've confirmed is a real typo.
#
# Values are looked up from CATEGORY_FRANCHISES itself (never re-typed as
# a string literal) so an alias can't accidentally resolve to a
# differently-cased spelling than the real canonical entity -- e.g.
# "FERRO" must resolve to the exact same "Ferrero" string that "Ferrero"
# itself resolves to, or the two would never merge into one entity.
CATEGORY_ALIASES: dict[str, str] = {
    "FERRO": _canonical_category_by_normalized()["FERRERO"],
}

FRANCHISE_ALIASES: dict[str, str] = {}


def resolve_category_title(raw_title: str) -> tuple[str | None, str]:
    """Resolve an extracted Category entity title to one of the 4
    canonical categories (GPI, IPM, Ferrero, Candy), or (None,
    "unresolved") if it doesn't match any of the three deterministic
    rules below. Never fuzzy-matches.

    Resolution order:
      1. exact match after normalization (handles case-only variants)
      2. explicit alias table (known typos, e.g. FERRO -> FERRERO)
      3. state-name-prefix pattern (e.g. "UTTARAKHAND CANDY" -> "CANDY")
         -- only fires when the prefix is a REAL state name from
         INDIAN_STATES and the suffix is a REAL canonical category, so
         this can never invent a mapping for an unrelated title.
    """
    normalized = _normalize_title(raw_title)
    canonical_by_normalized = _canonical_category_by_normalized()

    if normalized in canonical_by_normalized:
        return canonical_by_normalized[normalized], "exact"

    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized], "alias"

    for state in INDIAN_STATES:
        prefix = _normalize_title(state)
        if normalized.startswith(prefix + " "):
            suffix = normalized[len(prefix) + 1 :]
            if suffix in canonical_by_normalized:
                return canonical_by_normalized[suffix], "state_prefix"

    return None, "unresolved"


def _canonical_franchise_lookup() -> dict[str, str]:
    """normalized title -> canonical title, built from CATEGORY_FRANCHISES
    (the same vocabulary source already used for Category)."""
    lookup: dict[str, str] = {}
    for franchises in CATEGORY_FRANCHISES.values():
        for name in franchises:
            lookup[_normalize_title(name)] = name
    return lookup


def resolve_franchise_title(raw_title: str) -> tuple[str | None, str]:
    """Resolve an extracted Franchise entity title against the project's
    closed Franchise vocabulary (CATEGORY_FRANCHISES, flattened). A
    title that folds to the same normalized form as a real franchise
    (e.g. "GPI FRANCHISE 1" / "GPI_Franchise_1") resolves to that
    franchise's canonical spelling. A title that doesn't fold to
    anything in the vocabulary is deliberately left unresolved --
    "looks similar" is never enough to merge two entities."""
    normalized = _normalize_title(raw_title)
    lookup = _canonical_franchise_lookup()

    if normalized in lookup:
        return lookup[normalized], "exact_or_normalized"

    if normalized in FRANCHISE_ALIASES:
        return FRANCHISE_ALIASES[normalized], "alias"

    return None, "unresolved"


def _to_list(value) -> list:
    """Normalize a parquet list-column cell to a plain Python list.
    Pandas round-trips list-typed parquet columns as numpy arrays, and
    `array or []` raises ValueError (ambiguous truth value) instead of
    doing what a plain list/None would -- this is the single place that
    distinction gets handled, used everywhere a list-column cell is read."""
    if value is None:
        return []
    if isinstance(value, float):  # NaN
        return []
    return list(value)


def _violation(check: str, severity: str, subject: str, detail: str) -> dict:
    return {"check": check, "severity": severity, "subject": subject, "detail": detail}


def load_output_tables(input_dir: Path) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Load the 6 GraphRAG output parquet tables. A missing file becomes
    a missing_output_file violation instead of a crash, so the rest of
    validate() can still report on whatever tables ARE present."""
    tables: dict[str, pd.DataFrame] = {}
    violations: list[dict] = []
    for name in OUTPUT_TABLES:
        path = input_dir / f"{name}.parquet"
        if not path.exists():
            violations.append(
                _violation("missing_output_file", "critical", name, f"{path} does not exist")
            )
            continue
        tables[name] = pd.read_parquet(path)
    return tables, violations


def check_schema(tables: dict[str, pd.DataFrame]) -> list[dict]:
    violations = []
    for name, required_cols in REQUIRED_COLUMNS.items():
        if name not in tables:
            continue  # already flagged by load_output_tables
        missing = required_cols - set(tables[name].columns)
        if missing:
            violations.append(
                _violation(
                    "missing_required_column",
                    "critical",
                    name,
                    f"missing columns: {sorted(missing)}",
                )
            )
    return violations


def _state_from_filename(filename: str) -> str | None:
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    return match.group("state").replace("_", " ")


def _state_from_content(source_dir: Path, filename: str) -> str | None:
    path = source_dir / filename
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = CONTENT_STATE_RE.search(text)
    return match.group(1) if match else None


def check_manifest_source_consistency(
    documents_df: pd.DataFrame, manifest_df: pd.DataFrame, source_dir: Path
) -> tuple[list[dict], dict[str, str]]:
    """Returns (violations, verified_state_by_filename). A filename only
    lands in verified_state_by_filename when the state implied by its
    filename and the state written in its actual source content agree --
    that agreed value is the only thing check_missing_state_entity ever
    checks the graph against; the manifest is cross-checked against it
    but never used as the ground truth on its own."""
    violations: list[dict] = []
    verified: dict[str, str] = {}
    manifest_lookup = dict(zip(manifest_df["filename"], manifest_df["state_name"]))

    for filename in documents_df["title"]:
        filename_state = _state_from_filename(filename)
        content_state = _state_from_content(source_dir, filename)

        if filename_state is None:
            violations.append(
                _violation(
                    "unparsable_filename",
                    "medium",
                    filename,
                    "filename doesn't match the expected {State}_{YYYY-MM}.txt pattern",
                )
            )
            continue
        if content_state is None:
            violations.append(
                _violation(
                    "unreadable_source_document",
                    "medium",
                    filename,
                    f"no readable source document found under {source_dir} "
                    "(or it has no 'State: ...' line)",
                )
            )
            continue

        if filename_state != content_state:
            violations.append(
                _violation(
                    "filename_content_mismatch",
                    "high",
                    filename,
                    f"filename implies state={filename_state!r}, but the document's "
                    f"own 'State:' line says {content_state!r}",
                )
            )
            continue

        verified[filename] = filename_state
        manifest_state = manifest_lookup.get(filename)
        if manifest_state is not None and manifest_state != filename_state:
            violations.append(
                _violation(
                    "manifest_source_mismatch",
                    "high",
                    filename,
                    f"manifest says state={manifest_state!r}, but filename+content "
                    f"agree on {filename_state!r}",
                )
            )

    return violations, verified


def check_missing_state_entity(
    tables: dict[str, pd.DataFrame], verified_state_by_filename: dict[str, str]
) -> list[dict]:
    """Only evaluated for documents where filename and content already
    agreed on the state (see check_manifest_source_consistency) -- a
    reliable target to check the graph against. Documents whose sources
    disagreed are not guessed at here; they're already flagged there."""
    entities = tables["entities"]
    documents = tables["documents"]
    state_titles = set(entities.loc[entities["type"].str.upper() == "STATE", "title"])
    known_doc_titles = set(documents["title"])

    violations = []
    for filename, verified_state in verified_state_by_filename.items():
        if filename not in known_doc_titles:
            continue
        expected_title = verified_state.upper()
        if expected_title not in state_titles:
            violations.append(
                _violation(
                    "missing_state_entity",
                    "high",
                    filename,
                    f"no STATE entity titled {expected_title!r} found in entities.parquet",
                )
            )
    return violations


def check_referential_integrity(tables: dict[str, pd.DataFrame]) -> list[dict]:
    """Every id referenced inside text_units.parquet / communities.parquet
    list columns must resolve to a real row in entities.parquet /
    relationships.parquet."""
    violations = []
    entity_ids = set(tables["entities"]["id"])
    relationship_ids = set(tables["relationships"]["id"])

    for table_name in ("text_units", "communities"):
        df = tables[table_name]
        for _, row in df.iterrows():
            for eid in _to_list(row["entity_ids"]):
                if eid not in entity_ids:
                    violations.append(
                        _violation(
                            "orphaned_reference",
                            "medium",
                            f"{table_name}:{row['id']}",
                            f"entity_ids references unknown entity id {eid}",
                        )
                    )
            for rid in _to_list(row["relationship_ids"]):
                if rid not in relationship_ids:
                    violations.append(
                        _violation(
                            "orphaned_reference",
                            "medium",
                            f"{table_name}:{row['id']}",
                            f"relationship_ids references unknown relationship id {rid}",
                        )
                    )
    return violations


def check_dangling_relationships(tables: dict[str, pd.DataFrame]) -> list[dict]:
    """relationships.parquet's own source/target columns (entity titles,
    not ids) must resolve to a real entities.parquet title -- distinct
    from check_referential_integrity, which checks the list-column
    linkage in text_units/communities instead."""
    entity_titles = set(tables["entities"]["title"])
    relationships = tables["relationships"]
    bad = relationships[
        ~relationships["source"].isin(entity_titles) | ~relationships["target"].isin(entity_titles)
    ]
    return [
        _violation(
            "dangling_relationship",
            "medium",
            row["id"],
            f"source={row['source']!r} target={row['target']!r}, one or both not "
            "found among entities.parquet titles",
        )
        for _, row in bad.iterrows()
    ]


def check_forbidden_entity_types(tables: dict[str, pd.DataFrame]) -> list[dict]:
    entities = tables["entities"]
    bad = entities[entities["type"].str.upper().isin(FORBIDDEN_ENTITY_TYPES)]
    return [
        _violation(
            "forbidden_entity_type",
            "low",
            row["title"],
            f"entity type {row['type']!r} is explicitly forbidden by the extraction prompt",
        )
        for _, row in bad.iterrows()
    ]


def check_category_vocabulary_and_duplicates(tables: dict[str, pd.DataFrame]) -> list[dict]:
    """Mirrors check_franchise_vocabulary_and_duplicates: a Category
    title that doesn't resolve against the canonical 4-category
    vocabulary (even after alias/state-prefix rules) is
    invalid_category_entity; multiple Category entities that DO resolve
    to the same canonical category (e.g. the real pilot's "FERRERO",
    "FERRO", and "UTTARAKHAND FERRERO" all resolving to "Ferrero") are
    duplicate_category_entity -- canonicalize_index.py's merge step
    handles both Category and Franchise duplicates identically, so the
    validator should name both explicitly rather than only surfacing
    the Franchise case."""
    entities = tables["entities"]
    categories = entities[entities["type"].str.upper() == "CATEGORY"]

    violations = []
    resolved: dict[str, list[str]] = {}
    for _, row in categories.iterrows():
        canonical, _reason = resolve_category_title(row["title"])
        if canonical is None:
            violations.append(
                _violation(
                    "invalid_category_entity",
                    "high",
                    row["title"],
                    "does not resolve against the canonical 4-category vocabulary "
                    "(CATEGORY_FRANCHISES keys), even after alias/state-prefix rules",
                )
            )
        else:
            resolved.setdefault(canonical, []).append(row["title"])

    for canonical, titles in resolved.items():
        if len(titles) > 1:
            violations.append(
                _violation(
                    "duplicate_category_entity",
                    "medium",
                    canonical,
                    f"{len(titles)} entities resolve to this canonical category: {titles}",
                )
            )
    return violations


def check_franchise_vocabulary_and_duplicates(tables: dict[str, pd.DataFrame]) -> list[dict]:
    entities = tables["entities"]
    franchises = entities[entities["type"].str.upper() == "FRANCHISE"]

    violations = []
    resolved: dict[str, list[str]] = {}
    for _, row in franchises.iterrows():
        canonical, _reason = resolve_franchise_title(row["title"])
        if canonical is None:
            violations.append(
                _violation(
                    "unknown_franchise_entity",
                    "high",
                    row["title"],
                    "does not resolve against the canonical Franchise vocabulary "
                    "(CATEGORY_FRANCHISES) -- not auto-merged; needs an explicit "
                    "alias in FRANCHISE_ALIASES if it's a confirmed known variant",
                )
            )
        else:
            resolved.setdefault(canonical, []).append(row["title"])

    for canonical, titles in resolved.items():
        if len(titles) > 1:
            violations.append(
                _violation(
                    "duplicate_franchise_entity",
                    "medium",
                    canonical,
                    f"{len(titles)} entities resolve to this canonical franchise: {titles}",
                )
            )
    return violations


def check_community_report_consistency(tables: dict[str, pd.DataFrame]) -> list[dict]:
    n_communities = len(tables["communities"])
    n_reports = len(tables["community_reports"])
    if n_communities != n_reports:
        return [
            _violation(
                "community_report_mismatch",
                "high",
                "communities vs community_reports",
                f"{n_communities} communities but {n_reports} community reports",
            )
        ]
    return []


def validate(input_dir: Path, source_dir: Path, manifest_path: Path) -> list[dict]:
    """Runs every check in the fixed order documented in the module
    docstring and returns the full accumulated violation list. Each
    check is individually guarded so a missing/malformed table produces
    a violation instead of crashing the whole run -- a validator run
    should always be as informative as possible, never an unhandled
    traceback."""
    violations: list[dict] = []

    # 1. schema / required-column validation
    tables, load_violations = load_output_tables(input_dir)
    violations += load_violations
    violations += check_schema(tables)

    manifest_df = pd.read_csv(manifest_path)

    # 2. manifest-source consistency
    verified_state_by_filename: dict[str, str] = {}
    if "documents" in tables:
        ms_violations, verified_state_by_filename = check_manifest_source_consistency(
            tables["documents"], manifest_df, source_dir
        )
        violations += ms_violations

    # 3. referential integrity (list-column linkage)
    if {"text_units", "communities", "entities", "relationships"} <= tables.keys():
        violations += check_referential_integrity(tables)

    # 4. canonical vocabulary validation
    if "entities" in tables:
        violations += check_category_vocabulary_and_duplicates(tables)
        violations += check_franchise_vocabulary_and_duplicates(tables)
        violations += check_forbidden_entity_types(tables)

    # 5. duplicate entity detection is folded into check 4 above (it needs
    # the same vocabulary-resolution pass, so both run together to avoid
    # doing that resolution twice)

    # 6. required State entity validation
    if {"entities", "documents"} <= tables.keys() and verified_state_by_filename:
        violations += check_missing_state_entity(tables, verified_state_by_filename)

    # 7. relationship integrity / validity
    if {"entities", "relationships"} <= tables.keys():
        violations += check_dangling_relationships(tables)

    # 8. community/report consistency
    if {"communities", "community_reports"} <= tables.keys():
        violations += check_community_report_consistency(tables)

    # 9. final summary happens in print_report(), not here
    return violations


def print_report(violations: list[dict]) -> None:
    if not violations:
        print("VALIDATION PASSED -- 0 violations found.")
        return

    print(f"VALIDATION FOUND {len(violations)} violation(s):\n")
    by_check: dict[str, list[dict]] = {}
    for violation in violations:
        by_check.setdefault(violation["check"], []).append(violation)

    for check, items in sorted(by_check.items()):
        print(f"  {check} ({len(items)}):")
        for item in items[:10]:
            print(f"    [{item['severity']}] {item['subject']}: {item['detail']}")
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="GraphRAG output/ directory to validate"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Original Phase 4 .txt source documents directory",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="graphrag_docs_manifest.csv"
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="Always exit 0 regardless of violations found (inspection-only use)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to also write the violation list as JSON",
    )
    args = parser.parse_args()

    violations = validate(args.input_dir, args.source_dir, args.manifest)
    print_report(violations)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(violations, indent=2))

    if violations and not args.exit_zero:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
