"""
Tests for src/graph/validate_index.py.

Like tests/test_build_documents.py, these are self-contained: rather
than depending on a real GraphRAG run's output/ folder existing on
disk, each test builds a small, hand-crafted GraphRAG-shaped corpus
(entities/relationships/text_units/communities/community_reports/
documents.parquet, plus real source .txt files and a real manifest.csv)
in a tmp_path and writes it to disk with pandas exactly the way a real
GraphRAG run would, then calls validate() against it -- exercising the
exact same read path production runs use, not mocks.

_base_corpus() returns a complete, bug-free 2-document corpus (Bihar,
Punjab) as plain Python dict/lists; each test mutates exactly the piece
it needs before calling write_corpus(). This keeps every test scenario
independent and easy to reason about instead of one large parametrized
fixture with a big flag surface.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.graph.validate_index import validate


def _base_corpus() -> dict:
    """A complete, violation-free 2-document corpus."""
    docs = [
        ("Bihar_2025-10.txt", "Bihar", "October 2025"),
        ("Punjab_2026-02.txt", "Punjab", "February 2026"),
    ]

    entities: list[dict] = []
    relationships: list[dict] = []
    text_units: list[dict] = []
    communities: list[dict] = []
    community_reports: list[dict] = []
    documents: list[dict] = []
    manifest_rows: list[dict] = []
    source_texts: dict[str, str] = {}

    hrid_counter = {"entity": 0, "rel": 0}

    def new_entity(title, etype, description, text_unit_ids):
        i = hrid_counter["entity"]
        hrid_counter["entity"] += 1
        entity_id = f"e{i}"
        entities.append(
            {
                "id": entity_id,
                "human_readable_id": i,
                "title": title,
                "type": etype,
                "description": description,
                "text_unit_ids": text_unit_ids,
                "frequency": 1,
                "degree": 1,
            }
        )
        return entity_id

    def new_relationship(source, target, description, text_unit_ids):
        i = hrid_counter["rel"]
        hrid_counter["rel"] += 1
        rel_id = f"r{i}"
        relationships.append(
            {
                "id": rel_id,
                "human_readable_id": i,
                "source": source,
                "target": target,
                "description": description,
                "text_unit_ids": text_unit_ids,
                "weight": 1.0,
                "combined_degree": 1,
            }
        )
        return rel_id

    for filename, state, period in docs:
        doc_id = f"doc-{state.lower()}"
        tu_id = f"tu-{state.lower()}"
        documents.append({"id": doc_id, "human_readable_id": len(documents), "title": filename})
        manifest_rows.append(
            {"filename": filename, "state_name": state, "month": period[-4:] + "-" + period[:2]}
        )
        source_texts[filename] = f"State: {state}\nPeriod: {period}\n\nBody text about {state}.\n"

        entity_ids_for_doc = []
        relationship_ids_for_doc = []

        doc_entity_id = new_entity(
            f"{filename.upper()}", "DOCUMENT", f"Source document for {state}", [tu_id]
        )
        entity_ids_for_doc.append(doc_entity_id)

        state_entity_id = new_entity(
            state.upper(),
            "STATE",
            f"An Indian state; this document reports its S&D performance for {period}",
            [tu_id],
        )
        entity_ids_for_doc.append(state_entity_id)

        distributor_title = f"{state} Distributors Co"
        distributor_entity_id = new_entity(
            distributor_title, "DISTRIBUTOR", f"{distributor_title} operates in {state}", [tu_id]
        )
        entity_ids_for_doc.append(distributor_entity_id)

        obs_title = f"{state.upper()} PRODUCTIVITY {period.upper()}"
        obs_entity_id = new_entity(
            obs_title,
            "OBSERVATION",
            f"State-level Productivity for {state} in {period} was 80.0%",
            [tu_id],
        )
        entity_ids_for_doc.append(obs_entity_id)

        serves_id = new_relationship(
            distributor_title,
            state.upper(),
            f"SERVES: {distributor_title} operates in {state}",
            [tu_id],
        )
        relationship_ids_for_doc.append(serves_id)

        has_obs_id = new_relationship(
            state.upper(),
            obs_title,
            f"HAS_OBSERVATION: {state} has a measured Productivity fact",
            [tu_id],
        )
        relationship_ids_for_doc.append(has_obs_id)

        text_units.append(
            {
                "id": tu_id,
                "human_readable_id": len(text_units),
                "text": "body",
                "n_tokens": 10,
                "document_id": doc_id,
                "entity_ids": entity_ids_for_doc,
                "relationship_ids": relationship_ids_for_doc,
                "covariate_ids": [],
            }
        )

        community_index = len(communities)
        communities.append(
            {
                "id": f"comm-{state.lower()}",
                "human_readable_id": community_index,
                "community": community_index,
                "level": 0,
                "parent": -1,
                "children": [],
                "title": f"Community {community_index}",
                "entity_ids": entity_ids_for_doc,
                "relationship_ids": relationship_ids_for_doc,
                "text_unit_ids": [tu_id],
                "period": "2026-08-14",
                "size": len(entity_ids_for_doc),
            }
        )
        community_reports.append(
            {
                "id": f"cr-{state.lower()}",
                "human_readable_id": community_index,
                "community": community_index,
                "level": 0,
                "parent": -1,
                "children": [],
                "title": f"Report {state}",
                "summary": "summary",
                "full_content": f"Report about {state}",
                "rank": 5.0,
                "rating_explanation": "",
                "findings": [],
                "full_content_json": "{}",
                "period": "2026-08-14",
                "size": 1,
            }
        )

    # One shared, correctly-spelled Category and Franchise entity, tied to
    # the Bihar text unit, so a clean corpus has zero vocabulary violations.
    new_entity("Candy", "CATEGORY", "Product category", ["tu-bihar"])
    new_entity("GPI_Franchise_1", "FRANCHISE", "Franchise within the GPI category", ["tu-bihar"])

    return {
        "entities": entities,
        "relationships": relationships,
        "text_units": text_units,
        "communities": communities,
        "community_reports": community_reports,
        "documents": documents,
        "manifest_rows": manifest_rows,
        "source_texts": source_texts,
    }


def remove_state_entity(corpus: dict, state_title: str) -> dict:
    """Mutates corpus in place to simulate a Maharashtra-style extraction
    gap: the STATE entity for `state_title` (e.g. "BIHAR") never got
    created, and neither did anything that referenced it (its SERVES/
    HAS_OBSERVATION relationships). Removes the entity, every
    relationship touching it, and scrubs their ids out of every
    text_units/communities list column that referenced them -- so the
    resulting fixture is internally consistent (no orphaned_reference
    violations), the same way real extraction simply never emitted
    those rows in the first place rather than deleting them after the
    fact. Returns corpus for chaining."""
    removed_entity_ids = {
        e["id"] for e in corpus["entities"] if e["type"] == "STATE" and e["title"] == state_title
    }
    corpus["entities"] = [e for e in corpus["entities"] if e["id"] not in removed_entity_ids]

    # relationships.parquet stores entity titles in source/target, not ids
    removed_relationship_ids = {
        r["id"]
        for r in corpus["relationships"]
        if r["source"] == state_title or r["target"] == state_title
    }
    corpus["relationships"] = [
        r for r in corpus["relationships"] if r["id"] not in removed_relationship_ids
    ]

    for table_name in ("text_units", "communities"):
        for row in corpus[table_name]:
            row["entity_ids"] = [i for i in row["entity_ids"] if i not in removed_entity_ids]
            row["relationship_ids"] = [
                i for i in row["relationship_ids"] if i not in removed_relationship_ids
            ]

    return corpus


def write_corpus(tmp_path: Path, corpus: dict) -> dict:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    source_dir = tmp_path / "input_pilot"
    source_dir.mkdir()

    pd.DataFrame(corpus["entities"]).to_parquet(output_dir / "entities.parquet", index=False)
    pd.DataFrame(corpus["relationships"]).to_parquet(output_dir / "relationships.parquet", index=False)
    pd.DataFrame(corpus["text_units"]).to_parquet(output_dir / "text_units.parquet", index=False)
    pd.DataFrame(corpus["communities"]).to_parquet(output_dir / "communities.parquet", index=False)
    pd.DataFrame(corpus["community_reports"]).to_parquet(
        output_dir / "community_reports.parquet", index=False
    )
    pd.DataFrame(corpus["documents"]).to_parquet(output_dir / "documents.parquet", index=False)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(corpus["manifest_rows"]).to_csv(manifest_path, index=False)

    for filename, text in corpus["source_texts"].items():
        (source_dir / filename).write_text(text, encoding="utf-8")

    return {"input_dir": output_dir, "source_dir": source_dir, "manifest_path": manifest_path}


def _checks(violations: list[dict]) -> set[str]:
    return {v["check"] for v in violations}


def _run(tmp_path, corpus):
    paths = write_corpus(tmp_path, corpus)
    return validate(paths["input_dir"], paths["source_dir"], paths["manifest_path"])


# ---------------------------------------------------------------------------
# Clean corpus
# ---------------------------------------------------------------------------


def test_clean_corpus_has_zero_violations(tmp_path):
    violations = _run(tmp_path, _base_corpus())
    assert violations == []


# ---------------------------------------------------------------------------
# Missing State entity (Maharashtra-style)
# ---------------------------------------------------------------------------


def test_missing_state_entity_detected(tmp_path):
    corpus = remove_state_entity(_base_corpus(), "BIHAR")
    violations = _run(tmp_path, corpus)
    checks = _checks(violations)
    assert "missing_state_entity" in checks
    bihar_violations = [v for v in violations if v["check"] == "missing_state_entity"]
    assert bihar_violations[0]["subject"] == "Bihar_2025-10.txt"
    assert "manifest_source_mismatch" not in checks
    assert "filename_content_mismatch" not in checks


# ---------------------------------------------------------------------------
# Manifest vs. filename/content consistency
# ---------------------------------------------------------------------------


def test_manifest_source_mismatch_detected_and_missing_state_not_raised(tmp_path):
    """Manifest disagrees with filename+content (which still agree with
    each other) -- this must be reported as manifest_source_mismatch,
    and must NOT also raise missing_state_entity for the same document,
    since blaming GraphRAG extraction here would be wrong (Case B from
    the approved plan)."""
    corpus = _base_corpus()
    for row in corpus["manifest_rows"]:
        if row["filename"] == "Bihar_2025-10.txt":
            row["state_name"] = "Jharkhand"  # manifest now disagrees with filename+content
    violations = _run(tmp_path, corpus)
    checks = _checks(violations)
    assert "manifest_source_mismatch" in checks
    mismatch = [v for v in violations if v["check"] == "manifest_source_mismatch"][0]
    assert mismatch["subject"] == "Bihar_2025-10.txt"
    assert "Jharkhand" in mismatch["detail"]
    assert "Bihar" in mismatch["detail"]
    # Bihar's actual STATE entity is present and correct -- must not be
    # flagged missing just because the manifest was wrong.
    assert not any(
        v["check"] == "missing_state_entity" and v["subject"] == "Bihar_2025-10.txt"
        for v in violations
    )


def test_filename_content_mismatch_detected(tmp_path):
    """Filename says one state, the document's own 'State:' line says
    another -- a deeper data problem than a manifest typo, reported
    separately, and missing_state_entity is skipped for that document
    entirely since there's no reliable expected state to check against."""
    corpus = _base_corpus()
    corpus["source_texts"]["Punjab_2026-02.txt"] = (
        "State: Haryana\nPeriod: February 2026\n\nBody text.\n"
    )
    violations = _run(tmp_path, corpus)
    checks = _checks(violations)
    assert "filename_content_mismatch" in checks
    mismatch = [v for v in violations if v["check"] == "filename_content_mismatch"][0]
    assert mismatch["subject"] == "Punjab_2026-02.txt"
    assert not any(
        v["check"] == "missing_state_entity" and v["subject"] == "Punjab_2026-02.txt"
        for v in violations
    )


# ---------------------------------------------------------------------------
# Franchise vocabulary
# ---------------------------------------------------------------------------


def test_duplicate_franchise_with_known_canonical_mapping(tmp_path):
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-dup",
            "human_readable_id": 999,
            "title": "GPI FRANCHISE 1",
            "type": "FRANCHISE",
            "description": "underscore/space variant of the same franchise",
            "text_unit_ids": ["tu-punjab"],
            "frequency": 1,
            "degree": 1,
        }
    )
    violations = _run(tmp_path, corpus)
    checks = _checks(violations)
    assert "duplicate_franchise_entity" in checks
    assert "unknown_franchise_entity" not in checks


def test_unknown_franchise_not_reported_as_duplicate(tmp_path):
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-unknown",
            "human_readable_id": 999,
            "title": "SOMETHING_UNKNOWN",
            "type": "FRANCHISE",
            "description": "not a real franchise",
            "text_unit_ids": ["tu-bihar"],
            "frequency": 1,
            "degree": 1,
        }
    )
    violations = _run(tmp_path, corpus)
    checks = _checks(violations)
    assert "unknown_franchise_entity" in checks
    assert "duplicate_franchise_entity" not in checks
    unknown = [v for v in violations if v["check"] == "unknown_franchise_entity"][0]
    assert unknown["subject"] == "SOMETHING_UNKNOWN"


# ---------------------------------------------------------------------------
# Category vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_title", ["Kandy", "Nowhereland Candy", "XYZ"])
def test_invalid_category_entity_detected(tmp_path, bad_title):
    """Titles that don't resolve via any of the 3 deterministic rules
    (exact match, alias table, real-state-prefix) must be flagged --
    unlike "Ferro" (a known alias) or "Uttarakhand Candy" (a real state
    prefixing a real category), which this validator is specifically
    designed to resolve correctly rather than flag (see
    test_known_category_variants_resolve_without_violation below)."""
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-badcat",
            "human_readable_id": 999,
            "title": bad_title,
            "type": "CATEGORY",
            "description": "malformed category",
            "text_unit_ids": ["tu-bihar"],
            "frequency": 1,
            "degree": 1,
        }
    )
    violations = _run(tmp_path, corpus)
    assert "invalid_category_entity" in _checks(violations)


@pytest.mark.parametrize("known_variant", ["Ferro", "Uttarakhand Candy", "FERRERO", "candy"])
def test_known_category_variants_resolve_without_violation(tmp_path, known_variant):
    """The two real bugs found in the pilot (FERRO typo, state-prefixed
    category) must resolve cleanly, not be flagged -- this is what makes
    them fixable by canonicalize_index.py without a human-added alias."""
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-knowncat",
            "human_readable_id": 999,
            "title": known_variant,
            "type": "CATEGORY",
            "description": "known variant",
            "text_unit_ids": ["tu-bihar"],
            "frequency": 1,
            "degree": 1,
        }
    )
    violations = _run(tmp_path, corpus)
    assert "invalid_category_entity" not in _checks(violations)


# ---------------------------------------------------------------------------
# Forbidden entity types
# ---------------------------------------------------------------------------


def test_forbidden_zone_entity_detected(tmp_path):
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-zone",
            "human_readable_id": 999,
            "title": "BIHAR ZONE 1",
            "type": "ZONE",
            "description": "should never be extracted per the prompt",
            "text_unit_ids": ["tu-bihar"],
            "frequency": 1,
            "degree": 1,
        }
    )
    violations = _run(tmp_path, corpus)
    assert "forbidden_entity_type" in _checks(violations)


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def test_orphaned_reference_detected(tmp_path):
    corpus = _base_corpus()
    corpus["text_units"][0]["entity_ids"] = corpus["text_units"][0]["entity_ids"] + ["e-does-not-exist"]
    violations = _run(tmp_path, corpus)
    assert "orphaned_reference" in _checks(violations)


def test_dangling_relationship_detected(tmp_path):
    corpus = _base_corpus()
    corpus["relationships"].append(
        {
            "id": "r-dangling",
            "human_readable_id": 999,
            "source": "NONEXISTENT ENTITY",
            "target": "BIHAR",
            "description": "dangling edge",
            "text_unit_ids": ["tu-bihar"],
            "weight": 1.0,
            "combined_degree": 1,
        }
    )
    violations = _run(tmp_path, corpus)
    assert "dangling_relationship" in _checks(violations)


# ---------------------------------------------------------------------------
# Community/report consistency
# ---------------------------------------------------------------------------


def test_community_report_mismatch_detected(tmp_path):
    corpus = _base_corpus()
    corpus["communities"].append(
        {
            "id": "comm-extra",
            "human_readable_id": 999,
            "community": 999,
            "level": 0,
            "parent": -1,
            "children": [],
            "title": "Community 999",
            "entity_ids": [],
            "relationship_ids": [],
            "text_unit_ids": [],
            "period": "2026-08-14",
            "size": 0,
        }
    )
    violations = _run(tmp_path, corpus)
    assert "community_report_mismatch" in _checks(violations)


# ---------------------------------------------------------------------------
# Schema / missing files
# ---------------------------------------------------------------------------


def test_missing_output_file_detected_and_later_checks_skipped_cleanly(tmp_path):
    corpus = _base_corpus()
    paths = write_corpus(tmp_path, corpus)
    (paths["input_dir"] / "community_reports.parquet").unlink()

    violations = validate(paths["input_dir"], paths["source_dir"], paths["manifest_path"])
    checks = _checks(violations)
    assert "missing_output_file" in checks
    missing = [v for v in violations if v["check"] == "missing_output_file"][0]
    assert missing["subject"] == "community_reports"
    # every other check on the tables that DID load should still run
    # without crashing -- a clean corpus otherwise has no other violations
    assert checks == {"missing_output_file"}


def test_missing_required_column_detected(tmp_path):
    corpus = _base_corpus()
    paths = write_corpus(tmp_path, corpus)
    entities = pd.read_parquet(paths["input_dir"] / "entities.parquet")
    entities = entities.drop(columns=["description"])
    entities.to_parquet(paths["input_dir"] / "entities.parquet", index=False)

    violations = validate(paths["input_dir"], paths["source_dir"], paths["manifest_path"])
    assert "missing_required_column" in _checks(violations)
