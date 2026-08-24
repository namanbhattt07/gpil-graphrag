"""
Tests for src/graph/canonicalize_index.py.

Reuses the corpus builder from tests/test_validate_index.py (the same
GraphRAG-shaped fixture the validator tests use) and additionally
writes a minimal real LanceDB entity_description table via GraphRAG's
own LanceDBVectorStore (graphrag_vectors.lancedb), so these tests
exercise the exact add/update/delete path production runs use, not a
mock. Every embedding call in this suite goes through a stub embed_fn
(never the real OpenAI API) -- this suite makes zero network calls and
needs no API key, verified explicitly by test_no_api_call_when_nothing_
changes below, which uses a stub that raises if it's ever invoked.
"""

from pathlib import Path

import pandas as pd
import pytest

from graphrag_vectors.lancedb import LanceDBVectorStore
from graphrag_vectors.vector_store import VectorStoreDocument

from src.graph.canonicalize_index import canonicalize
from tests.test_validate_index import _base_corpus, remove_state_entity, write_corpus

VECTOR_SIZE = 8


def _write_lancedb(output_dir: Path, entity_ids: list[str]) -> None:
    store = LanceDBVectorStore(
        db_uri=str(output_dir / "lancedb"),
        index_name="entity_description",
        vector_size=VECTOR_SIZE,
        fields={},
    )
    store.connect()
    store.create_index()
    docs = [
        VectorStoreDocument(id=eid, vector=[float(i)] * VECTOR_SIZE)
        for i, eid in enumerate(entity_ids)
    ]
    store.load_documents(docs)


def _open_store(output_dir: Path) -> LanceDBVectorStore:
    store = LanceDBVectorStore(
        db_uri=str(output_dir / "lancedb"),
        index_name="entity_description",
        vector_size=VECTOR_SIZE,
        fields={},
    )
    store.connect()
    return store


def _lancedb_ids(output_dir: Path) -> set[str]:
    return set(_open_store(output_dir).document_collection.to_pandas()["id"])


def _lancedb_vector(output_dir: Path, entity_id: str) -> list[float]:
    df = _open_store(output_dir).document_collection.to_pandas()
    row = df[df["id"] == entity_id].iloc[0]
    return list(row["vector"])


def _stub_embed_fn(calls: list[str]):
    def embed(text: str) -> list[float]:
        calls.append(text)
        return [float(len(calls) * 10)] * VECTOR_SIZE

    return embed


def _raising_embed_fn(text: str) -> list[float]:
    raise AssertionError(f"embed_fn should not have been called, but was called with: {text!r}")


def _setup(tmp_path: Path, corpus: dict) -> dict:
    paths = write_corpus(tmp_path, corpus)
    entity_ids = [e["id"] for e in corpus["entities"]]
    _write_lancedb(paths["input_dir"], entity_ids)
    return paths


# ---------------------------------------------------------------------------
# Franchise dedup
# ---------------------------------------------------------------------------


def test_known_franchise_duplicate_merges(tmp_path):
    corpus = _base_corpus()
    survivor_id = next(e["id"] for e in corpus["entities"] if e["title"] == "GPI_Franchise_1")
    survivor_hrid = next(
        e["human_readable_id"] for e in corpus["entities"] if e["id"] == survivor_id
    )
    survivor_description = next(
        e["description"] for e in corpus["entities"] if e["id"] == survivor_id
    )
    dup_id = "e-dup"
    corpus["entities"].append(
        {
            "id": dup_id,
            "human_readable_id": 999,
            "title": "GPI FRANCHISE 1",
            "type": "FRANCHISE",
            "description": "space-formatted variant of the same franchise",
            "text_unit_ids": ["tu-punjab"],
            "frequency": 1,
            "degree": 1,
        }
    )
    paths = _setup(tmp_path, corpus)

    calls: list[str] = []
    output_dir = tmp_path / "fixed"
    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        embed_fn=_stub_embed_fn(calls),
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    franchise_rows = entities[entities["title"] == "GPI_Franchise_1"]
    assert len(franchise_rows) == 1
    row = franchise_rows.iloc[0]

    # (1) final description is the merge of both
    assert survivor_description in row["description"]
    assert "space-formatted variant of the same franchise" in row["description"]
    # (2) surviving entity keeps its UUID
    assert row["id"] == survivor_id
    # (3) surviving entity keeps its human_readable_id
    assert row["human_readable_id"] == survivor_hrid
    # (4) losing entity is removed
    assert dup_id not in set(entities["id"])

    assert report["franchise_merges"] == [
        {"losing_title": "GPI FRANCHISE 1", "surviving_title": "GPI_Franchise_1"}
    ]
    assert report["entities_before"] - report["entities_after"] == 1

    # (5) surviving entity receives a new embedding for the final description
    assert calls == [row["description"]]
    reembedded = report["entities_reembedded"]
    assert len(reembedded) == 1
    assert reembedded[0]["entity_id"] == survivor_id
    assert reembedded[0]["reason"] == "description_changed_after_entity_merge"
    assert report["embedding_calls"] == 1
    assert _lancedb_vector(output_dir, survivor_id) == [10.0] * VECTOR_SIZE

    # (6) losing entity's embedding is removed
    assert dup_id not in _lancedb_ids(output_dir)


def test_unknown_franchise_not_silently_merged(tmp_path):
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
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"
    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        embed_fn=_raising_embed_fn,
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    assert "SOMETHING_UNKNOWN" in set(entities["title"])
    assert "e-unknown" in set(entities["id"])
    assert report["unresolved_franchises"] == ["SOMETHING_UNKNOWN"]

    remaining = {r["check"]: r["count"] for r in report["final_validation_remaining_violations"]}
    assert remaining.get("unknown_franchise_entity") == 1
    assert report["final_validation_status"] != "failed"


# ---------------------------------------------------------------------------
# Category aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_title,expected_canonical", [("Ferro", "Ferrero"), ("Punjab Candy", "Candy")])
def test_category_alias_merges_correctly(tmp_path, bad_title, expected_canonical):
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
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"
    calls: list[str] = []
    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        embed_fn=_stub_embed_fn(calls),
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    category_titles = set(entities[entities["type"] == "CATEGORY"]["title"])
    assert bad_title not in category_titles
    assert expected_canonical in category_titles
    assert any(m["losing_title"] == bad_title for m in report["category_merges"])


# ---------------------------------------------------------------------------
# Missing-State backfill: opt-in only, one API call per entity, none otherwise
# ---------------------------------------------------------------------------


def test_missing_state_not_backfilled_without_flag(tmp_path):
    corpus = remove_state_entity(_base_corpus(), "BIHAR")
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"

    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        backfill_missing_entities=False,
        embed_fn=_raising_embed_fn,  # must never be called
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    assert "BIHAR" not in set(entities[entities["type"] == "STATE"]["title"])
    assert report["missing_entity_backfills"] == []
    remaining = {r["check"]: r["count"] for r in report["final_validation_remaining_violations"]}
    assert remaining.get("missing_state_entity") == 1
    assert report["final_validation_status"] != "failed"


def test_missing_state_backfilled_only_with_flag(tmp_path):
    corpus = remove_state_entity(_base_corpus(), "BIHAR")
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"
    calls: list[str] = []

    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        backfill_missing_entities=True,
        embed_fn=_stub_embed_fn(calls),
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    bihar_rows = entities[(entities["type"] == "STATE") & (entities["title"] == "BIHAR")]
    assert len(bihar_rows) == 1
    new_entity_id = bihar_rows.iloc[0]["id"]
    assert new_entity_id not in {e["id"] for e in corpus["entities"]}  # freshly minted UUID
    assert bihar_rows.iloc[0]["human_readable_id"] == entities["human_readable_id"].max()

    assert len(report["missing_entity_backfills"]) == 1
    assert report["missing_entity_backfills"][0]["state"] == "BIHAR"
    assert calls  # embed_fn was invoked exactly for the backfilled entity's description
    assert new_entity_id in _lancedb_ids(output_dir)

    remaining = {r["check"]: r["count"] for r in report["final_validation_remaining_violations"]}
    assert "missing_state_entity" not in remaining
    assert report["final_validation_status"] == "clean"

    # The backfilled entity joins its document's existing community(s) --
    # that community's report now predates this new member, so it must be
    # tracked as stale, not just merge-caused staleness.
    communities = pd.read_parquet(output_dir / "communities.parquet")
    joined = communities[communities["entity_ids"].apply(lambda ids: new_entity_id in list(ids))]
    assert len(joined) >= 1
    joined_ids = set(joined["community"])
    assert joined_ids <= set(report["affected_community_ids"])
    assert joined_ids <= set(report["stale_community_reports"])
    assert report["missing_entity_backfills"][0]["joined_community_ids"] == sorted(joined_ids)


# ---------------------------------------------------------------------------
# Orphaned forbidden-entity removal: opt-in only, zero API calls, and only
# for entities provably disconnected from the rest of the graph
# ---------------------------------------------------------------------------


def _add_zone_entity(corpus: dict, entity_id: str = "e-zone") -> dict:
    """Appends a ZONE entity tied to the Bihar text unit, mirroring the
    real pilot's BIHAR ZONE 2: present in one text unit's entity_ids,
    but (by default) zero relationships and zero community membership."""
    corpus["entities"].append(
        {
            "id": entity_id,
            "human_readable_id": 999,
            "title": "BIHAR ZONE 2",
            "type": "ZONE",
            "description": "should never be extracted per the prompt",
            "text_unit_ids": ["tu-bihar"],
            "frequency": 1,
            "degree": 0,
        }
    )
    for tu in corpus["text_units"]:
        if tu["id"] == "tu-bihar":
            tu["entity_ids"] = tu["entity_ids"] + [entity_id]
    return corpus


def test_orphaned_zone_entity_not_removed_without_flag(tmp_path):
    corpus = _add_zone_entity(_base_corpus())
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"

    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        remove_orphaned_forbidden_entities=False,
        embed_fn=_raising_embed_fn,
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    assert "e-zone" in set(entities["id"])
    assert report["forbidden_entities_removed"] == []
    remaining = {r["check"]: r["count"] for r in report["final_validation_remaining_violations"]}
    assert remaining.get("forbidden_entity_type") == 1


def test_orphaned_zone_entity_removed_only_with_flag(tmp_path):
    corpus = _add_zone_entity(_base_corpus())
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"

    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        remove_orphaned_forbidden_entities=True,
        embed_fn=_raising_embed_fn,  # removal alone must make zero API calls
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    assert "e-zone" not in set(entities["id"])

    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    bihar_tu = text_units[text_units["id"] == "tu-bihar"].iloc[0]
    assert "e-zone" not in list(bihar_tu["entity_ids"])

    assert "e-zone" not in _lancedb_ids(output_dir)

    assert len(report["forbidden_entities_removed"]) == 1
    removed = report["forbidden_entities_removed"][0]
    assert removed["entity_id"] == "e-zone"
    assert removed["title"] == "BIHAR ZONE 2"
    assert removed["relationship_rows_removed"] == 0
    assert removed["text_unit_references_removed"] == 1
    assert removed["community_references_removed"] == 0
    assert removed["vector_rows_removed"] == 1
    assert report["forbidden_entities_skipped"] == []
    assert report["embedding_calls"] == 0

    remaining = {r["check"]: r["count"] for r in report["final_validation_remaining_violations"]}
    assert "forbidden_entity_type" not in remaining
    assert report["final_validation_status"] == "clean"


def test_forbidden_entity_with_relationship_is_not_removed(tmp_path):
    corpus = _add_zone_entity(_base_corpus())
    corpus["relationships"].append(
        {
            "id": "r-zone",
            "human_readable_id": 999,
            "source": "BIHAR ZONE 2",
            "target": "BIHAR",
            "description": "some extracted edge",
            "text_unit_ids": ["tu-bihar"],
            "weight": 1.0,
            "combined_degree": 1,
        }
    )
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"

    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        remove_orphaned_forbidden_entities=True,
        embed_fn=_raising_embed_fn,
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    assert "e-zone" in set(entities["id"])
    assert report["forbidden_entities_removed"] == []
    assert len(report["forbidden_entities_skipped"]) == 1
    skipped = report["forbidden_entities_skipped"][0]
    assert skipped["entity_id"] == "e-zone"
    assert skipped["relationships_found"] == 1
    assert skipped["communities_found"] == 0


def test_forbidden_entity_with_community_membership_is_not_removed(tmp_path):
    corpus = _add_zone_entity(_base_corpus())
    corpus["communities"][0]["entity_ids"] = corpus["communities"][0]["entity_ids"] + ["e-zone"]
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"

    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        remove_orphaned_forbidden_entities=True,
        embed_fn=_raising_embed_fn,
    )

    entities = pd.read_parquet(output_dir / "entities.parquet")
    assert "e-zone" in set(entities["id"])
    assert report["forbidden_entities_removed"] == []
    assert len(report["forbidden_entities_skipped"]) == 1
    skipped = report["forbidden_entities_skipped"][0]
    assert skipped["relationships_found"] == 0
    assert skipped["communities_found"] == 1


def test_orphaned_zone_removal_second_pass_is_a_no_op(tmp_path):
    corpus = _add_zone_entity(_base_corpus())
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"

    canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        remove_orphaned_forbidden_entities=True,
        embed_fn=_raising_embed_fn,
    )

    second_output_dir = tmp_path / "fixed_again"
    second_report = canonicalize(
        input_dir=output_dir,
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=second_output_dir,
        remove_orphaned_forbidden_entities=True,
        embed_fn=_raising_embed_fn,
    )

    assert second_report["forbidden_entities_removed"] == []
    assert second_report["forbidden_entities_skipped"] == []
    assert second_report["entities_before"] == second_report["entities_after"]


# ---------------------------------------------------------------------------
# No API calls when nothing changes
# ---------------------------------------------------------------------------


def test_no_api_call_when_nothing_changes(tmp_path):
    paths = _setup(tmp_path, _base_corpus())
    output_dir = tmp_path / "fixed"
    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        embed_fn=_raising_embed_fn,  # would raise AssertionError if ever called
    )
    assert report["embedding_calls"] == 0
    assert report["entities_reembedded"] == []
    assert report["final_validation_status"] == "clean"


# ---------------------------------------------------------------------------
# Stale community-report tracking
# ---------------------------------------------------------------------------


def test_affected_communities_tracked_as_stale(tmp_path):
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-dup",
            "human_readable_id": 999,
            "title": "GPI FRANCHISE 1",
            "type": "FRANCHISE",
            "description": "space variant",
            "text_unit_ids": ["tu-punjab"],
            "frequency": 1,
            "degree": 1,
        }
    )
    # put the losing entity in a community, exactly like real merged
    # franchises would already belong to some community
    corpus["communities"][1]["entity_ids"] = corpus["communities"][1]["entity_ids"] + ["e-dup"]
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"

    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        embed_fn=_stub_embed_fn([]),
    )

    assert corpus["communities"][1]["community"] in report["affected_community_ids"]
    assert corpus["communities"][1]["community"] in report["stale_community_reports"]

    communities = pd.read_parquet(output_dir / "communities.parquet")
    punjab_community = communities[communities["community"] == corpus["communities"][1]["community"]].iloc[0]
    assert "e-dup" not in list(punjab_community["entity_ids"])
    survivor_id = next(e["id"] for e in corpus["entities"] if e["title"] == "GPI_Franchise_1")
    assert survivor_id in list(punjab_community["entity_ids"])


# ---------------------------------------------------------------------------
# canonicalization_report.json contents
# ---------------------------------------------------------------------------


def test_canonicalization_report_json_written_with_expected_fields(tmp_path):
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-dup",
            "human_readable_id": 999,
            "title": "GPI FRANCHISE 1",
            "type": "FRANCHISE",
            "description": "space variant",
            "text_unit_ids": ["tu-punjab"],
            "frequency": 1,
            "degree": 1,
        }
    )
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"
    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        embed_fn=_stub_embed_fn([]),
    )

    report_path = output_dir / "canonicalization_report.json"
    assert report_path.exists()
    import json

    on_disk = json.loads(report_path.read_text())
    assert on_disk == report

    expected_fields = {
        "input_dir",
        "output_dir",
        "timestamp",
        "entities_before",
        "entities_after",
        "relationships_before",
        "relationships_after",
        "category_merges",
        "franchise_merges",
        "unresolved_franchises",
        "missing_entity_backfills",
        "affected_community_ids",
        "stale_community_reports",
        "entities_reembedded",
        "embedding_calls",
        "final_validation_status",
        "final_validation_remaining_violations",
    }
    assert expected_fields <= set(report.keys())
    assert report["entities_before"] == len(corpus["entities"])
    assert report["entities_after"] == len(corpus["entities"]) - 1  # one franchise merged away


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_second_pass_is_a_no_op(tmp_path):
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-dup",
            "human_readable_id": 999,
            "title": "GPI FRANCHISE 1",
            "type": "FRANCHISE",
            "description": "space variant",
            "text_unit_ids": ["tu-punjab"],
            "frequency": 1,
            "degree": 1,
        }
    )
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"

    first_calls: list[str] = []
    first_report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        embed_fn=_stub_embed_fn(first_calls),
    )
    assert first_report["embedding_calls"] == 1

    second_output_dir = tmp_path / "fixed_again"
    second_report = canonicalize(
        input_dir=output_dir,
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=second_output_dir,
        embed_fn=_raising_embed_fn,  # zero additional embedding calls expected
    )

    assert second_report["category_merges"] == []
    assert second_report["franchise_merges"] == []
    assert second_report["missing_entity_backfills"] == []
    assert second_report["embedding_calls"] == 0
    assert second_report["entities_before"] == second_report["entities_after"]
    assert second_report["final_validation_status"] == "clean"


# ---------------------------------------------------------------------------
# Final validator behavior
# ---------------------------------------------------------------------------


def test_final_validator_passes_clean_for_fully_fixable_corpus(tmp_path):
    corpus = _base_corpus()
    corpus["entities"].append(
        {
            "id": "e-dup",
            "human_readable_id": 999,
            "title": "GPI FRANCHISE 1",
            "type": "FRANCHISE",
            "description": "space variant",
            "text_unit_ids": ["tu-punjab"],
            "frequency": 1,
            "degree": 1,
        }
    )
    paths = _setup(tmp_path, corpus)
    output_dir = tmp_path / "fixed"
    report = canonicalize(
        input_dir=paths["input_dir"],
        source_dir=paths["source_dir"],
        manifest_path=paths["manifest_path"],
        output_dir=output_dir,
        embed_fn=_stub_embed_fn([]),
    )
    assert report["final_validation_status"] == "clean"
    assert report["final_validation_remaining_violations"] == []
