"""
canonicalize_index.py

WHAT THIS SCRIPT DOES (plain language)
Takes a GraphRAG output/ folder that validate_index.py has flagged
problems in and fixes the deterministic subset of them: Category/
Franchise entities that are really the same real-world thing but got
extracted with different spelling (an underscore vs. a space, a typo,
a state name incorrectly prepended) get merged into one entity, with
every other output file that references them -- relationships.parquet,
text_units.parquet, communities.parquet, the entity_description LanceDB
table -- updated to match. Optionally (only with
--backfill-missing-entities), it also synthesizes a State entity that
extraction skipped entirely, the way it did for Maharashtra in the
GPT-4o-mini pilot.

WHY IT LOOKS THE WAY IT DOES
Never overwrites --input-dir unless --in-place is passed -- always
writes a full copy to --output-dir first, so the original completed run
stays recoverable no matter what this script does.

Only merges a Category/Franchise entity when its title resolves against
this project's own closed vocabulary (see validate_index.py's
resolve_category_title / resolve_franchise_title) -- an entity that
merely LOOKS similar to another is never merged; it's left alone and
reported as unknown_franchise_entity / invalid_category_entity for a
human to add an explicit alias for for if it's a confirmed real variant.

Never renumbers an existing entity's human_readable_id -- community
reports already generated for this run cite entities by that id (e.g.
"[Data: Entities (443)..."); renumbering would silently break those
citations. New ids (backfilled entities) are only ever appended above
the current max, the same convention GraphRAG's own incremental-update
code uses (graphrag/index/update/entities.py).

Descriptions of merged entities are combined with a plain string join
(no LLM call) -- lower quality than GraphRAG's own summarize_descriptions
merge, acceptable for a narrow, rare correction. Because that changes
the surviving entity's description text, its LanceDB vector would
otherwise go stale (a description reading "A | B" next to an embedding
computed only from "A"). This script re-embeds exactly those survivors
via one embedding call each -- never unchanged entities, never a full
rebuild -- and records the exact count and ids in
canonicalization_report.json under entities_reembedded/embedding_calls.
This re-embedding, plus the opt-in missing-entity backfill's one
embedding call per backfilled entity, are the ONLY network calls this
script ever makes; if nothing merges and --backfill-missing-entities
isn't passed, it makes zero API calls.

community_reports.parquet is never rewritten here -- regenerating a
report needs a completion-model call, out of scope for this layer.
Instead, every community whose entity membership changed because of a
merge or backfill is recorded as "stale" in canonicalization_report.json
(affected_community_ids / stale_community_reports), so this is known
and explicitly tracked rather than silently stale.

Optionally (only with --remove-orphaned-forbidden-entities), also
removes forbidden-type entities (ZONE/CHANNEL -- see
validate_index.FORBIDDEN_ENTITY_TYPES) that are PROVABLY orphaned: zero
relationships reference them and zero communities list their id. This
is the exact, narrow condition manually verified for the real pilot's
"BIHAR ZONE 2" entity before removing it -- automating that specific,
zero-dependency case, never a general "delete anything forbidden"
sweep. A forbidden entity that has relationships or community
membership is deliberately left untouched (removing or remapping those
needs the same case-by-case evidence review already done for BIHAR
ZONE 2) and is listed in canonicalization_report.json's
forbidden_entities_skipped for a human to review. Makes zero API
calls -- nothing survives with a changed description to re-embed.

The script always finishes by re-running validate_index.py's checks
against its own output (step 10/11) and records whether that came back
clean: violation types this script claims to fix (invalid_category_entity,
duplicate_franchise_entity) must be gone, and no violation type that
wasn't present before may appear (a regression check) -- otherwise
final_validation_status is "failed" and the process exits non-zero.
Violation types this script never unconditionally claims to fix
(unknown_franchise_entity, manifest_source_mismatch,
missing_state_entity without the backfill flag, forbidden_entity_type
for entities not provably orphaned, etc.) are allowed to remain and are
listed as known/tracked, not treated as a failure.

Fixed step order:
  1. copy input to output (skipped in --in-place mode)
  2. deterministic Category canonicalization
  3. deterministic Franchise canonicalization
  4. remap relationships
  5. remap text-unit and community references
  6. update entity embeddings for changed/removed identities
  7. opt-in missing-entity backfill (--backfill-missing-entities only)
  8. opt-in orphaned forbidden-entity removal
     (--remove-orphaned-forbidden-entities only)
  9. produce canonicalization_report.json
  10. run the validator again
  11. fail if final validation surfaces a regression or a new problem

Run with (from the project root, venv activated):

    python -m src.graph.canonicalize_index \\
        --input-dir data/graphrag_index/pilot_run_mini/output \\
        --source-dir data/graphrag_index/input_pilot \\
        --manifest data/graphrag_docs_manifest.csv \\
        --output-dir data/graphrag_index/pilot_run_mini/output_fixed
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.graph.validate_index import (
    FORBIDDEN_ENTITY_TYPES,
    load_output_tables,
    resolve_category_title,
    resolve_franchise_title,
    validate,
    _state_from_filename,
    _to_list,
)

EmbedFn = Callable[[str], list[float]]


def _default_embed_fn() -> EmbedFn:
    """Real embedding call via the OpenAI SDK directly, matching the
    embedding_model configured in .env / settings.yaml. Not wrapped in
    GraphRAG's own cache/retry machinery -- this script makes at most a
    handful of calls per run (only for entities whose description
    actually changed, plus opt-in backfills), so that coupling isn't
    worth it. Constructing the client makes no network call by itself;
    the network call only happens when the returned function is
    actually invoked."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_base)
    model = settings.embedding_model

    def embed(text: str) -> list[float]:
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    return embed


def _open_entity_vector_store(lancedb_dir: Path):
    """Uses GraphRAG's own LanceDBVectorStore (graphrag_vectors.lancedb)
    rather than hand-rolling pyarrow writes, so every column this
    project's GraphRAG version expects (the exploded create_date_*/
    update_date_* fields) gets populated exactly the way GraphRAG itself
    populates them -- see graphrag_vectors/timestamp.py::explode_timestamp,
    invoked internally by this store's own _prepare_document/_prepare_update.

    Reads the table's own vector column width instead of assuming 1536
    (text-embedding-3-small's dimension) -- load_documents()/insert()
    build the vector array using LanceDBVectorStore's own vector_size
    attribute, so it must match the table's real fixed_size_list width or
    every insert fails with an ArrowInvalid length error, whether that's
    because a different embedding model was configured, or (as in this
    project's own test suite) a smaller vector size is used for a
    synthetic fixture."""
    import lancedb as lancedb_module
    from graphrag_vectors.lancedb import LanceDBVectorStore

    raw_db = lancedb_module.connect(str(lancedb_dir))
    vector_size = raw_db.open_table("entity_description").schema.field("vector").type.list_size

    store = LanceDBVectorStore(
        db_uri=str(lancedb_dir),
        index_name="entity_description",
        vector_size=vector_size,
        fields={},
    )
    store.connect()
    return store


def _vector_store_document(entity_id: str, vector: list[float]):
    from graphrag_vectors.vector_store import VectorStoreDocument

    return VectorStoreDocument(id=entity_id, vector=vector)


def _merge_entity_group(
    entities: pd.DataFrame, id_to_canonical: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Collapse every entity whose id is a key in id_to_canonical into
    one surviving row per canonical title. Mirrors the merge idea
    GraphRAG's own incremental-update code uses in
    graphrag/index/update/entities.py::_group_and_resolve_entities (pick
    one id, concatenate descriptions/text_unit_ids, recompute frequency)
    -- except keyed off a canonical title already verified against a
    closed vocabulary rather than a raw exact-string match, and never
    renumbering human_readable_id.

    Uses plain dict/list manipulation instead of vectorized pandas
    assignment for the list-valued columns (text_unit_ids) -- these
    tables are at most a few thousand rows, so there's no performance
    reason to fight pandas's broadcasting rules for list-in-a-cell
    assignment, and the explicit version is much easier to verify correct.

    Returns (updated_entities_df, id_mapping {losing_id: surviving_id},
    changed_descriptions {surviving_id: new_description}) -- the third
    return value is exactly what step 6 (embedding refresh) iterates over.
    """
    records: dict[str, dict] = {row["id"]: dict(row) for row in entities.to_dict("records")}
    id_mapping: dict[str, str] = {}
    changed_descriptions: dict[str, str] = {}

    groups: dict[str, list[str]] = {}
    for entity_id, canonical in id_to_canonical.items():
        groups.setdefault(canonical, []).append(entity_id)

    for canonical, ids in groups.items():
        group_records = [records[i] for i in ids]
        exact = [r for r in group_records if r["title"] == canonical]
        survivor = exact[0] if exact else min(group_records, key=lambda r: r["human_readable_id"])
        survivor_id = survivor["id"]

        if len(group_records) > 1:
            merged_description = " | ".join(
                dict.fromkeys(str(r["description"]) for r in group_records)
            )
            merged_text_unit_ids = list(
                dict.fromkeys(
                    itertools.chain.from_iterable(
                        (r["text_unit_ids"] if r["text_unit_ids"] is not None else [])
                        for r in group_records
                    )
                )
            )
            if merged_description != str(survivor["description"]):
                changed_descriptions[survivor_id] = merged_description
            records[survivor_id]["description"] = merged_description
            records[survivor_id]["text_unit_ids"] = merged_text_unit_ids
            records[survivor_id]["frequency"] = len(merged_text_unit_ids)

        records[survivor_id]["title"] = canonical

        for entity_id in ids:
            if entity_id != survivor_id:
                id_mapping[entity_id] = survivor_id
                del records[entity_id]

    merged_df = pd.DataFrame(list(records.values())).reset_index(drop=True)
    return merged_df, id_mapping, changed_descriptions


def _remap_relationships(
    relationships: pd.DataFrame, title_rename: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rewrites relationships.parquet's source/target for every renamed
    entity, then re-collapses any resulting duplicate (source, target)
    edges using the same groupby(["source", "target"]) aggregation
    GraphRAG's own incremental-update code uses in
    graphrag/index/update/relationships.py::_update_and_merge_relationships
    (keep the lowest human_readable_id, concatenate descriptions/
    text_unit_ids, recompute degree from the merged edge set).

    Returns (updated_relationships_df, id_mapping {losing_id: surviving_id})."""
    if not title_rename:
        return relationships, {}

    relationships = relationships.copy()
    relationships["source"] = relationships["source"].replace(title_rename)
    relationships["target"] = relationships["target"].replace(title_rename)

    records = relationships.to_dict("records")
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["source"], record["target"]), []).append(record)

    merged_records = []
    id_mapping: dict[str, str] = {}
    for (_source, _target), group in groups.items():
        if len(group) == 1:
            merged_records.append(group[0])
            continue
        survivor = dict(min(group, key=lambda r: r["human_readable_id"]))
        merged_description = " | ".join(dict.fromkeys(str(r["description"]) for r in group))
        merged_text_unit_ids = list(
            dict.fromkeys(
                itertools.chain.from_iterable(
                    (r["text_unit_ids"] if r["text_unit_ids"] is not None else [])
                    for r in group
                )
            )
        )
        survivor["description"] = merged_description
        survivor["text_unit_ids"] = merged_text_unit_ids
        survivor["weight"] = float(np.mean([r["weight"] for r in group]))
        merged_records.append(survivor)
        for record in group:
            if record["id"] != survivor["id"]:
                id_mapping[record["id"]] = survivor["id"]

    merged_df = pd.DataFrame(merged_records).reset_index(drop=True)
    merged_df["combined_degree"] = merged_df.groupby("source")["target"].transform(
        "count"
    ) + merged_df.groupby("target")["source"].transform("count")
    return merged_df, id_mapping


def _remap_list_column(df: pd.DataFrame, column: str, id_mapping: dict[str, str]) -> pd.DataFrame:
    if not id_mapping:
        return df
    df = df.copy()

    def remap(ids):
        if ids is None:
            return ids
        return list(dict.fromkeys(id_mapping.get(i, i) for i in ids))

    df[column] = df[column].apply(remap)
    return df


def _affected_community_ids(communities: pd.DataFrame, losing_entity_ids: set[str]) -> list[int]:
    if not losing_entity_ids:
        return []
    affected = []
    for _, row in communities.iterrows():
        if losing_entity_ids & set(_to_list(row["entity_ids"])):
            affected.append(row["community"])
    return sorted(set(affected))


def _backfill_missing_state(
    filename: str,
    verified_state: str,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    text_units: pd.DataFrame,
    communities: pd.DataFrame,
    documents: pd.DataFrame,
    embed_fn: EmbedFn,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, str, list[float], list[int]]:
    """Synthesizes the one missing STATE entity for `filename`,
    deterministically:
      - description follows the same minimal template already used
        verbatim elsewhere in the graph for other states, e.g. the
        PUNJAB entity: "An Indian state; this document reports its S&D
        performance for {Period}."
      - SERVES edges (Distributor -> State) and HAS_OBSERVATION edges
        (State -> each state-level Observation) mirror the exact
        relationship phrasing already used for a sibling state, e.g.
        Bihar's real "SERVES"/"HAS_OBSERVATION" edges.
      - community membership is cloned from the coarse-level (level 0/1)
        community id(s) already shared by the document's own Document
        entity (verified in the prior audit to be identical between a
        document's Document entity and its Distributor entities at
        those two levels, diverging only at the finer level-2 grouping).

    Calls embed_fn exactly once. Returns the updated tables plus a
    backfill summary record, the new entity's id, and its embedding
    vector (the caller is responsible for inserting that vector into
    LanceDB, the same way it does for re-embedded merge survivors)."""
    doc_rows = documents[documents["title"] == filename]
    if doc_rows.empty:
        raise ValueError(f"no document row found for {filename!r}")
    doc_id = doc_rows.iloc[0]["id"]

    tu_mask = text_units["document_id"] == doc_id
    if not tu_mask.any():
        raise ValueError(f"no text unit found for document {filename!r}")
    tu_idx = text_units.index[tu_mask][0]
    tu_row = text_units.loc[tu_idx]

    sibling_ids = set(_to_list(tu_row["entity_ids"]))
    siblings = entities[entities["id"].isin(sibling_ids)]

    document_entities = siblings[siblings["type"].str.upper() == "DOCUMENT"]
    if document_entities.empty:
        raise ValueError(f"no Document-type entity found for {filename!r}")
    document_entity = document_entities.iloc[0]

    period_entities = siblings[siblings["type"].str.upper() == "PERIOD"]
    period_title = period_entities.iloc[0]["title"].title() if len(period_entities) else "an unknown period"

    distributor_entities = siblings[siblings["type"].str.upper() == "DISTRIBUTOR"]
    observation_entities = siblings[siblings["type"].str.upper() == "OBSERVATION"]

    state_title = verified_state.upper()
    distributor_titles = list(distributor_entities["title"])

    def _is_state_level(title: str) -> bool:
        if not str(title).startswith(state_title):
            return False
        return not any(str(title).startswith(d) for d in distributor_titles)

    state_level_observations = observation_entities[
        observation_entities["title"].apply(_is_state_level)
    ]

    new_entity_id = str(uuid.uuid4())
    new_human_readable_id = int(entities["human_readable_id"].max()) + 1
    description = (
        f"An Indian state; this document reports its S&D performance for {period_title}"
    )
    new_entity_row = {
        "id": new_entity_id,
        "human_readable_id": new_human_readable_id,
        "title": state_title,
        "type": "STATE",
        "description": description,
        "text_unit_ids": [tu_row["id"]],
        "frequency": 1,
        "degree": len(distributor_entities) + len(state_level_observations),
    }
    entities = pd.concat([entities, pd.DataFrame([new_entity_row])], ignore_index=True)

    next_rel_hrid = int(relationships["human_readable_id"].max()) + 1
    new_relationship_rows = []
    for _, distributor in distributor_entities.iterrows():
        new_relationship_rows.append(
            {
                "id": str(uuid.uuid4()),
                "human_readable_id": next_rel_hrid,
                "source": distributor["title"],
                "target": state_title,
                "description": f"SERVES: {distributor['title'].title()} operates in {verified_state}",
                "weight": 1.0,
                "combined_degree": 0,
                "text_unit_ids": [tu_row["id"]],
            }
        )
        next_rel_hrid += 1
    for _, observation in state_level_observations.iterrows():
        new_relationship_rows.append(
            {
                "id": str(uuid.uuid4()),
                "human_readable_id": next_rel_hrid,
                "source": state_title,
                "target": observation["title"],
                "description": (
                    f"HAS_OBSERVATION: {verified_state} has a measured fact for "
                    f"{observation['title'].title()}"
                ),
                "weight": 1.0,
                "combined_degree": 0,
                "text_unit_ids": [tu_row["id"]],
            }
        )
        next_rel_hrid += 1

    new_rel_df = pd.DataFrame(new_relationship_rows)
    relationships = pd.concat([relationships, new_rel_df], ignore_index=True)
    relationships["combined_degree"] = relationships.groupby("source")["target"].transform(
        "count"
    ) + relationships.groupby("target")["source"].transform("count")

    new_rel_ids = list(new_rel_df["id"]) if len(new_rel_df) else []
    text_units = text_units.copy()
    text_units.at[tu_idx, "entity_ids"] = _to_list(tu_row["entity_ids"]) + [new_entity_id]
    text_units.at[tu_idx, "relationship_ids"] = _to_list(tu_row["relationship_ids"]) + new_rel_ids

    communities = communities.copy()
    doc_entity_id = document_entity["id"]
    doc_communities_mask = communities["entity_ids"].apply(
        lambda ids: doc_entity_id in _to_list(ids)
    )
    joined_community_ids: list[int] = []
    for idx in communities.index[doc_communities_mask]:
        communities.at[idx, "entity_ids"] = _to_list(communities.at[idx, "entity_ids"]) + [
            new_entity_id
        ]
        # cast off numpy's int64 -- json.dumps can't serialize it, and this
        # id ends up in canonicalization_report.json
        joined_community_ids.append(int(communities.at[idx, "community"]))

    vector = embed_fn(description)

    backfill_record = {
        "state": state_title,
        "document": filename,
        "entity_id": new_entity_id,
        "distributor_edges_added": len(distributor_entities),
        "observation_edges_added": len(state_level_observations),
        "joined_community_ids": sorted(joined_community_ids),
    }
    return (
        entities,
        relationships,
        text_units,
        communities,
        backfill_record,
        new_entity_id,
        vector,
        joined_community_ids,
    )


def _remove_orphaned_forbidden_entities(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    text_units: pd.DataFrame,
    communities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    """Removes forbidden-type entities (ZONE/CHANNEL) but ONLY when doing
    so is provably safe: zero relationships.parquet rows reference the
    entity's title as source or target, AND zero communities.parquet
    rows list the entity's id in entity_ids. An entity meeting both
    conditions is a true orphan -- there is no relationship to remap
    and no community membership to preserve, so removal needs no
    replacement-entity judgment call, only deletion.

    A forbidden entity that fails either condition is left completely
    untouched and reported separately in the second return value
    (skipped records) -- deciding what to do with its relationships/
    community membership needs the same case-by-case evidence review
    already done manually for the real pilot's BIHAR ZONE 2 entity
    (which happened to have zero of both, making it eligible here).
    This function only ever automates that exact zero-dependency case.

    Never modifies relationships.parquet or communities.parquet -- by
    construction, removed entities have zero rows in either that would
    need updating. Only entities.parquet (drop the row) and
    text_units.parquet (drop the id from entity_ids lists) are touched.
    Makes no embedding calls -- nothing survives with a changed
    description; the caller is responsible for deleting each removed
    entity's own vector row from LanceDB, the same way it already does
    for merge losers."""
    forbidden = entities[entities["type"].str.upper().isin(FORBIDDEN_ENTITY_TYPES)]

    removed_records: list[dict] = []
    skipped_records: list[dict] = []
    removed_ids: set[str] = set()

    for _, row in forbidden.iterrows():
        title = row["title"]
        entity_id = row["id"]

        relationship_count = int(
            ((relationships["source"] == title) | (relationships["target"] == title)).sum()
        )
        community_count = int(
            communities["entity_ids"].apply(lambda ids: entity_id in _to_list(ids)).sum()
        )

        if relationship_count == 0 and community_count == 0:
            removed_ids.add(entity_id)
            text_unit_reference_count = int(
                text_units["entity_ids"].apply(lambda ids: entity_id in _to_list(ids)).sum()
            )
            removed_records.append(
                {
                    "entity_id": entity_id,
                    "human_readable_id": int(row["human_readable_id"]),
                    "title": title,
                    "type": row["type"],
                    "relationship_rows_removed": 0,
                    "text_unit_references_removed": text_unit_reference_count,
                    "community_references_removed": 0,
                    "vector_rows_removed": 1,
                }
            )
        else:
            skipped_records.append(
                {
                    "entity_id": entity_id,
                    "human_readable_id": int(row["human_readable_id"]),
                    "title": title,
                    "type": row["type"],
                    "relationships_found": relationship_count,
                    "communities_found": community_count,
                    "reason": "not a provable orphan (has relationships and/or community "
                    "membership) -- needs a case-by-case decision, not automatic removal",
                }
            )

    if not removed_ids:
        return entities, text_units, removed_records, skipped_records

    entities = entities[~entities["id"].isin(removed_ids)].reset_index(drop=True)

    def _drop_removed(ids):
        return [i for i in _to_list(ids) if i not in removed_ids]

    text_units = text_units.copy()
    text_units["entity_ids"] = text_units["entity_ids"].apply(_drop_removed)

    return entities, text_units, removed_records, skipped_records


def _write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_parquet(output_dir / f"{name}.parquet", index=False)


def canonicalize(
    input_dir: Path,
    source_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    backfill_missing_entities: bool = False,
    remove_orphaned_forbidden_entities: bool = False,
    embed_fn: EmbedFn | None = None,
) -> dict:
    """Runs the full canonicalization pipeline (see module docstring for
    the fixed 10-step order) and returns the same dict that gets written
    to canonicalization_report.json."""
    # Step 9 needs an initial-violations baseline to compare the final
    # result against, so it's captured before anything is touched.
    initial_violations = validate(input_dir, source_dir, manifest_path)
    initial_checks_present = {v["check"] for v in initial_violations}

    # Step 1: copy input to output (skipped in --in-place mode, i.e. when
    # output_dir already IS input_dir -- copying a directory onto itself
    # via rmtree+copytree would destroy it).
    if output_dir != input_dir:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(input_dir, output_dir)

    tables, load_violations = load_output_tables(output_dir)
    if load_violations:
        raise ValueError(
            f"--input-dir is not a valid GraphRAG output directory: {load_violations}"
        )

    entities = tables["entities"]
    relationships = tables["relationships"]
    text_units = tables["text_units"]
    communities = tables["communities"]
    community_reports = tables["community_reports"]
    documents = tables["documents"]

    entities_before = len(entities)
    relationships_before = len(relationships)

    # Steps 2-3: resolve every Category/Franchise entity against the
    # closed vocabulary. Anything that doesn't resolve is left untouched.
    id_to_canonical: dict[str, str] = {}
    unresolved_franchises: list[str] = []
    for _, row in entities.iterrows():
        entity_type = str(row["type"]).upper()
        if entity_type == "CATEGORY":
            canonical, _reason = resolve_category_title(row["title"])
            if canonical is not None:
                id_to_canonical[row["id"]] = canonical
        elif entity_type == "FRANCHISE":
            canonical, _reason = resolve_franchise_title(row["title"])
            if canonical is not None:
                id_to_canonical[row["id"]] = canonical
            else:
                unresolved_franchises.append(row["title"])

    title_rename = {
        row["title"]: id_to_canonical[row["id"]]
        for _, row in entities.iterrows()
        if row["id"] in id_to_canonical and row["title"] != id_to_canonical[row["id"]]
    }
    id_to_original_title = dict(zip(entities["id"], entities["title"]))
    id_to_original_type = dict(zip(entities["id"], entities["type"]))

    entities, id_mapping, changed_descriptions = _merge_entity_group(entities, id_to_canonical)

    # Every entity whose title actually changed is reported here -- both
    # real merges (a losing id disappears into a survivor) AND single-row
    # renames (e.g. a lone "Ferro" becomes "Ferrero" with no other row to
    # merge with, so its own id/human_readable_id survive unchanged).
    # "losing_title" means "the title that no longer exists after this
    # operation", not necessarily "an entity that no longer exists".
    category_merges = [
        {"losing_title": id_to_original_title[eid], "surviving_title": id_to_canonical[eid]}
        for eid in id_to_canonical
        if str(id_to_original_type[eid]).upper() == "CATEGORY"
        and id_to_original_title[eid] != id_to_canonical[eid]
    ]
    franchise_merges = [
        {"losing_title": id_to_original_title[eid], "surviving_title": id_to_canonical[eid]}
        for eid in id_to_canonical
        if str(id_to_original_type[eid]).upper() == "FRANCHISE"
        and id_to_original_title[eid] != id_to_canonical[eid]
    ]

    # Step 4: remap relationships
    relationships, relationship_id_mapping = _remap_relationships(relationships, title_rename)

    # Step 5: remap text-unit and community references; compute which
    # communities are now stale because a member entity/relationship id
    # they list no longer exists under its old id.
    losing_entity_ids = set(id_mapping.keys())
    affected_community_ids = _affected_community_ids(communities, losing_entity_ids)
    text_units = _remap_list_column(text_units, "entity_ids", id_mapping)
    text_units = _remap_list_column(text_units, "relationship_ids", relationship_id_mapping)
    communities = _remap_list_column(communities, "entity_ids", id_mapping)
    communities = _remap_list_column(communities, "relationship_ids", relationship_id_mapping)

    stale_community_reports = sorted(
        set(affected_community_ids) & set(community_reports["community"])
    )

    # Step 6: keep entity_description LanceDB in sync -- delete losing
    # vectors, re-embed exactly the survivors whose description changed.
    lancedb_dir = output_dir / "lancedb"
    entities_reembedded: list[dict] = []
    embed = embed_fn or _default_embed_fn()
    store = _open_entity_vector_store(lancedb_dir) if lancedb_dir.exists() else None

    if store is not None and losing_entity_ids:
        store.remove(list(losing_entity_ids))

    if store is not None:
        for survivor_id, new_description in changed_descriptions.items():
            vector = embed(new_description)
            store.update(_vector_store_document(survivor_id, vector))
            title = entities.loc[entities["id"] == survivor_id, "title"].iloc[0]
            entities_reembedded.append(
                {
                    "entity_id": survivor_id,
                    "title": title,
                    "reason": "description_changed_after_entity_merge",
                }
            )

    _write_tables(
        output_dir,
        {
            "entities": entities,
            "relationships": relationships,
            "text_units": text_units,
            "communities": communities,
        },
    )

    # Step 7: opt-in missing-entity backfill
    missing_entity_backfills: list[dict] = []
    backfill_affected_community_ids: list[int] = []
    if backfill_missing_entities:
        pre_backfill_violations = validate(output_dir, source_dir, manifest_path)
        missing_state_violations = [
            v for v in pre_backfill_violations if v["check"] == "missing_state_entity"
        ]
        for violation in missing_state_violations:
            filename = violation["subject"]
            verified_state = _state_from_filename(filename)
            if verified_state is None:
                continue
            (
                entities,
                relationships,
                text_units,
                communities,
                record,
                new_entity_id,
                vector,
                joined_community_ids,
            ) = _backfill_missing_state(
                filename=filename,
                verified_state=verified_state,
                entities=entities,
                relationships=relationships,
                text_units=text_units,
                communities=communities,
                documents=documents,
                embed_fn=embed,
            )
            missing_entity_backfills.append(record)
            backfill_affected_community_ids.extend(joined_community_ids)
            if store is not None:
                store.insert(_vector_store_document(new_entity_id, vector))

        _write_tables(
            output_dir,
            {
                "entities": entities,
                "relationships": relationships,
                "text_units": text_units,
                "communities": communities,
            },
        )

    # Step 8: opt-in orphaned forbidden-entity removal. Only entities
    # provably disconnected from the rest of the graph (zero
    # relationships, zero community membership) are removed -- anything
    # else is left untouched and reported in forbidden_entities_skipped.
    forbidden_entities_removed: list[dict] = []
    forbidden_entities_skipped: list[dict] = []
    if remove_orphaned_forbidden_entities:
        entities, text_units, forbidden_entities_removed, forbidden_entities_skipped = (
            _remove_orphaned_forbidden_entities(entities, relationships, text_units, communities)
        )
        if forbidden_entities_removed and store is not None:
            store.remove([r["entity_id"] for r in forbidden_entities_removed])

        _write_tables(
            output_dir,
            {
                "entities": entities,
                "relationships": relationships,
                "text_units": text_units,
                "communities": communities,
            },
        )

    # Community IDs affected by either a merge (step 5) or a backfilled
    # entity joining an existing community (step 7) are both "stale" in
    # the same sense: a report already exists for that community and no
    # longer reflects its current membership. Both sources are unioned
    # here, after the backfill step has run, rather than only tracking
    # merge-caused staleness.
    affected_community_ids = sorted(set(affected_community_ids) | set(backfill_affected_community_ids))
    stale_community_reports = sorted(
        set(affected_community_ids) & set(community_reports["community"])
    )

    # Step 9: run the validator again
    final_violations = validate(output_dir, source_dir, manifest_path)
    final_checks_present = {v["check"] for v in final_violations}

    # Step 10: fail only on a regression (a violation type this script
    # claims to fix is still present) or an unrecognized new problem
    # (a violation type that wasn't present in the original input at all).
    fixed_checks = {
        "invalid_category_entity",
        "duplicate_category_entity",
        "duplicate_franchise_entity",
    }
    still_present_fixed_types = final_checks_present & fixed_checks
    new_check_types = final_checks_present - initial_checks_present

    if still_present_fixed_types or new_check_types:
        final_validation_status = "failed"
    elif final_violations:
        final_validation_status = "clean_with_known_remaining"
    else:
        final_validation_status = "clean"

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entities_before": entities_before,
        "entities_after": len(entities),
        "relationships_before": relationships_before,
        "relationships_after": len(relationships),
        "category_merges": category_merges,
        "franchise_merges": franchise_merges,
        "unresolved_franchises": sorted(set(unresolved_franchises)),
        "missing_entity_backfills": missing_entity_backfills,
        "forbidden_entities_removed": forbidden_entities_removed,
        "forbidden_entities_skipped": forbidden_entities_skipped,
        "affected_community_ids": affected_community_ids,
        "stale_community_reports": stale_community_reports,
        "entities_reembedded": entities_reembedded,
        "embedding_calls": len(entities_reembedded) + len(missing_entity_backfills),
        "final_validation_status": final_validation_status,
        "final_validation_remaining_violations": [
            {"check": check, "count": sum(1 for v in final_violations if v["check"] == check)}
            for check in sorted(final_checks_present)
        ],
        "regression_violation_types": sorted(still_present_fixed_types),
        "new_violation_types": sorted(new_check_types),
    }

    (output_dir / "canonicalization_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite --input-dir instead of writing a copy. Off by default.",
    )
    parser.add_argument(
        "--backfill-missing-entities",
        action="store_true",
        help="Also synthesize missing State entities. Makes one embedding call per "
        "backfilled entity. Off by default.",
    )
    parser.add_argument(
        "--remove-orphaned-forbidden-entities",
        action="store_true",
        help="Also remove forbidden-type (ZONE/CHANNEL) entities that are provably "
        "orphaned: zero relationships and zero community membership. Makes zero API "
        "calls. Off by default.",
    )
    args = parser.parse_args()

    if args.in_place:
        output_dir = args.input_dir
    else:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required unless --in-place is passed")
        output_dir = args.output_dir

    report = canonicalize(
        input_dir=args.input_dir,
        source_dir=args.source_dir,
        manifest_path=args.manifest,
        output_dir=output_dir,
        backfill_missing_entities=args.backfill_missing_entities,
        remove_orphaned_forbidden_entities=args.remove_orphaned_forbidden_entities,
    )

    print(f"Canonicalization complete. Report written to {output_dir / 'canonicalization_report.json'}")
    print(json.dumps(report, indent=2))

    if report["final_validation_status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
