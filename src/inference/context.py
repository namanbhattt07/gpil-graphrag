"""
Phase 7 -- thin wrapper around GraphRAG's own local-search machinery.

WHAT THIS FILE IS FOR (plain language):
The Phase 7 guard pipeline needs to sit BETWEEN retrieval and answer
generation so it can inspect the retrieved evidence before letting an LLM
write anything. GraphRAG's own `graphrag.api.local_search()` doesn't give us
that seam -- it builds context AND calls the completion model in one
opaque function. So this module does what `local_search()` does internally
(load the index's parquet tables, build the LocalSearch engine, build the
retrieval context for a query), but stops right after context-building and
hands the pieces back separately:
  - `engine`         -- the fully-wired LocalSearch object (has .model and
                         .system_prompt on it, ready for answer.py to use
                         if/when the premise check says it's safe to proceed)
  - `context_result`  -- GraphRAG's own ContextBuilderResult: context_chunks
                         (the formatted prompt text) and context_records
                         (a dict of DataFrames: entities/relationships/
                         reports/sources/claims) that premise_check.py and
                         grounding_check.py both read.

This file does not call an LLM by itself. The one real API call it makes is
the query-embedding call GraphRAG's own retrieval needs (to vector-match the
question against entity descriptions) -- unavoidable, and the same call the
plain `graphrag query` CLI makes for the same question.

Reuses GraphRAG's actual library code throughout (get_local_search_engine,
DataReader, read_indexer_* adapters) rather than re-implementing index
loading or retrieval -- see docs/ Phase 7 architecture audit for why.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.utils.api import get_embedding_store, load_search_prompt


@dataclass
class QueryContext:
    """Everything the Phase 7 pipeline needs after context-building, before
    (maybe) calling the answer-generation LLM."""

    engine: LocalSearch
    context_result: ContextBuilderResult

    @property
    def context_records(self) -> dict[str, pd.DataFrame]:
        """The retrieved evidence, split by type -- this is what
        premise_check.py and grounding_check.py inspect directly."""
        return self.context_result.context_records


def _load_config(
    index_root: Path,
    output_dir: Path,
    reporting_dir: Path | None = None,
) -> GraphRagConfig:
    """Load the index's settings.yaml, pointed at a specific output
    snapshot (e.g. output_fixed_clean) instead of whatever `output_storage`
    the file says by default -- mirrors how the `graphrag query --data`
    CLI flag works. reporting_dir optionally redirects the query.log
    GraphRAG writes on every run, so callers that want to keep an index
    directory read-only can send logs elsewhere.

    graphrag_common's load_config() changes the process's working directory
    to index_root's parent as a side effect (set_cwd=True by default), so a
    RELATIVE output_dir/reporting_dir passed in here would otherwise get
    silently re-resolved against that new cwd instead of the caller's --
    resolving to absolute paths first avoids that trap.
    """
    output_dir = Path(output_dir).resolve()
    cli_overrides: dict = {"output_storage": {"base_dir": str(output_dir)}}
    if reporting_dir is not None:
        reporting_dir = Path(reporting_dir).resolve()
        cli_overrides["reporting"] = {"base_dir": str(reporting_dir)}
    return load_config(root_dir=Path(index_root).resolve(), cli_overrides=cli_overrides)


async def _load_output_tables(config: GraphRagConfig) -> dict[str, pd.DataFrame]:
    """Read entities/relationships/communities/community_reports/text_units
    (and covariates, if present) back out of the index's parquet files.
    Same DataReader GraphRAG's own CLI uses internally, so column types
    match exactly what get_local_search_engine() expects."""
    storage = create_storage(config.output_storage)
    table_provider = create_table_provider(config.table_provider, storage=storage)
    reader = DataReader(table_provider)

    tables = {
        "entities": await reader.entities(),
        "communities": await reader.communities(),
        "community_reports": await reader.community_reports(),
        "text_units": await reader.text_units(),
        "relationships": await reader.relationships(),
    }
    tables["covariates"] = (
        await reader.covariates() if await table_provider.has("covariates") else None
    )
    return tables


def build_query_context(
    index_root: Path,
    output_dir: Path,
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
    reporting_dir: Path | None = None,
) -> QueryContext:
    """Load the index and build local-search retrieval context for `query`,
    without generating an answer. This is the ONE function in the whole
    Phase 7 pipeline that talks to GraphRAG/the index/the embedding API --
    everything downstream (premise_check, grounding_check) is pure Python
    working off what this returns.
    """
    config = _load_config(index_root, output_dir, reporting_dir)
    tables = asyncio.run(_load_output_tables(config))

    entities = read_indexer_entities(
        tables["entities"], tables["communities"], community_level
    )
    covariates = (
        read_indexer_covariates(tables["covariates"])
        if tables["covariates"] is not None
        else []
    )
    reports = read_indexer_reports(
        tables["community_reports"], tables["communities"], community_level
    )
    text_units = read_indexer_text_units(tables["text_units"])
    relationships = read_indexer_relationships(tables["relationships"])

    description_embedding_store = get_embedding_store(
        config=config.vector_store, embedding_name=entity_description_embedding
    )
    system_prompt = load_search_prompt(config.local_search.prompt)

    engine = get_local_search_engine(
        config=config,
        reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates={"claims": covariates},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
        system_prompt=system_prompt,
    )

    # This is the one real API call in this whole module: embeds `query`
    # and vector-searches it against entity description embeddings, then
    # ranks/truncates entities+relationships+reports+text units into a
    # prompt-sized context. Pure retrieval -- no completion call yet.
    context_result = engine.context_builder.build_context(
        query=query,
        **engine.context_builder_params,
    )

    return QueryContext(engine=engine, context_result=context_result)
