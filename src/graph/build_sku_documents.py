"""
Phase 7e -- SKU-level narrative document builder (Problem 1).

WHAT THIS SCRIPT DOES (plain language):
src/graph/build_documents.py's 672 State x Month documents (the ones
actually indexed into GraphRAG) never mention a single SKU -- Phase 4's
document design only ever rolled orders.csv up to Category grain. This
script does the SAME kind of rollup-to-narrative-text transform Phase 4
does, one level deeper: it turns src/kpis/compute_kpis.py's
kpi_state_month_sku.csv (28 states x 24 months x ~59 SKUs) into one
narrative .txt document per state/month, using EXACTLY the same rendering
function (src/inference/sku_evidence.py's render_sku_evidence_document())
the live inference pipeline uses to build SKU evidence on the fly for a
question -- so a document this script writes to disk and the evidence text
pipeline.py injects into a live answer are byte-for-byte identical for the
same state/month, and a citation like "sku-Punjab-March_2025" that
sku_evidence.py hands back really does correspond to the file this script
would write for Punjab/March 2025.

WHY THIS IS A SEPARATE CORPUS, NOT MERGED INTO data/graphrag_input/:
data/graphrag_input/'s 672 documents are what the live GraphRAG index
(data/graphrag_index/) was actually built from. Writing into that same
directory (or regenerating build_documents.py's own output) would force a
full GraphRAG re-index (real LLM entity-extraction + embedding spend) just
to make these SKU facts nominally present in the corpus -- when the live
inference pipeline (src/inference/sku_evidence.py, wired in via
src/inference/pipeline.py) already answers SKU questions correctly without
that cost, using this exact same rendered text as evidence. This script's
output (data/graphrag_sku_input/) exists for two reasons: (1) it is a real,
on-disk, human-inspectable artifact backing every "sku-<State>-<Period>"
citation the pipeline ever hands back -- provenance that can be opened and
read, not just claimed; (2) it is ready to be indexed into GraphRAG later,
if/when this project decides the SKU-level graph relationships (SKU ->
Franchise -> Category entity links) are worth the re-indexing cost -- a
decision this script deliberately does not make on its own.

OUTPUT:
  data/graphrag_sku_input/<State>_<YYYY-MM>_sku.txt -- one file per
    state/month, mirroring build_documents.py's own filename convention
    with a "_sku" suffix so the two corpora's files are never confused.
  data/graphrag_sku_docs_manifest.csv -- filename -> state/month lookup,
    mirroring build_documents.py's own manifest.

Run with:  python -m src.graph.build_sku_documents
(from the project root, with the venv activated. Requires
data/kpi_state_month_sku.csv to already exist -- run
`python -m src.kpis.compute_kpis` first if it doesn't.)
"""

from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.inference.sku_evidence import yyyymm_to_words, render_sku_evidence_document


def _safe_filename(state_name: str, month_str: str) -> str:
    """Mirrors build_documents.py's own _safe_filename(), with a '_sku'
    suffix so this corpus's files can never be mistaken for (or collide
    with) the main indexed corpus's files."""
    safe_state = state_name.replace(" ", "_")
    return f"{safe_state}_{month_str}_sku.txt"


def build_all_sku_documents(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Build one document per (state, month) present in
    kpi_state_month_sku.csv and write them to output_dir. Returns the
    manifest DataFrame (also written to disk), mirroring
    build_documents.py's build_all_documents()."""
    sku_kpis = pd.read_csv(data_dir / "kpi_state_month_sku.csv", dtype={"month": str})

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    groups = sku_kpis.sort_values(["state_name", "month", "sku_id"]).groupby(
        ["state_name", "month"], sort=True
    )
    total = groups.ngroups
    for i, ((state_name, month_str), rows) in enumerate(groups, start=1):
        period_label = yyyymm_to_words(month_str)
        document_text = render_sku_evidence_document(state_name, period_label, rows)

        filename = _safe_filename(state_name, month_str)
        (output_dir / filename).write_text(document_text, encoding="utf-8")

        print(f"[{i}/{total}] wrote {filename}", flush=True)

        manifest_rows.append({"filename": filename, "state_name": state_name, "month": month_str})

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(data_dir / "graphrag_sku_docs_manifest.csv", index=False)
    return manifest


def main():
    """Entry point for `python -m src.graph.build_sku_documents`."""
    settings = get_settings()
    data_dir = settings.project_root / "data"
    output_dir = data_dir / "graphrag_sku_input"

    manifest = build_all_sku_documents(data_dir, output_dir)

    print(f"Wrote {len(manifest)} documents to {output_dir}")
    print(f"Manifest saved to {data_dir / 'graphrag_sku_docs_manifest.csv'}")


if __name__ == "__main__":
    main()
