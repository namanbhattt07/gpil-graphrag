"""
src/ is the main Python package for the GPIL GraphRAG chat system.

Subpackages, each corresponding to one or more phases of the build plan:
- data_gen  -> Phase 2: generates synthetic Sales & Distribution data
- kpis      -> Phase 3: turns raw data into region x month KPIs
- graph     -> Phases 4-6: builds GraphRAG documents, runs indexing, retrieves evidence
- inference -> Phase 7: reasons over evidence to produce grounded, cited answers
- ui        -> Phase 8: Streamlit chat interface

Each subpackage is currently a placeholder and gets filled in during its
corresponding phase.
"""
