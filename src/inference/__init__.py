"""
inference: the reasoning layer that turns evidence into grounded answers (Phase 7).

Will hold three separate, independently testable stages:
- Stage A: structure raw evidence into clean fact objects
- Stage B: guarded chain-of-thought reasoning over those facts
- Stage C: grounding check that verifies every claim traces back to evidence
Empty until Phase 7.
"""
