# Ibis3: Universal Topological Coherence Operator

This repository implements a clean integration of the Universal Topological Coherence Operator across numeric, market, and narrative domains.

## Core Concept

The central coherence score is:

C = α c + (1 - α) r - β H

Where:
- `c` is the chiasm coherence from non-local boundary alignment.
- `r` is the phase synchronization score (Kuramoto-style / PLV / PLI).
- `H` is the entropy of the correlation structure.
- `α` and `β` are configurable weights.

The system also uses phase space reconstruction via Takens embedding to validate emergent geometry and closure.

## Structure

- `src/` - core topology and adapter logic.
- `engines/` - IBIS telemetry engines for signal clarity, narrative infection, identity resonance, entanglement, and phase transitions.
- `validation/` - validation suite and legacy examples.
- `data/` - placeholder for market or model data.
- `notebooks/` - placeholder for analysis notebooks.

## Running the Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the unified pipeline:

```bash
python unified_pipeline.py
```

Use `SyncModule` directly for quick phase-sync diagnostics:

```python
from src.coherence import SyncModule
sync = SyncModule()
report = sync.order_parameter(price_series)
```

Run validation:

```bash
python unified_pipeline.py --validate
```

Fetch real market data for ETH or SOL:

```bash
python unified_pipeline.py --market ETH --days 30
```

Analyze narrative input:

```bash
python unified_pipeline.py --text "The market is stabilizing and coherence is emerging."
```

## Outputs

The pipeline saves a JSON report by default to `report.json`.
The validation suite saves `validation/validation_results.json` with discrimination metrics and alignment scoring.

## Notes

- The new `DomainAdapter` normalizes numeric and text input, applies the unified phase analyzer, runs Takens phase-space geometry from `src/geometry_engine.py`, and integrates IBIS telemetry engines.
- The validation suite computes discrimination power between synthetic coherent, drifting, and regime-shift patterns.
- The system is designed to be end-to-end runnable with `python unified_pipeline.py --validate`.

## Next Steps

- Apply the pipeline to real Solana or ETH historical data in `data/`.
- Extend the narrative adapter with sentiment and semantic embeddings.
- Add notebook summaries in `notebooks/` for geometry and operator behavior.
