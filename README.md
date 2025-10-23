# CEH-Style Cost & Risk Model (Quickstart)

A lightweight Python model that mimics **NASA CEH 4.0** workflows:
- **Parametric CERs** (cost-estimating relationships) per subsystem
- **Deterministic P50** costs
- **Monte Carlo** risk bands (P10/P50/P90)
- **Sensitivity analysis** (driver elasticities)
- **CSV in / CSV out**, easy to open in Excel

> ⚠️ Coefficients in this quickstart are **illustrative**. Replace with the ones you use in your project.

## Files
- `subsystems.csv` — inputs (edit this)
- `ceh_model.py` — main script
- `requirements.txt` — dependencies
- `run_example.sh` — helper to run locally (Mac/Linux)

## Quick Run
```bash
python3 ceh_model.py
```
Outputs are written to `out/`:
- `summary.csv` — P10/P50/P90 and by-subsystem table
- `sensitivity.csv` — ranked cost impact of ±5% driver nudges
