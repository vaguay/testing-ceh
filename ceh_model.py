#!/usr/bin/env python3
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

# ---------------------------
# CER REGISTRY (illustrative)
# Replace A and exponents with your project-specific values.
# sigma_ln is lognormal sigma for uncertainty sampling.
# ---------------------------

CER = {
    "Power": {
        "A": 1.85,
        "drivers": {"mass": 0.62, "peak_power": 0.38},
        "sigma_ln": 0.25,
    },
    "Thermal": {
        "A": 0.74,
        "drivers": {"mass": 0.55},
        "sigma_ln": 0.30,
    },
    "Avionics": {
        "A": 1.10,
        "drivers": {"mass": 0.50, "part_count": 0.22},
        "sigma_ln": 0.28,
    },
    "Comms": {
        "A": 0.85,
        "drivers": {"mass": 0.42, "peak_power": 0.35},
        "sigma_ln": 0.26,
    },
    "Payload": {
        "A": 2.10,
        "drivers": {"mass": 0.70, "part_count": 0.20},
        "sigma_ln": 0.35,
    },
}

# Recommended default nudge for sensitivity (%)
SENS_NUDGE = 0.05
N_MC = 10000  # Monte Carlo draws

def cer_cost(row: pd.Series) -> float:
    meta = CER[row["type"]]
    val = meta["A"]
    for drv, exp in meta["drivers"].items():
        x = max(float(row.get(drv, 0.0)), 1e-9)  # avoid zeros
        val *= (x ** exp)
    return float(val)

def deterministic_costs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cost_p50"] = out.apply(cer_cost, axis=1)
    return out

def monte_carlo_totals(df_p50: pd.DataFrame, n=N_MC, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    totals = np.zeros(n)
    types = df_p50["type"].values
    p50s = df_p50["cost_p50"].values
    sigmas = np.array([CER[t]["sigma_ln"] for t in types])
    for i in range(n):
        mults = rng.lognormal(mean=0.0, sigma=sigmas)
        totals[i] = np.sum(p50s * mults)
    return totals

def sensitivity_analysis(df: pd.DataFrame, baseline_total: float, nudge=SENS_NUDGE) -> pd.DataFrame:
    rows = []
    for idx, r in df.iterrows():
        meta = CER[r["type"]]
        for drv in meta["drivers"].keys():
            for direction in ["up","down"]:
                factor = (1+nudge) if direction=="up" else (1-nudge)
                r2 = r.copy()
                r2[drv] = r2[drv] * factor
                # recompute this row's cost; others unchanged
                base_row_cost = r["cost_p50"]
                new_row_cost = cer_cost(r2)
                delta = (new_row_cost - base_row_cost)
                # approximate mission total change by row delta
                total_new = baseline_total + delta
                impact = (total_new - baseline_total)
                rows.append({
                    "subsystem": r["subsystem"],
                    "type": r["type"],
                    "driver": drv,
                    "direction": direction,
                    "impact_abs": impact,
                    "impact_pct_of_total": impact / baseline_total if baseline_total else 0.0,
                })
    sens = pd.DataFrame(rows)
    # aggregate up/down into an absolute envelope per driver
    agg = (sens
           .groupby(["type","driver"], as_index=False)
           .agg(max_abs_impact=("impact_abs", lambda x: float(np.max(np.abs(x)))),
                max_pct_of_total=("impact_pct_of_total", lambda x: float(np.max(np.abs(x)))))
          )
    agg = agg.sort_values("max_abs_impact", ascending=False).reset_index(drop=True)
    return agg, sens

def main():
    df = pd.read_csv("subsystems.csv")
    df_p50 = deterministic_costs(df)
    mission_p50 = df_p50["cost_p50"].sum()

    totals = monte_carlo_totals(df_p50)
    p10, p50, p90 = np.percentile(totals, [10,50,90])

    # Sensitivity
    sens_ranked, sens_full = sensitivity_analysis(df_p50, baseline_total=mission_p50)

    # OUTPUTS
    os.makedirs("out", exist_ok=True)
    df_p50.to_csv("out/by_subsystem_p50.csv", index=False)
    pd.DataFrame([{"P10":p10, "P50":p50, "P90":p90, "Mission_P50_Sum": mission_p50}]).to_csv("out/summary.csv", index=False)
    sens_ranked.to_csv("out/sensitivity.csv", index=False)
    sens_full.to_csv("out/sensitivity_full.csv", index=False)

    print("=== SUMMARY ===")
    print(f"P10: {p10:,.2f}  P50: {p50:,.2f}  P90: {p90:,.2f}")
    print(f"Mission P50 (deterministic sum): {mission_p50:,.2f}")
    print("\nTop sensitivity drivers:")
    print(sens_ranked.head(8).to_string(index=False))

if __name__ == "__main__":
    import os
    main()
