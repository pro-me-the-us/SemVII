import numpy as np
import pandas as pd
from typing import List
from pathlib import Path as FilePath

from src.experiment.runner import RunResult
from src.experiment.config import get_scenario_folder

def aggregate_results(results: List[RunResult]) -> pd.DataFrame:
    """Aggregate results into mean ± std and medians per (scenario, solver, alpha, beta)."""
    if not results:
        return pd.DataFrame()
        
    records = []
    for r in results:
        records.append({
            "scenario": r.scenario_id,
            "K": r.num_agents,
            "solver": r.solver,
            "alpha": r.alpha,
            "beta": r.beta,
            "ell": r.total_length,
            "Sp": r.total_Sp,
            "tau": r.total_tau,
            "time": r.runtime,
            "converged": float(r.converged),
            "iterations_outer": r.iterations_outer,
            "iterations_inner": r.iterations_inner,
            "num_paths_generated": r.num_paths_generated,
        })
        
    df_raw = pd.DataFrame(records)
    
    grouped = df_raw.groupby(["scenario", "K", "solver", "alpha", "beta"])
    
    rows = []
    for (scenario, K, solver, alpha, beta), group in grouped:
        # Filter to only converged runs for metrics (ell, Sp, tau)
        valid_group = group[group["converged"] == 1.0]
        
        if len(valid_group) > 0:
            tau_vals = valid_group["tau"].values
            tau_log = np.log(np.maximum(tau_vals, 0.0) + 1.0)
            
            row = {
                "scenario": scenario,
                "K": K,
                "solver": solver,
                "alpha": alpha,
                "beta": beta,
                "ell_mean": valid_group["ell"].mean(),
                "ell_std": valid_group["ell"].std(),
                "Sp_mean": valid_group["Sp"].mean(),
                "Sp_std": valid_group["Sp"].std(),
                "tau_mean": valid_group["tau"].mean(),
                "tau_std": valid_group["tau"].std(),
                "tau_median": valid_group["tau"].median(),
                "tau_q25": valid_group["tau"].quantile(0.25),
                "tau_q75": valid_group["tau"].quantile(0.75),
                "tau_geomean": float(np.exp(np.mean(tau_log)) - 1.0),
                "time_mean": group["time"].mean(),  # Runtime averaged over all runs
                "time_std": group["time"].std(),
                "converged_rate": group["converged"].mean(),
                "iter_outer_mean": group["iterations_outer"].mean(),
                "iter_inner_mean": group["iterations_inner"].mean(),
                "p_bar_size_mean": group["num_paths_generated"].mean(),
            }
        else:
            row = {
                "scenario": scenario,
                "K": K,
                "solver": solver,
                "alpha": alpha,
                "beta": beta,
                "ell_mean": np.nan,
                "ell_std": np.nan,
                "Sp_mean": np.nan,
                "Sp_std": np.nan,
                "tau_mean": np.nan,
                "tau_std": np.nan,
                "tau_median": np.nan,
                "tau_q25": np.nan,
                "tau_q75": np.nan,
                "tau_geomean": np.nan,
                "time_mean": group["time"].mean(),
                "time_std": group["time"].std(),
                "converged_rate": group["converged"].mean(),
                "iter_outer_mean": group["iterations_outer"].mean(),
                "iter_inner_mean": group["iterations_inner"].mean(),
                "p_bar_size_mean": group["num_paths_generated"].mean(),
            }
        rows.append(row)
        
    df_agg = pd.DataFrame(rows).fillna(0.0)
    return df_agg

def save_aggregated_results(df: pd.DataFrame, results_dir: str = "results") -> None:
    """Save aggregated results to per-scenario CSV files."""
    for scenario in df["scenario"].unique():
        folder = get_scenario_folder(scenario)
        path_dir = FilePath(results_dir) / folder
        path_dir.mkdir(parents=True, exist_ok=True)
        
        df_scenario = df[df["scenario"] == scenario]
        
        df_cbs = df_scenario[df_scenario["solver"] == "cbs"]
        if not df_cbs.empty:
            df_cbs.to_csv(path_dir / "cbs_results.csv", index=False)
            df_cbs.to_csv(FilePath(results_dir) / f"cbs_results_{folder}.csv", index=False)
            
        df_qa = df_scenario[df_scenario["solver"] == "qa"]
        if not df_qa.empty:
            df_qa.to_csv(path_dir / "qa_results.csv", index=False)
            df_qa.to_csv(FilePath(results_dir) / f"qa_results_{folder}.csv", index=False)

def format_table_i(df: pd.DataFrame) -> str:
    """Format aggregated results as a readable table matching Table I format."""
    lines = []
    lines.append(f"{'Scenario (K)':<15} | {'Solver':<6} | {'(a, b)':<12} | {'l_mean +/- std':<15} | {'S_p mean +/- std':<15} | {'tau mean (median)':<20} | {'Time (s)':<12} | {'Conv. Rate':<10} | {'Outer/Inner/Pbar':<16}")
    lines.append("-" * 140)
    
    for _, row in df.iterrows():
        scenario_k = f"{row['scenario']} ({row['K']})"
        solver = row['solver'].upper()
        ab = f"({row['alpha']:.2f}, {row['beta']:.2f})"
        
        ell = f"{row['ell_mean']:.1f} +/- {row['ell_std']:.1f}"
        Sp = f"{row['Sp_mean']:.1f} +/- {row['Sp_std']:.1f}"
        tau = f"{row['tau_mean']:.1f} ({row['tau_median']:.1f})"
        time = f"{row['time_mean']:.2f} +/- {row['time_std']:.2f}"
        conv = f"{row['converged_rate']*100:.0f}%"
        iters = f"{row['iter_outer_mean']:.1f}/{row['iter_inner_mean']:.1f}/{row['p_bar_size_mean']:.1f}"
        
        lines.append(f"{scenario_k:<15} | {solver:<6} | {ab:<12} | {ell:<15} | {Sp:<15} | {tau:<20} | {time:<12} | {conv:<10} | {iters:<16}")
        
    return "\n".join(lines)
