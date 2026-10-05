"""Plotting module for MAPF experiment results.

Generates per-scenario plots:
1. length_vs_turncost.png — scatter/line of ℓ_mean vs τ_mean (log scale) across β sweep
2. sp_by_beta.png — bar chart of Sp_mean (± std) per β
3. runtime.png — bar chart of runtime (± std) per solver/config
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path as FilePath
from typing import Optional
import logging

from src.experiment.config import get_scenario_folder

logger = logging.getLogger(__name__)

def plot_length_vs_turncost(
    df: pd.DataFrame,
    scenario_id: str,
    output_dir: str
) -> None:
    """Plot ℓ_mean (x) vs τ_mean (y, log scale) for QA across β sweep.
    Overlay the single CBS point for reference.
    
    df should be the aggregated DataFrame from metrics.py, filtered to this scenario.
    
    Style:
    - QA points connected with line, labeled by β value
    - CBS as a star marker in different color
    - Log scale on y-axis for τ
    - Clear legend, grid, title
    - Save to output_dir/length_vs_turncost.png
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_qa = df[df['solver'] == 'qa'].sort_values('beta')
    df_cbs = df[df['solver'] == 'cbs']
    
    if not df_qa.empty:
        ax.plot(df_qa['ell_mean'], df_qa['tau_mean'], marker='o', linestyle='-', color='blue', label='QA (varying β)')
        for _, row in df_qa.iterrows():
            ax.annotate(f"β={row['beta']:.2f}", (row['ell_mean'], row['tau_mean']), textcoords="offset points", xytext=(0,10), ha='center')
            
    if not df_cbs.empty:
        cbs_row = df_cbs.iloc[0]
        ax.scatter(cbs_row['ell_mean'], cbs_row['tau_mean'], marker='*', color='red', s=200, label='CBS', zorder=5)
        
    ax.set_yscale('log')
    ax.set_xlabel('Mean Path Length (ℓ_mean)')
    ax.set_ylabel('Mean Turn Cost (τ_mean) - Log Scale')
    ax.set_title(f'Length vs Turn Cost for {scenario_id}')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    out_path = FilePath(output_dir) / 'length_vs_turncost.png'
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_sp_by_beta(
    df: pd.DataFrame,
    scenario_id: str,
    output_dir: str
) -> None:
    """Bar chart of Sp_mean (± std as error bars) per β value.
    Shows both QA and CBS side by side.
    
    Style:
    - Grouped bar chart: CBS bar + QA bars per β
    - Error bars showing ± std
    - Clear labels, title, legend
    - Save to output_dir/sp_by_beta.png
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_qa = df[df['solver'] == 'qa'].sort_values('beta')
    df_cbs = df[df['solver'] == 'cbs']
    
    labels = []
    means = []
    stds = []
    colors = []
    
    if not df_cbs.empty:
        labels.append("CBS")
        means.append(df_cbs.iloc[0]['Sp_mean'])
        stds.append(df_cbs.iloc[0]['Sp_std'])
        colors.append('red')
        
    for _, row in df_qa.iterrows():
        labels.append(f"QA β={row['beta']:.2f}")
        means.append(row['Sp_mean'])
        stds.append(row['Sp_std'])
        colors.append('blue')
        
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Spatial Preference Cost (Sp_mean)')
    ax.set_title(f'Spatial Preference Cost by β for {scenario_id}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path = FilePath(output_dir) / 'sp_by_beta.png'
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_runtime(
    df: pd.DataFrame,
    scenario_id: str,
    output_dir: str
) -> None:
    """Bar chart of runtime mean (± std) per solver/config.
    
    Style:
    - One bar per (solver, α, β) config
    - Error bars showing ± std
    - Clear labels, title
    - Save to output_dir/runtime.png
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    means = []
    stds = []
    colors = []
    
    for _, row in df.iterrows():
        solver = row['solver'].upper()
        if solver == 'CBS':
            labels.append('CBS')
            colors.append('red')
        else:
            labels.append(f"QA (α={row['alpha']:.2f}, β={row['beta']:.2f})")
            colors.append('blue')
        means.append(row['time_mean'])
        stds.append(row['time_std'])
        
    x = np.arange(len(labels))
    # Clip to small positive value to avoid log(0) for zero-runtime edge cases
    log_safe_means = [max(m, 1e-6) for m in means]
    ax.bar(x, log_safe_means, color=colors, capsize=5, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('Mean Runtime (s) — Log Scale')
    ax.set_title(f'Runtime Comparison for {scenario_id}')
    ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    
    out_path = FilePath(output_dir) / 'runtime.png'
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def generate_all_plots(
    df: pd.DataFrame,
    results_dir: str = "results"
) -> None:
    """Generate all plots for all scenarios.
    
    For each unique scenario in df:
    1. Create output_dir = results_dir/<scenario_folder>/plots/
    2. Generate all three plot types
    """
    for scenario_id in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario_id]
        
        folder = get_scenario_folder(scenario_id)
        output_dir = FilePath(results_dir) / folder / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_length_vs_turncost(scenario_df, scenario_id, str(output_dir))
        plot_sp_by_beta(scenario_df, scenario_id, str(output_dir))
        plot_runtime(scenario_df, scenario_id, str(output_dir))
        logger.info(f"Generated plots for scenario {scenario_id} at {output_dir}")
