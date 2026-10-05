"""Regenerate runtime plots with log-scale y-axis from existing CSVs."""
import pandas as pd
from pathlib import Path
from src.report.plots import plot_runtime

scenarios = [
    ("Empty 6x6-2a", "empty_6x6_2a"),
    ("Empty 6x6-4a", "empty_6x6_4a"),
    ("Room 6x6-2a",  "room_6x6_2a"),
]

results_dir = Path("results")

for scenario_id, folder in scenarios:
    dfs = []
    for solver in ["cbs", "qa"]:
        f = results_dir / f"{solver}_results_{folder}.csv"
        if f.exists():
            dfs.append(pd.read_csv(f))
    if not dfs:
        print(f"No data for {scenario_id}, skipping.")
        continue
    df = pd.concat(dfs, ignore_index=True)
    # rename 'scenario' col if present (metrics saves as 'scenario')
    if 'scenario' not in df.columns and 'scenario_id' in df.columns:
        df.rename(columns={'scenario_id': 'scenario'}, inplace=True)
    output_dir = results_dir / folder / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_runtime(df, scenario_id, str(output_dir))
    print(f"Regenerated runtime plot for {scenario_id} -> {output_dir}/runtime.png")
