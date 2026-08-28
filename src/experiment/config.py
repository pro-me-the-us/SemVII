"""Experiment configuration for MAPF Turn-Aware QUBO-and-Price.

Defines the full (scenario, alpha, beta, seed, solver) matrix for all experiments.
Matches Table I of the paper for three scenarios.
"""
from dataclasses import dataclass
from typing import List, Literal

@dataclass(frozen=True)
class RunConfig:
    """Configuration for a single experiment run."""
    scenario_id: str
    alpha: float
    beta: float
    seed: int
    solver: Literal["cbs", "qa"]

# QUBO solver parameters
THETA_OH: float = 20.0      # One-hot penalty (must be > max conflicts * theta_c to prevent dropping agents)
THETA_C: float = 2.5       # Conflict penalty (> 2)
SA_NUM_READS: int = 100    # SA samples per QUBO solve (reduced for faster execution)
MAX_OUTER_ITER: int = 30   # Branch-and-price outer loop cap (fail fast on degenerate cases)
MAX_INNER_ITER: int = 30   # Column generation inner loop cap
MAX_TIME_HORIZON: int = 30 # Max timestep for spatio-temporal graph

NUM_SEEDS: int = 10        # Seeds 0..9

def generate_run_configs() -> List[RunConfig]:
    """Generate all RunConfig objects for the full experiment.
    
    Scenarios and their (alpha, beta) configs:
    - Empty 6x6-2a:  QA with (0.50, 0.50). CBS once (alpha=1, beta=0).
    - Empty 6x6-4a:  QA with (1.00, 0.00), (0.75, 0.25), (0.50, 0.50). CBS once.
    - Room 6x6-2a:   QA with (1.00, 0.00), (0.50, 0.50). CBS once.
    
    CBS is always (alpha=1.0, beta=0.0) regardless of scenario config.
    Each config is run with seeds 0..9.
    
    Returns list of RunConfig objects.
    """
    configs = []
    
    scenarios = {
        "Empty 6x6-2a": [(0.50, 0.50)],
        "Empty 6x6-4a": [(1.00, 0.00), (0.75, 0.25), (0.50, 0.50)],
        "Room 6x6-2a": [(1.00, 0.00), (0.50, 0.50)],
    }
    
    for scenario_id, qa_params in scenarios.items():
        # CBS run (always once with alpha=1.0, beta=0.0)
        for seed in range(NUM_SEEDS):
            configs.append(RunConfig(scenario_id, 1.0, 0.0, seed, "cbs"))
            
        # QA runs
        for alpha, beta in qa_params:
            for seed in range(NUM_SEEDS):
                configs.append(RunConfig(scenario_id, alpha, beta, seed, "qa"))
                
    return configs

def get_scenario_folder(scenario_id: str) -> str:
    """Convert scenario_id to folder name. e.g. 'Empty 6x6-2a' -> 'empty_6x6_2a'"""
    return scenario_id.lower().replace(" ", "_").replace("-", "_")

def configs_for_scenario(scenario_id: str) -> List[RunConfig]:
    """Get all run configs for a specific scenario."""
    all_configs = generate_run_configs()
    return [c for c in all_configs if c.scenario_id == scenario_id]
