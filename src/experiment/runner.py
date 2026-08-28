"""Experiment runner for MAPF solvers.

Executes both CBS and QUBO-and-Price solvers, collects results,
and serializes per-run data for aggregation.
"""
import json
import os
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path as FilePath
from typing import Dict, List, Optional

from src.graph.grids import get_scenario, Scenario
from src.turncost.turncost import compute_path_costs, Path
from src.cbs.cbs import cbs_solve
from src.branch_and_price.solver import turn_aware_qubo_and_price
from src.experiment.config import (
    RunConfig, generate_run_configs, get_scenario_folder,
    THETA_OH, THETA_C, SA_NUM_READS, MAX_OUTER_ITER, MAX_INNER_ITER, MAX_TIME_HORIZON
)

logger = logging.getLogger(__name__)

@dataclass 
class RunResult:
    """Result from a single experiment run."""
    scenario_id: str
    solver: str
    alpha: float
    beta: float
    seed: int
    total_length: float       # sum of path lengths across agents
    total_Sp: float           # sum of Sp across agents  
    total_tau: float          # sum of tau_exp across agents
    runtime: float            # wall-clock seconds
    converged: bool           # True if solver converged (always True for CBS)
    num_agents: int
    iterations_outer: int = 0
    iterations_inner: int = 0
    num_paths_generated: int = 0
    paths: Optional[Dict] = None     # serializable path data for raw output

def _serialize_paths(paths: List[Path]) -> Dict:
    """Serialize list of paths to JSON-friendly dict."""
    if paths is None:
        return {}
    res = {}
    for p in paths:
        res[p.agent_id] = {
            "length": p.length,
            "Sp": p.Sp,
            "tau_exp": p.tau_exp,
            "tau_linear": p.tau_linear,
            "nodes": [{"v": n.v, "t": n.t} for n in p.nodes]
        }
    return res

def run_single(config: RunConfig, results_dir: str = "results") -> RunResult:
    """Execute a single experiment run."""
    scenario = get_scenario(config.scenario_id)
    
    start_time = time.time()
    converged = False
    iterations_outer = 0
    iterations_inner = 0
    num_paths_generated = 0
    
    paths: List[Path] = []
    if config.solver == "cbs":
        cbs_result = cbs_solve(scenario)
        converged = True
        iterations_outer = 1
        iterations_inner = 0
        num_paths_generated = len(scenario.agents)
        if cbs_result is not None:
            paths = list(cbs_result.values())
            for p in paths:
                compute_path_costs(p)
        else:
            logger.error(f"CBS returned None for {config.scenario_id}")
            paths = []
    elif config.solver == "qa":
        solution = turn_aware_qubo_and_price(
            scenario=scenario,
            alpha=config.alpha,
            beta=config.beta,
            seed=config.seed,
            theta_oh=THETA_OH,
            theta_c=THETA_C,
            sa_reads=SA_NUM_READS,
            max_outer_iter=MAX_OUTER_ITER,
            max_inner_iter=MAX_INNER_ITER,
            max_time_horizon=MAX_TIME_HORIZON
        )
        if solution is not None:
            paths = list(solution.paths.values())
            converged = solution.converged
            iterations_outer = solution.iterations_outer
            iterations_inner = solution.iterations_inner
            num_paths_generated = solution.num_paths_generated
        else:
            logger.error(f"QA returned None for {config.scenario_id}")
            paths = []
    else:
        raise ValueError(f"Unknown solver: {config.solver}")
        
    runtime = time.time() - start_time
    
    total_length = sum(p.length for p in paths)
    total_Sp = sum(p.Sp for p in paths)
    total_tau = sum(p.tau_exp for p in paths)
    
    result = RunResult(
        scenario_id=config.scenario_id,
        solver=config.solver,
        alpha=config.alpha,
        beta=config.beta,
        seed=config.seed,
        total_length=total_length,
        total_Sp=total_Sp,
        total_tau=total_tau,
        runtime=runtime,
        converged=converged,
        num_agents=len(scenario.agents),
        iterations_outer=iterations_outer,
        iterations_inner=iterations_inner,
        num_paths_generated=num_paths_generated,
        paths=_serialize_paths(paths)
    )
    
    save_raw_result(result, results_dir)
    return result

def save_raw_result(result: RunResult, results_dir: str = "results") -> None:
    """Save raw result data to JSON file in the appropriate scenario folder."""
    folder = get_scenario_folder(result.scenario_id)
    path_dir = FilePath(results_dir) / folder / "raw"
    path_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{result.solver}_{result.alpha:.2f}_{result.beta:.2f}_{result.seed}.json"
    filepath = path_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)

def run_all(configs: Optional[List[RunConfig]] = None, results_dir: str = "results") -> List[RunResult]:
    """Run all experiment configurations."""
    if configs is None:
        configs = generate_run_configs()
        
    results = []
    for i, config in enumerate(configs):
        logger.info(f"Running config {i+1}/{len(configs)}: {config}")
        res = run_single(config, results_dir)
        results.append(res)
        
    return results
