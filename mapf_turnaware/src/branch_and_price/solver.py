"""Turn-Aware QUBO-and-Price solver (Algorithm 1).

Full branch-and-price driver implementing the outer conflict loop
and inner column generation loop with QUBO-based path selection.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

from src.graph.grids import Scenario, AgentSpec, Coord
from src.turncost.turncost import (
    STNode, Path, compute_path_costs, normalize_path_set, normalized_costs
)
from src.graph.spatiotemporal import build_augmented_graph, dijkstra_augmented
from src.qubo.qubo_builder import (
    RestrictedPathSet, QUBOInstance, build_qubo, 
    build_conflict_matrix, detect_path_conflicts
)
from src.qubo.qubo_solver import solve_qubo
from src.qubo.pricing import extract_omega, pricing_step
from src.cbs.low_level import space_time_astar

@dataclass
class Solution:
    """Solution from the QUBO-and-Price solver."""
    paths: Dict[int, Path]     # agent_id -> selected path
    total_length: float        # sum of path lengths
    total_Sp: float            # sum of Sp values
    total_tau: float           # sum of tau_exp values  
    converged: bool            # True if solution is verified optimal (not iteration-capped)
    iterations_outer: int      # number of outer loop iterations
    iterations_inner: int      # total inner loop iterations across all outer
    num_paths_generated: int   # total paths in P_bar at termination
    runtime: float             # wall-clock seconds


def _init_path_set(scenario: Scenario, max_time: int) -> Tuple[List[Path], List[int]]:
    """Initialize P_bar with one shortest path per agent."""
    paths = []
    agent_of = []
    for agent in scenario.agents:
        path = space_time_astar(scenario.grid, agent, set(), max_time=max_time)
        if path is not None:
            paths.append(path)
            agent_of.append(agent.agent_id)
        else:
            raise ValueError(f"No initial path found for agent {agent.agent_id}")
    return paths, agent_of


def _build_restricted_path_set(
    paths: List[Path], 
    agent_of: List[int],
    alpha: float,
    beta: float
) -> RestrictedPathSet:
    """Build a RestrictedPathSet from current paths."""
    ell_max, tau_max = normalize_path_set(paths)
    ell_hat_list, tau_hat_list = normalized_costs(paths, ell_max, tau_max)
    conflict_matrix = build_conflict_matrix(paths)
    
    return RestrictedPathSet(
        paths=paths,
        agent_of=agent_of,
        ell_hat=np.array(ell_hat_list, dtype=np.float32),
        tau_hat=np.array(tau_hat_list, dtype=np.float32),
        conflict_matrix=conflict_matrix
    )


def _select_paths_from_solution(
    z: np.ndarray,
    path_set: RestrictedPathSet,
    scenario: Scenario
) -> Dict[int, Path]:
    """Extract one path per agent from QUBO solution z.
    If multiple paths selected for one agent, pick lowest cost.
    If no path selected for an agent, use the first path for that agent.
    """
    selected = {}
    costs = {}
    
    for i, selected_flag in enumerate(z):
        if selected_flag == 1:
            p = path_set.paths[i]
            agent_id = p.agent_id
            # cost can be approximated by length + tau_exp or similar.
            # Using raw objective since alpha/beta isn't directly passed, but tau_exp is a good proxy.
            # Let's just use tau_exp + length as a tie-breaker.
            cost = p.length + p.tau_exp
            if agent_id not in selected or cost < costs[agent_id]:
                selected[agent_id] = p
                costs[agent_id] = cost
                
    # Fallback for unselected agents
    for agent in scenario.agents:
        agent_id = agent.agent_id
        if agent_id not in selected:
            for p in path_set.paths:
                if p.agent_id == agent_id:
                    selected[agent_id] = p
                    break
                    
    return selected


def _detect_solution_conflicts(
    selected_paths: Dict[int, Path]
) -> List[Tuple[int, int, str, Coord, Optional[Coord], int]]:
    """Detect conflicts in the selected solution paths.
    Returns list of (agent_a, agent_b, type, v, v2, t) tuples.
    """
    conflicts = []
    agent_ids = list(selected_paths.keys())
    
    def get_pos(path: Path, t: int) -> Coord:
        if not path.nodes:
            return (-1, -1)
        for node in path.nodes:
            if node.t == t:
                return node.v
        return path.nodes[-1].v

    for i in range(len(agent_ids)):
        a_id = agent_ids[i]
        path_a = selected_paths[a_id]
        
        for j in range(i+1, len(agent_ids)):
            b_id = agent_ids[j]
            path_b = selected_paths[b_id]
            
            len_a = len(path_a.nodes)
            len_b = len(path_b.nodes)
            max_t = max(path_a.nodes[-1].t if len_a > 0 else 0, path_b.nodes[-1].t if len_b > 0 else 0)
            
            for t in range(max_t + 1):
                pos_a = get_pos(path_a, t)
                pos_b = get_pos(path_b, t)
                
                # Vertex conflict
                if pos_a == pos_b:
                    conflicts.append((a_id, b_id, "vertex", pos_a, None, t))
                    continue
                    
                # Edge conflict
                if t > 0:
                    pos_a_prev = get_pos(path_a, t-1)
                    pos_b_prev = get_pos(path_b, t-1)
                    if pos_a == pos_b_prev and pos_b == pos_a_prev:
                        conflicts.append((a_id, b_id, "edge", pos_a_prev, pos_a, t))
                        
    return conflicts


def _compute_omega_from_conflicts(
    selected_paths: Dict[int, Path],
    conflicts: List[Tuple[int, int, str, Coord, Optional[Coord], int]],
    theta_c: float
) -> Dict[Tuple[Coord, Coord, int], float]:
    """Compute omega penalties from detected conflicts in the selected solution.
    
    For each conflict, penalizes the incoming edges of both conflicting agents
    so that the pricing subproblem will find alternative routes.
    """
    omega: Dict[Tuple[Coord, Coord, int], float] = {}
    
    def get_pos(path: Path, t: int) -> Coord:
        if not path.nodes:
            return (-1, -1)
        for node in path.nodes:
            if node.t == t:
                return node.v
        return path.nodes[-1].v
    
    for (agent_a, agent_b, ctype, v, v2, t) in conflicts:
        path_a = selected_paths[agent_a]
        path_b = selected_paths[agent_b]
        
        if ctype == "vertex":
            # Penalize incoming edges to (v, t) for both agents
            if t > 0:
                prev_a = get_pos(path_a, t - 1)
                edge_a = (prev_a, v, t - 1)
                omega[edge_a] = omega.get(edge_a, 0.0) + theta_c
                
                prev_b = get_pos(path_b, t - 1)
                edge_b = (prev_b, v, t - 1)
                omega[edge_b] = omega.get(edge_b, 0.0) + theta_c
            else:
                # t==0: penalize outgoing edges from start
                next_a = get_pos(path_a, 1)
                edge_a = (v, next_a, 0)
                omega[edge_a] = omega.get(edge_a, 0.0) + theta_c
                
                next_b = get_pos(path_b, 1)
                edge_b = (v, next_b, 0)
                omega[edge_b] = omega.get(edge_b, 0.0) + theta_c
                
        elif ctype == "edge":
            # v=from_pos, v2=to_pos for agent_a; reversed for agent_b
            edge_a = (v, v2, t - 1)
            edge_b = (v2, v, t - 1)
            omega[edge_a] = omega.get(edge_a, 0.0) + theta_c
            omega[edge_b] = omega.get(edge_b, 0.0) + theta_c
    
    return omega


def turn_aware_qubo_and_price(
    scenario: Scenario,
    alpha: float,
    beta: float,
    seed: int,
    sa_reads: int = 200,
    theta_oh: float = 1.5,
    theta_c: float = 2.5,
    max_outer_iter: int = 50,
    max_inner_iter: int = 30,
    max_time_horizon: int = 30
) -> Optional[Solution]:
    """Solve MAPF using Turn-Aware QUBO-and-Price (Algorithm 1).
    
    Outer loop: detect conflicts in selected solution, accumulate omega penalties,
    run inner column-generation loop to find alternative paths, re-solve QUBO.
    Inner loop: pricing subproblem generates new columns (paths) with reduced cost
    under current omega, then QUBO is re-solved with enlarged path pool.
    """
    start_time = time.time()
    logger.info(f"Starting turn_aware_qubo_and_price for scenario {scenario.scenario_id}")
    
    # 1. INITIALIZATION
    try:
        paths, agent_of = _init_path_set(scenario, max_time_horizon)
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        return None
    
    # Accumulated omega: persists and grows across outer iterations
    omega: Dict[Tuple[Coord, Coord, int], float] = {}
    
    iterations_inner_total = 0
    converged = False
    z = None
    path_set = None
    
    # 2. OUTER LOOP
    for outer_iter in range(max_outer_iter):
        logger.info(f"Outer iteration {outer_iter+1}/{max_outer_iter}")
        
        # a. Inner column-generation loop with current omega
        for inner_iter in range(max_inner_iter):
            logger.debug(f"  Inner iteration {inner_iter+1}/{max_inner_iter}")
            iterations_inner_total += 1
            
            new_paths_added = 0
            ell_max, tau_max = normalize_path_set(paths)
            
            for agent in scenario.agents:
                new_path = pricing_step(
                    scenario=scenario,
                    agent=agent,
                    alpha=alpha,
                    beta=beta,
                    omega=omega,
                    ell_max=ell_max,
                    tau_max=tau_max,
                    existing_paths=paths,
                    max_time=max_time_horizon
                )
                
                if new_path is not None:
                    paths.append(new_path)
                    agent_of.append(agent.agent_id)
                    new_paths_added += 1
            
            if new_paths_added == 0:
                break
        
        # b. Build QUBO from current path set P_bar and solve
        path_set = _build_restricted_path_set(paths, agent_of, alpha, beta)
        qubo_inst = build_qubo(path_set, alpha, beta, theta_oh, theta_c)
        z = solve_qubo(qubo_inst.Q, num_reads=sa_reads, seed=seed)
        
        # c. Extract selected solution (with fallbacks for unselected agents)
        selected_paths = _select_paths_from_solution(z, path_set, scenario)
        
        # d. Check for conflicts in the actual selected solution
        conflicts = _detect_solution_conflicts(selected_paths)
        
        if not conflicts:
            logger.info("No conflicts found. Converged.")
            converged = True
            break
        
        logger.info(f"  Found {len(conflicts)} conflict(s), accumulating omega penalties")
        
        # e. Accumulate omega from detected conflicts
        #    This increases penalties on conflict edges so the next pricing round
        #    generates alternative paths that avoid these locations/times.
        conflict_omega = _compute_omega_from_conflicts(selected_paths, conflicts, theta_c)
        for edge, val in conflict_omega.items():
            omega[edge] = omega.get(edge, 0.0) + val
    
    if not converged:
        logger.warning("Reached maximum outer iterations without converging.")
    
    # 3. SOLUTION EXTRACTION
    final_selected_paths = _select_paths_from_solution(z, path_set, scenario)
    
    # Verify conflict-free status of final solution
    final_conflicts = _detect_solution_conflicts(final_selected_paths)
    if final_conflicts and converged:
        logger.error("BUG: marked as converged but final solution has conflicts")
        converged = False
    
    total_length = sum(p.length for p in final_selected_paths.values())
    total_Sp = sum(p.Sp for p in final_selected_paths.values())
    total_tau = sum(p.tau_exp for p in final_selected_paths.values())
    
    runtime = time.time() - start_time
    
    return Solution(
        paths=final_selected_paths,
        total_length=total_length,
        total_Sp=total_Sp,
        total_tau=total_tau,
        converged=converged,
        iterations_outer=outer_iter + 1,
        iterations_inner=iterations_inner_total,
        num_paths_generated=len(paths),
        runtime=runtime
    )

