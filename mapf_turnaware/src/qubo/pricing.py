"""Pricing subproblem for QUBO-and-Price column generation.

Implements:
- Direction-augmented shortest path pricing (paper §IV-E)
- Lagrangian multiplier extraction from QUBO solution (reconstructed eq. 23)
- Reduced cost computation for new path generation
"""
from typing import Dict, Tuple, List, Optional
import numpy as np

from src.graph.grids import Coord, AgentSpec, Scenario
from src.turncost.turncost import Path
from src.qubo.qubo_builder import RestrictedPathSet
from src.graph.spatiotemporal import build_augmented_graph, dijkstra_augmented

def extract_omega(
    z: np.ndarray,
    path_set: RestrictedPathSet,
    theta_c: float = 2.5
) -> Dict[Tuple[Coord, Coord, int], float]:
    """Extract Lagrangian multipliers omega from QUBO solution.
    
    Penalizes edges involved in vertex or edge conflicts between selected paths.
    
    Args:
        z: Binary solution vector from QUBO solver
        path_set: The restricted path set
        theta_c: Conflict penalty strength
    
    Returns:
        Dict mapping (v1, v2, t) -> omega value (>= 0 always)
    """
    omega: Dict[Tuple[Coord, Coord, int], float] = {}
    selected_indices = np.where(z == 1)[0]
    
    def get_node_pos(path: Path, t: int) -> Coord:
        if not path.nodes:
            return (-1, -1)
        for node in path.nodes:
            if node.t == t:
                return node.v
        return path.nodes[-1].v

    C = path_set.conflict_matrix
    # For each pair of conflicting paths that are selected
    for idx_1 in range(len(selected_indices)):
        i = selected_indices[idx_1]
        for idx_2 in range(idx_1 + 1, len(selected_indices)):
            j = selected_indices[idx_2]
            
            if C[i, j] == 1.0:
                path_i = path_set.paths[i]
                path_j = path_set.paths[j]
                
                max_t = max(path_i.nodes[-1].t if path_i.nodes else 0,
                            path_j.nodes[-1].t if path_j.nodes else 0)
                
                for t in range(max_t + 1):
                    pos_i = get_node_pos(path_i, t)
                    pos_j = get_node_pos(path_j, t)
                    
                    # 1. Vertex conflict at (pos_i, t)
                    if pos_i == pos_j:
                        if t > 0:
                            pos_i_prev = get_node_pos(path_i, t-1)
                            edge_i = (pos_i_prev, pos_i, t-1)
                            omega[edge_i] = omega.get(edge_i, 0.0) + theta_c
                            
                            pos_j_prev = get_node_pos(path_j, t-1)
                            edge_j = (pos_j_prev, pos_j, t-1)
                            omega[edge_j] = omega.get(edge_j, 0.0) + theta_c
                        else:
                            pos_i_next = get_node_pos(path_i, 1)
                            edge_i = (pos_i, pos_i_next, 0)
                            omega[edge_i] = omega.get(edge_i, 0.0) + theta_c
                            
                            pos_j_next = get_node_pos(path_j, 1)
                            edge_j = (pos_j, pos_j_next, 0)
                            omega[edge_j] = omega.get(edge_j, 0.0) + theta_c

                    # 2. Edge conflict at t-1 -> t
                    elif t > 0:
                        pos_i_prev = get_node_pos(path_i, t-1)
                        pos_j_prev = get_node_pos(path_j, t-1)
                        if pos_i == pos_j_prev and pos_j == pos_i_prev:
                            edge_i = (pos_i_prev, pos_i, t-1)
                            edge_j = (pos_j_prev, pos_j, t-1)
                            omega[edge_i] = omega.get(edge_i, 0.0) + theta_c
                            omega[edge_j] = omega.get(edge_j, 0.0) + theta_c

    return omega

def pricing_step(
    scenario: Scenario,
    agent: AgentSpec,
    alpha: float,
    beta: float,
    omega: Dict[Tuple[Coord, Coord, int], float],
    ell_max: float,
    tau_max: float,
    existing_paths: List[Path],
    max_time: int = 30
) -> Optional[Path]:
    """Run the pricing subproblem for one agent.
    
    Builds the direction-augmented graph with current omega values,
    runs Dijkstra to find the minimum reduced cost path,
    and returns it if it's not already in the existing path set.
    
    Args:
        scenario: Current scenario
        agent: The agent to price for
        alpha, beta: Objective weights
        omega: Current Lagrangian multipliers
        ell_max, tau_max: Current normalization constants
        existing_paths: Paths already in P_bar (to check for duplicates)
        max_time: Maximum timestep
    
    Returns:
        New Path if found and not duplicate, None otherwise.
    """
    G = build_augmented_graph(
        grid=scenario.grid,
        agent=agent,
        alpha=alpha,
        beta=beta,
        omega=omega,
        ell_max=ell_max,
        tau_max=tau_max,
        max_time=max_time,
        forbidden=None
    )
    
    new_path = dijkstra_augmented(G, agent, max_time)
    if new_path is None:
        return None
        
    # Check for duplicates: same sequence of (v, t)
    new_seq = [(n.v, n.t) for n in new_path.nodes]
    
    for p in existing_paths:
        if p.agent_id != agent.agent_id:
            continue
        p_seq = [(n.v, n.t) for n in p.nodes]
        if new_seq == p_seq:
            return None
            
    return new_path
