"""QUBO matrix construction for MAPF turn-aware optimization.

Implements equations (16)-(18) from the paper:
- Diagonal: c_p^{alpha,beta} - theta_oh 
- Off-diagonal (same agent): 2 * theta_oh
- Off-diagonal (conflicting): theta_c
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set

from src.turncost.turncost import Path

@dataclass
class RestrictedPathSet:
    """The restricted path set P_bar used in column generation."""
    paths: List[Path]        # All paths in the pool
    agent_of: List[int]      # a(p) for each path index p
    ell_hat: np.ndarray      # normalized lengths
    tau_hat: np.ndarray      # normalized turn costs
    conflict_matrix: np.ndarray  # C, n x n {0,1}

@dataclass
class QUBOInstance:
    """A complete QUBO instance ready for solving."""
    Q: np.ndarray            # n x n symmetric matrix
    path_set: RestrictedPathSet
    theta_oh: float
    theta_c: float

def detect_path_conflicts(path_a: Path, path_b: Path) -> bool:
    """Check if two paths have vertex or edge conflicts.
    
    Vertex conflict: same (v, t) in both paths.
    Edge conflict: agents traverse same edge in opposite directions at same t.
    Pad shorter path by assuming agent stays at final position.
    """
    if path_a.agent_id == path_b.agent_id:
        return False
        
    len_a = len(path_a.nodes)
    len_b = len(path_b.nodes)
    max_t = max(path_a.nodes[-1].t if len_a > 0 else 0, path_b.nodes[-1].t if len_b > 0 else 0)
    
    def get_pos(path: Path, t: int) -> Tuple[int, int]:
        if not path.nodes:
            return (-1, -1)
        for node in path.nodes:
            if node.t == t:
                return node.v
        return path.nodes[-1].v # pad with final position
        
    for t in range(max_t + 1):
        pos_a = get_pos(path_a, t)
        pos_b = get_pos(path_b, t)
        if pos_a == pos_b:
            return True
            
        if t > 0:
            pos_a_prev = get_pos(path_a, t-1)
            pos_b_prev = get_pos(path_b, t-1)
            if pos_a == pos_b_prev and pos_b == pos_a_prev:
                return True
                
    return False

def build_conflict_matrix(paths: List[Path]) -> np.ndarray:
    """Build the n x n binary conflict matrix C.
    C[i,j] = 1 if paths[i] and paths[j] conflict, 0 otherwise.
    Diagonal is 0.
    """
    n = len(paths)
    C = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            if detect_path_conflicts(paths[i], paths[j]):
                C[i, j] = 1.0
                C[j, i] = 1.0
    return C

def build_same_agent_matrix(agent_of: List[int]) -> np.ndarray:
    """Build B matrix where B[p,q] = 1 iff a(p) = a(q) and p != q."""
    n = len(agent_of)
    B = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            if agent_of[i] == agent_of[j]:
                B[i, j] = 1.0
                B[j, i] = 1.0
    return B

def build_qubo(
    path_set: RestrictedPathSet,
    alpha: float,
    beta: float,  
    theta_oh: float = 1.5,
    theta_c: float = 2.5
) -> QUBOInstance:
    """Build the QUBO matrix Q per equations (16)-(18).
    
    Q[p,p] = c_p - theta_oh  (diagonal)
    Q[p,q] += 2*theta_oh * B[p,q]  (same-agent penalty)
    Q[p,q] += theta_c * C[p,q]     (conflict penalty)
    
    where c_p = alpha * ell_hat[p] + beta * tau_hat[p]
    
    Returns a QUBOInstance with the symmetric Q matrix.
    """
    n = len(path_set.paths)
    Q = np.zeros((n, n), dtype=np.float32)
    
    B = build_same_agent_matrix(path_set.agent_of)
    C = path_set.conflict_matrix
    
    c_p = alpha * path_set.ell_hat + beta * path_set.tau_hat
    
    for p in range(n):
        Q[p, p] = c_p[p] - theta_oh
        for q in range(p + 1, n):
            val = 0.0
            if B[p, q] == 1.0:
                val += 2.0 * theta_oh
            if C[p, q] == 1.0:
                val += theta_c
            Q[p, q] = val
            Q[q, p] = val
            
    return QUBOInstance(Q=Q, path_set=path_set, theta_oh=theta_oh, theta_c=theta_c)
