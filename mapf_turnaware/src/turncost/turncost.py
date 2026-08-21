"""Turn cost calculations for MAPF paths.

Implements:
- Per-step sinusoidal turn penalty: sk = sin(angle(d, d'))
- Path-level aggregation: Sp = sum(sk), tau_p = exp(Sp) - 1
- Linear variant: tau_linear = Sp (kept for parity with paper)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Coord = Tuple[int, int]
Direction = Optional[Coord]  # unit vector (dx,dy) in {(-1,0),(1,0),(0,-1),(0,1)}, or None for start

@dataclass(frozen=True)
class STNode:
    """Spatio-temporal node (v, t)."""
    v: Coord
    t: int

@dataclass
class Path:
    """A path for one agent through spatio-temporal graph.
    
    Attributes:
        agent_id: ID of the agent this path belongs to
        nodes: List of (v, t) nodes along the path
        directions: Direction of each move (len = len(nodes)-1)
        length: Path length ℓp (sum of edge weights; waits cost 0)
        s_values: Per-step turn penalty sk
        Sp: Sum of s_values (linear turn sum)
        tau_exp: exp(Sp) - 1 (exponential turn cost)
        tau_linear: Sp (linear variant, for ablation parity)
    """
    agent_id: int
    nodes: List[STNode]
    directions: List[Direction] = field(default_factory=list)
    length: float = 0.0
    s_values: List[float] = field(default_factory=list)
    Sp: float = 0.0
    tau_exp: float = 0.0
    tau_linear: float = 0.0

def compute_sk(d_prev: Direction, d_curr: Direction) -> float:
    """Compute sinusoidal turn penalty between two consecutive directions.
    
    On a 4-connected grid:
      - Straight (0°): sk = 0
      - 90° turn (left/right): sk = 1  
      - U-turn (180°): sk = 0
      - If either direction is None (start/wait with no prior direction): sk = 0
      - Wait (same position, direction unchanged): sk = 0
    """
    if d_prev is None or d_curr is None:
        return 0.0
    # For a 4-connected grid unit vectors, dot product evaluates to 0 for 90 degree turns
    dot_product = d_prev[0] * d_curr[0] + d_prev[1] * d_curr[1]
    if dot_product == 0:
        return 1.0
    return 0.0

def extract_direction(from_node: STNode, to_node: STNode) -> Direction:
    """Extract the direction of movement from one node to the next.
    Returns None if the agent waits in place.
    Returns (dx, dy) unit vector otherwise.
    """
    dx = to_node.v[0] - from_node.v[0]
    dy = to_node.v[1] - from_node.v[1]
    if dx == 0 and dy == 0:
        return None
    return (dx, dy)

def compute_path_costs(path: Path) -> Path:
    """Compute all turn cost metrics for a path in-place and return it.
    
    Fills in: directions, length, s_values, Sp, tau_exp, tau_linear.
    - length = number of actual moves (not waits) along the path
    - directions = direction of each step (None for waits)
    - s_values[i] = compute_sk(directions[i-1], directions[i]) for i >= 1
                    (s_values[0] = 0 since there's no prior direction)
    - For computing turn costs, WAITS DO NOT COUNT as direction changes.
      A wait preserves the previous direction. So when computing sk,
      use the last non-None direction as the effective current direction.
    - Sp = sum(s_values)
    - tau_exp = exp(Sp) - 1
    - tau_linear = Sp
    """
    path.directions = []
    path.length = 0.0
    path.s_values = []
    path.Sp = 0.0
    path.tau_exp = 0.0
    path.tau_linear = 0.0
    
    if len(path.nodes) <= 1:
        return path
        
    for i in range(len(path.nodes) - 1):
        d = extract_direction(path.nodes[i], path.nodes[i+1])
        path.directions.append(d)
        path.length += 1.0  # Every timestep (including waits) adds 1 to length

            
    path.s_values = [0.0] * len(path.directions)
    last_non_none_direction = None
    
    for i in range(len(path.directions)):
        curr_d = path.directions[i]
        curr_effective = curr_d if curr_d is not None else last_non_none_direction
        
        if i == 0:
            path.s_values[i] = 0.0
        else:
            path.s_values[i] = compute_sk(last_non_none_direction, curr_effective)
            
        if curr_d is not None:
            last_non_none_direction = curr_d
            
    path.Sp = sum(path.s_values)
    path.tau_exp = math.exp(path.Sp) - 1.0
    path.tau_linear = path.Sp
    
    return path

def normalize_path_set(paths: List[Path]) -> Tuple[float, float]:
    """Compute normalization constants for a path set.
    Returns (ell_max, tau_max) where:
      ell_max = max(p.length for p in paths)
      tau_max = max(p.tau_exp for p in paths)
    Handles edge cases where max is 0 (set to 1.0 to avoid division by zero).
    """
    if not paths:
        return 1.0, 1.0
        
    ell_max = max((p.length for p in paths), default=0.0)
    tau_max = max((p.tau_exp for p in paths), default=0.0)
    
    if ell_max == 0.0:
        ell_max = 1.0
    if tau_max == 0.0:
        tau_max = 1.0
        
    return ell_max, tau_max

def normalized_costs(paths: List[Path], ell_max: float, tau_max: float) -> Tuple[List[float], List[float]]:
    """Return normalized (ell_hat, tau_hat) lists for the path set.
    ell_hat[i] = paths[i].length / ell_max
    tau_hat[i] = paths[i].tau_exp / tau_max
    """
    ell_hat = [p.length / ell_max for p in paths]
    tau_hat = [p.tau_exp / tau_max for p in paths]
    return ell_hat, tau_hat
