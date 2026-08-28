"""Single-agent space-time A* for CBS low-level search.

Finds the shortest path for one agent on the spatio-temporal graph GT,
respecting a set of constraints (vertex and edge) imposed by CBS.
Uses Manhattan distance heuristic, unit move cost, 0-cost waiting.
"""
import heapq
import itertools
from dataclasses import dataclass
from typing import Union, Set, Optional, Tuple, Dict, List

from src.graph.grids import GridSpec, AgentSpec, Coord
from src.turncost.turncost import Path, STNode, compute_path_costs


@dataclass(frozen=True)
class VertexConstraint:
    agent_id: int
    v: Coord       # forbidden position
    t: int         # forbidden timestep


@dataclass(frozen=True) 
class EdgeConstraint:
    agent_id: int
    v1: Coord      # from position
    v2: Coord      # to position  
    t: int         # timestep of the move


Constraint = Union[VertexConstraint, EdgeConstraint]


def space_time_astar(
    grid: GridSpec,
    agent: AgentSpec, 
    constraints: Set[Constraint],
    max_time: int = 100
) -> Optional[Path]:
    """Find shortest path for a single agent respecting constraints.
    
    Args:
        grid: The grid specification
        agent: Agent with start and goal
        constraints: Set of vertex/edge constraints for this agent
        max_time: Maximum timestep to search up to (prevents infinite expansion)
    
    Returns:
        Path object with computed costs, or None if no path exists.
    """
    start_pos = agent.start
    goal_pos = agent.goal
    
    # Filter constraints for this agent to speed up checks
    my_constraints = {c for c in constraints if c.agent_id == agent.agent_id}
    
    def heuristic(pos: Coord) -> float:
        return abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
        
    # Priority queue: (f_score, timestep, g_score, counter, current_pos, path_nodes)
    open_set = []
    counter = itertools.count()
    start_h = heuristic(start_pos)
    initial_path = [STNode(start_pos, 0)]
    heapq.heappush(open_set, (start_h, 0, 0, next(counter), start_pos, initial_path))
    
    # Visited state: (pos, t) -> g_score
    visited: Dict[Tuple[Coord, int], int] = {(start_pos, 0): 0}
    
    while open_set:
        f, t, g, _, curr, path_nodes = heapq.heappop(open_set)
        
        # Check if we reached goal and can stop here
        if curr == goal_pos:
            future_conflict = False
            for c in my_constraints:
                if isinstance(c, VertexConstraint) and c.v == goal_pos and c.t > t:
                    future_conflict = True
                    break
            
            if not future_conflict:
                path = Path(agent_id=agent.agent_id, nodes=path_nodes)
                return compute_path_costs(path)
                
        if t >= max_time:
            continue
            
        actions = grid.neighbors(curr) + [curr]
        
        for nxt in actions:
            nxt_t = t + 1
            
            # Check vertex constraint
            v_conflict = False
            for c in my_constraints:
                if isinstance(c, VertexConstraint) and c.v == nxt and c.t == nxt_t:
                    v_conflict = True
                    break
            if v_conflict:
                continue
                
            # Check edge constraint
            e_conflict = False
            for c in my_constraints:
                if isinstance(c, EdgeConstraint) and c.v1 == curr and c.v2 == nxt and c.t == nxt_t:
                    e_conflict = True
                    break
            if e_conflict:
                continue
                
            move_cost = 0 if nxt == curr else 1  # Wait costs 0, move costs 1
            new_g = g + move_cost
            
            state = (nxt, nxt_t)
            if state in visited and visited[state] <= new_g:
                continue
                
            visited[state] = new_g
            
            h = heuristic(nxt)
            f_score = new_g + h
            
            new_path = list(path_nodes)
            new_path.append(STNode(nxt, nxt_t))
            
            heapq.heappush(open_set, (f_score, nxt_t, new_g, next(counter), nxt, new_path))
            
    return None
