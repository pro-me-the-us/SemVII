"""Conflict-Based Search (CBS) for multi-agent path finding.

Standard two-level CBS with no turn awareness (alpha=1, beta=0 objective).
Used as the reference baseline. This is a from-scratch implementation.
"""
import itertools
import heapq
from dataclasses import dataclass
from typing import Optional, Dict, Set

from src.graph.grids import Scenario, Coord
from src.turncost.turncost import Path
from src.cbs.low_level import space_time_astar, Constraint, VertexConstraint, EdgeConstraint


@dataclass(frozen=True)
class Conflict:
    """A conflict between two agents."""
    agent_a: int
    agent_b: int
    conflict_type: str   # 'vertex' or 'edge'
    v: Coord             # conflict location (for vertex conflict)
    v2: Optional[Coord]  # second vertex (for edge conflict only)
    t: int               # timestep


class CBSNode:
    """A node in the CBS constraint tree."""
    def __init__(self):
        self.constraints: Set[Constraint] = set()
        self.paths: Dict[int, Path] = {}  # agent_id -> Path
        self.cost: float = 0.0  # sum of path lengths
    
    def __lt__(self, other):
        return self.cost < other.cost


def find_first_conflict(paths: Dict[int, Path]) -> Optional[Conflict]:
    """Find the first vertex or edge conflict among all agent paths.
    
    Vertex conflict: two agents at the same (v, t).
    Edge conflict: two agents traverse the same edge in opposite directions at the same t.
    
    Check all pairs of agents at all timesteps up to the max path length.
    Return the first conflict found, or None if paths are conflict-free.
    """
    if not paths:
        return None
        
    max_t = max(len(p.nodes) - 1 for p in paths.values())
    agent_ids = list(paths.keys())
    
    for t in range(max_t + 1):
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a1 = agent_ids[i]
                a2 = agent_ids[j]
                
                p1 = paths[a1]
                p2 = paths[a2]
                
                # Get positions at time t, padding if necessary
                v1 = p1.nodes[t].v if t < len(p1.nodes) else p1.nodes[-1].v
                v2 = p2.nodes[t].v if t < len(p2.nodes) else p2.nodes[-1].v
                
                # Vertex conflict
                if v1 == v2:
                    return Conflict(
                        agent_a=a1, agent_b=a2,
                        conflict_type='vertex',
                        v=v1, v2=None, t=t
                    )
                    
                # Edge conflict (only possible if t > 0)
                if t > 0:
                    prev_v1 = p1.nodes[t-1].v if t-1 < len(p1.nodes) else p1.nodes[-1].v
                    prev_v2 = p2.nodes[t-1].v if t-1 < len(p2.nodes) else p2.nodes[-1].v
                    
                    if v1 == prev_v2 and v2 == prev_v1 and v1 != prev_v1:
                        return Conflict(
                            agent_a=a1, agent_b=a2,
                            conflict_type='edge',
                            v=prev_v1, v2=v1, t=t
                        )
    return None


def cbs_solve(scenario: Scenario, max_time: int = 100) -> Optional[Dict[int, Path]]:
    """Solve MAPF instance using Conflict-Based Search.
    
    Args:
        scenario: The scenario with grid and agents
        max_time: Maximum timestep for low-level search
    
    Returns:
        Dict mapping agent_id to conflict-free Path, or None if unsolvable.
    
    Algorithm:
    1. Root node: solve each agent independently with no constraints
    2. Find first conflict in current paths
    3. If no conflict, return solution
    4. Branch: for each agent involved in conflict, create child node
       with added constraint forbidding that agent's conflicting action
    5. Re-plan the constrained agent's path
    6. Push valid child nodes to priority queue
    7. Repeat from step 2
    """
    root = CBSNode()
    
    for agent in scenario.agents:
        path = space_time_astar(scenario.grid, agent, set(), max_time)
        if path is None:
            return None  # No solution even without constraints
        root.paths[agent.agent_id] = path
        root.cost += path.length
        
    counter = itertools.count()
    open_set = []
    # Push root. We use counter to avoid comparing CBSNode constraints/paths directly
    heapq.heappush(open_set, (root.cost, next(counter), root))
    
    while open_set:
        cost, _, current = heapq.heappop(open_set)
        
        conflict = find_first_conflict(current.paths)
        if conflict is None:
            return current.paths
            
        # Create children based on the conflict
        if conflict.conflict_type == 'vertex':
            # Child 1: agent_a cannot be at v at time t
            c1 = VertexConstraint(conflict.agent_a, conflict.v, conflict.t)
            # Child 2: agent_b cannot be at v at time t
            c2 = VertexConstraint(conflict.agent_b, conflict.v, conflict.t)
            new_constraints = [c1, c2]
            agents_to_replan = [conflict.agent_a, conflict.agent_b]
        else:
            # Edge conflict
            c1 = EdgeConstraint(conflict.agent_a, conflict.v, conflict.v2, conflict.t)
            c2 = EdgeConstraint(conflict.agent_b, conflict.v2, conflict.v, conflict.t)
            new_constraints = [c1, c2]
            agents_to_replan = [conflict.agent_a, conflict.agent_b]
            
        for i in range(2):
            child = CBSNode()
            child.constraints = current.constraints | {new_constraints[i]}
            child.paths = current.paths.copy()
            
            agent_id = agents_to_replan[i]
            # Find the agent spec
            agent_spec = next(a for a in scenario.agents if a.agent_id == agent_id)
            
            # Replan
            new_path = space_time_astar(scenario.grid, agent_spec, child.constraints, max_time)
            
            if new_path is not None:
                child.paths[agent_id] = new_path
                child.cost = sum(p.length for p in child.paths.values())
                heapq.heappush(open_set, (child.cost, next(counter), child))
                
    return None
