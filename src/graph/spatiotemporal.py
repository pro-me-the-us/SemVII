"""Direction-augmented spatio-temporal graph for turn-aware MAPF.

Implements the graph GT with nodes (v, t, d) where:
- v: spatial position (x, y)
- t: timestep
- d: direction of arrival at v

Edge weights follow eq. (21):
  w_tilde = (alpha/ell_max) * w(v,v') + omega(v,v') + (beta/tau_max) * sin(angle(d, d'))
"""
import heapq
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Set, List
import networkx as nx

from src.graph.grids import GridSpec, AgentSpec, Coord
from src.turncost.turncost import compute_sk, STNode, Path, compute_path_costs, Direction

@dataclass(frozen=True)
class AugNode:
    """Direction-augmented spatio-temporal node (v, t, d)."""
    v: Coord
    t: int
    d: Direction

def build_augmented_graph(
    grid: GridSpec,
    agent: AgentSpec,
    alpha: float,
    beta: float,
    omega: Dict[Tuple[Coord, Coord, int], float],
    ell_max: float,
    tau_max: float,
    max_time: int = 30,
    forbidden: Optional[Set[Tuple[Coord, int]]] = None
) -> nx.DiGraph:
    """Build the direction-augmented graph for one agent.
    
    Nodes: (v, t, d) for all passable v in grid, t in 0..max_time, 
           d in {(1,0),(-1,0),(0,1),(0,-1), None}
           None direction used only at t=0 for the start node.
    
    Edges: from (v, t, d) to (v', t+1, d') where:
      - v' is a neighbor of v (move) or v' = v (wait)
      - d' is the direction of movement from v to v', or d if waiting (preserve direction)
      - weight = (alpha/ell_max)*w + omega_e + (beta/tau_max)*sk
        where w=1 for moves, w=0 for waits
        sk = compute_sk(d, d')
        omega_e = omega.get((v, v', t), 0.0)
    
    Returns a networkx DiGraph with AugNode instances as nodes.
    """
    if forbidden is None:
        forbidden = set()
        
    G = nx.DiGraph()
    
    # Possible directions + wait direction (None)
    dirs = [(1,0), (-1,0), (0,1), (0,-1), None]
    
    # Generate nodes and edges up to max_time
    for t in range(max_time):
        for x in range(grid.width):
            for y in range(grid.height):
                v = (x, y)
                if not grid.is_passable(v) or (v, t) in forbidden:
                    continue
                
                # Iterate over possible current directions
                for d in dirs:
                    u_node = AugNode(v, t, d)
                    
                    # Wait move
                    if (v, t+1) not in forbidden:
                        v_node_wait = AugNode(v, t+1, d)
                        # Wait move: w=0 (no move cost), sk=0, omega based on wait
                        omega_e = omega.get((v, v, t), 0.0)
                        cost = omega_e  # w=0 per plan §5.2: "wait cost = 0"
                        G.add_edge(u_node, v_node_wait, weight=cost)

                        
                    # Real moves
                    for nbr in grid.neighbors(v):
                        if (nbr, t+1) not in forbidden:
                            dx = nbr[0] - v[0]
                            dy = nbr[1] - v[1]
                            d_prime = (dx, dy)
                            
                            v_node_move = AugNode(nbr, t+1, d_prime)
                            
                            sk = compute_sk(d, d_prime)
                            omega_e = omega.get((v, nbr, t), 0.0)
                            
                            cost = (alpha / ell_max) * 1.0 + omega_e + (beta / tau_max) * sk
                            G.add_edge(u_node, v_node_move, weight=cost)
                            
    return G

def dijkstra_augmented(
    graph: nx.DiGraph,
    agent: AgentSpec,
    max_time: int = 30
) -> Optional[Path]:
    """Run Dijkstra on the augmented graph to find minimum-cost path.
    
    Start: AugNode(agent.start, 0, None)
    Goal: any AugNode(agent.goal, t, d) for any t, d
    
    Returns a Path object with computed costs, or None if unreachable.
    Uses heapq-based Dijkstra (not networkx's built-in) for control over goal detection.
    """
    start_node = AugNode(agent.start, 0, None)
    if start_node not in graph:
        return None
        
    distances = {start_node: 0.0}
    parents = {start_node: None}
    pq = [(0.0, id(start_node), start_node)]
    
    best_goal_node = None
    min_goal_dist = float('inf')
    
    while pq:
        dist, _, curr = heapq.heappop(pq)
        
        if dist > distances.get(curr, float('inf')):
            continue
            
        if curr.v == agent.goal and curr.t == max_time and dist < min_goal_dist:
            min_goal_dist = dist
            best_goal_node = curr
            
        for nbr, edge_data in graph[curr].items():
            new_dist = dist + edge_data['weight']
            if new_dist < distances.get(nbr, float('inf')):
                distances[nbr] = new_dist
                parents[nbr] = curr
                heapq.heappush(pq, (new_dist, id(nbr), nbr))
                
    if best_goal_node is None:
        return None
        
    # Reconstruct path
    curr = best_goal_node
    aug_path = []
    while curr is not None:
        aug_path.append(curr)
        curr = parents[curr]
    aug_path.reverse()
    
    st_nodes = [STNode(n.v, n.t) for n in aug_path]
    path = Path(agent_id=agent.agent_id, nodes=st_nodes)
    return compute_path_costs(path)
