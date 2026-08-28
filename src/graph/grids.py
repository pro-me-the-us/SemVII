from dataclasses import dataclass
from typing import Tuple, FrozenSet, List
import random

Coord = Tuple[int, int]  # (x, y)

@dataclass(frozen=True)
class GridSpec:
    name: str            # "empty_6x6" | "room_6x6"
    width: int
    height: int
    blocked: FrozenSet[Coord]  # wall cells
    
    def is_passable(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height and coord not in self.blocked
    
    def neighbors(self, coord: Coord) -> List[Coord]:
        """Return passable 4-connected neighbors of coord."""
        x, y = coord
        candidates = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        return [c for c in candidates if self.is_passable(c)]

@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    start: Coord
    goal: Coord

@dataclass(frozen=True)
class Scenario:
    scenario_id: str       # e.g. "Empty 6x6-2a"
    grid: GridSpec
    agents: Tuple[AgentSpec, ...]

EMPTY_6X6 = GridSpec(name="empty_6x6", width=6, height=6, blocked=frozenset())

# Room 6x6 with off-center doorway at (3,1)
ROOM_6X6 = GridSpec(
    name="room_6x6", width=6, height=6,
    blocked=frozenset({(3,0), (3,2), (3,3), (3,4), (3,5)})
)

SCENARIO_EMPTY_2A = Scenario(
    scenario_id="Empty 6x6-2a",
    grid=EMPTY_6X6,
    agents=(
        AgentSpec(agent_id=0, start=(0,2), goal=(5,3)),
        AgentSpec(agent_id=1, start=(5,2), goal=(0,3)),
    )
)

SCENARIO_EMPTY_4A = Scenario(
    scenario_id="Empty 6x6-4a",
    grid=EMPTY_6X6,
    agents=(
        AgentSpec(agent_id=0, start=(0,2), goal=(5,3)),
        AgentSpec(agent_id=1, start=(5,2), goal=(0,3)),
        AgentSpec(agent_id=2, start=(2,5), goal=(3,0)),
        AgentSpec(agent_id=3, start=(2,0), goal=(3,5)),
    )
)

SCENARIO_ROOM_2A = Scenario(
    scenario_id="Room 6x6-2a",
    grid=ROOM_6X6,
    agents=(
        AgentSpec(agent_id=0, start=(0,0), goal=(5,5)),
        AgentSpec(agent_id=1, start=(5,5), goal=(0,0)),
    )
)

ALL_SCENARIOS = {
    "Empty 6x6-2a": SCENARIO_EMPTY_2A,
    "Empty 6x6-4a": SCENARIO_EMPTY_4A,
    "Room 6x6-2a": SCENARIO_ROOM_2A,
}

def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_id}. Available: {list(ALL_SCENARIOS.keys())}")
    return ALL_SCENARIOS[scenario_id]
