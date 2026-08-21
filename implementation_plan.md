# Implementation Plan: Turn-Aware QUBO-and-Price MAPF
### (Exponential Turn Aggregation, Path-Based Branch-and-Price, QUBO via Local Simulated Annealing)

**Status:** Design document only — no code yet. This plan is meant to be handed to a
follow-up implementation session as a build spec.

---

## 1. Goals and Scope

Reproduce, end-to-end, the experiments in Table I of the paper for three scenarios:

| Scenario         | Grid  | K (agents) | (α, β) configs                                  |
|-------------------|-------|:----------:|--------------------------------------------------|
| Empty 6x6-2a      | Empty 6×6 | 2 | (0.50, 0.50) |
| Empty 6x6-4a      | Empty 6×6 | 4 | (1.00, 0.00), (0.75, 0.25), (0.50, 0.50) |
| Room 6x6-2a       | Room 6×6  | 2 | (1.00, 0.00), (0.50, 0.50) |

For every (scenario, α, β) cell we run **two solvers**, each **10 times** with different
random seeds (matching the paper's protocol), and report mean ± std of:
`ℓ` (path length), `Sp` (raw sinusoidal turn sum), `τ` (exponential turn cost), runtime.

- **Baseline solver:** Conflict-Based Search (CBS), distance-only (α=1, β=0 objective,
  turns not optimized — implemented from scratch, no library).
- **Main solver:** Turn-Aware QUBO-and-Price (Algorithm 1 in the paper), with the QUBO
  subproblem solved by a **local simulated-annealing QUBO solver** standing in for
  quantum annealing (see §7 for why, and how this swaps to real QPU hardware later).

CBS is a fixed baseline (it does not sweep α/β — it's distance-only by definition), but
we still run it once per scenario (not per α/β) for comparison purposes, 10 seeds each.
The QUBO-and-Price solver is run once per (scenario, α, β) cell, 10 seeds each.

---

## 2. Repository / Folder Structure

```
mapf_turnaware/
├── src/
│   ├── graph/
│   │   ├── grids.py              # grid + scenario definitions (§4)
│   │   └── spatiotemporal.py     # GT expansion, direction-augmented graph (§5.3)
│   ├── turncost/
│   │   └── turncost.py           # sk, Sp, τp, linear variant, normalization (§5.2)
│   ├── cbs/
│   │   ├── cbs.py                # from-scratch Conflict-Based Search (§6)
│   │   └── low_level.py          # single-agent A* on GT (space-time A*)
│   ├── qubo/
│   │   ├── qubo_builder.py       # builds Q per eq. (16)-(18) (§7.2)
│   │   ├── qubo_solver.py        # local SA solver wrapper (stand-in for QA) (§7.3)
│   │   └── pricing.py            # direction-augmented shortest path pricing (§7.4)
│   ├── branch_and_price/
│   │   └── solver.py             # Algorithm 1 driver, outer/inner loop (§7.5)
│   ├── experiment/
│   │   ├── config.py             # scenario/α/β/seed matrix (§8.1)
│   │   ├── runner.py             # executes both solvers, collects rows (§8.2)
│   │   └── metrics.py            # ℓ, Sp, τ, runtime aggregation, mean±std (§8.3)
│   └── report/
│       └── plots.py              # per-scenario plotting (§9)
├── results/
│   ├── empty_6x6_2a/
│   │   ├── cbs_results.csv
│   │   ├── qa_results.csv
│   │   ├── raw/                  # per-seed, per-run raw path + solver logs (JSON)
│   │   └── plots/
│   │       ├── length_vs_turncost.png
│   │       ├── sp_by_beta.png
│   │       └── runtime.png
│   ├── empty_6x6_4a/
│   │   ├── cbs_results.csv
│   │   ├── qa_results.csv
│   │   ├── raw/
│   │   └── plots/
│   └── room_6x6_2a/
│       ├── cbs_results.csv
│       ├── qa_results.csv
│       ├── raw/
│       └── plots/
├── tests/
│   ├── test_turncost.py
│   ├── test_cbs.py
│   ├── test_qubo_builder.py
│   ├── test_pricing.py
│   └── test_branch_and_price_small.py
├── requirements.txt
└── main.py                        # CLI entry: run all / run one scenario
```

**Rule for §1(6):** every scenario folder is self-contained and holds both solvers'
results, distinguished purely by filename (`cbs_results.csv` vs `qa_results.csv`), plus
a `plots/` subfolder and a `raw/` subfolder for full reproducibility (exact paths
selected, per-seed logs, QUBO matrices if small enough to serialize).

---

## 3. Core Data Structures

```python
# --- graph/grids.py ---
Coord = Tuple[int, int]                 # (x, y)

@dataclass(frozen=True)
class GridSpec:
    name: str                           # "empty_6x6" | "room_6x6"
    width: int
    height: int
    blocked: FrozenSet[Coord]           # wall cells, empty set for the open grid

@dataclass(frozen=True)
class Scenario:
    scenario_id: str                    # "Empty 6x6-2a"
    grid: GridSpec
    agents: Tuple["AgentSpec", ...]

@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    start: Coord
    goal: Coord

# --- graph/spatiotemporal.py ---
Direction = Optional[Coord]             # unit vector (dx,dy) in {(-1,0),(1,0),(0,-1),(0,1)}, or None for "wait"/"start"

@dataclass(frozen=True)
class STNode:                           # spatio-temporal node, (v, t)
    v: Coord
    t: int

@dataclass(frozen=True)
class AugNode:                          # direction-augmented node, (v, t, d)
    v: Coord
    t: int
    d: Direction

# --- turncost/turncost.py ---
@dataclass
class Path:
    agent_id: int
    nodes: List[STNode]                 # (v,0) ... (v_goal, t')
    directions: List[Direction]         # len = len(nodes)-1, direction of each move
    length: float                       # ℓp  (sum of edge weights, incl. wait cost = 0 or small)
    s_values: List[float]               # sk per turn step
    Sp: float                           # sum sk  (linear turn sum)
    tau_exp: float                      # e^Sp - 1
    tau_linear: float                   # Sp (linear-variant turn cost, used only for the "linear variant" ablation, not required for the 3 requested scenarios but kept for parity with the paper)

# --- qubo/qubo_builder.py ---
@dataclass
class RestrictedPathSet:
    paths: List[Path]                   # P̄
    agent_of: List[int]                 # a(p) for each index p
    ell_hat: np.ndarray                 # normalized lengths
    tau_hat: np.ndarray                 # normalized turn costs
    conflict_matrix: np.ndarray         # C, n x n {0,1}

@dataclass
class QUBOInstance:
    Q: np.ndarray                       # n x n symmetric matrix (eq. 18)
    path_set: RestrictedPathSet
    theta_oh: float
    theta_c: float

# --- experiment/config.py ---
@dataclass(frozen=True)
class RunConfig:
    scenario_id: str
    alpha: float
    beta: float
    seed: int
    solver: Literal["cbs", "qa"]
```

Design choices worth flagging:
- `Path.tau_linear` field is kept only so the codebase can later reproduce the paper's
  "linear variant (ours)" ablation row without restructuring — it is **not** computed
  or reported for the three requested scenarios unless explicitly asked.
- `Direction` is a normalized `(dx, dy) ∈ {(1,0),(-1,0),(0,1),(0,-1)}` on a 4-connected
  grid; `θ(d, d′)` is computed from the angle between two such vectors, giving
  `θ ∈ {0°, 90°, 180°}` exactly as the paper specifies for 4-connected grids (Sec. III-A).
- Waiting in place is represented as `d′ = d` (direction unchanged) → `θ = 0°` → no
  penalty, matching "no turn penalty for waiting."

---

## 4. Scenario Definitions (fixed, documented — §2 & §3 of your requirements)

### 4.1 Empty 6×6 grid
`GridSpec(width=6, height=6, blocked=frozenset())`, coordinates `(x, y)` with
`x, y ∈ {0..5}`, origin at bottom-left.

**Empty 6x6-2a** (K=2) — start/goal chosen as opposite corners so both agents must
cross paths, forcing at least one conflict to resolve:

| Agent | Start | Goal |
|---|---|---|
| 0 | (0, 0) | (5, 5) |
| 1 | (5, 0) | (0, 5) |

Manhattan distance = 10 each. Table I reports `ℓ = 20.0` for β=0.5 (i.e. exactly the sum
of two shortest paths, 10+10) — consistent with this start/goal choice.

**Empty 6x6-4a** (K=4) — four corners, each agent to the opposite corner (rotationally
symmetric, all crossing at center):

| Agent | Start | Goal |
|---|---|---|
| 0 | (0, 0) | (5, 5) |
| 1 | (5, 0) | (0, 5) |
| 2 | (5, 5) | (0, 0) |
| 3 | (0, 5) | (5, 0) |

Sum of 4 shortest-path Manhattan distances = 4 × 10 = 40; but reported β=0 length is
28 (K=4 row), which reflects that CBS/branch-and-price only need to avoid collisions,
not that all 4 pairs are simultaneously shortest — **note**: this discrepancy versus a
naive expectation is a known reproducibility gap since the paper does not publish its
exact start/goal set. We will document our own fixed set above and **not** attempt to
reverse-engineer exact numeric parity with Table I; our job is to reproduce the
*qualitative* trends (β↑ ⇒ Sp↓, ℓ↑; exponential ≫ linear amplification), not the exact
digits, since those depend on unpublished instance data.

### 4.2 Room 6×6 grid
A 6×6 grid split by a vertical interior wall at `x = 3` spanning `y ∈ {0,1,2,4,5}` with a
single doorway gap at `(3, 3)`, creating two "rooms" connected by one corridor cell —
a minimal, reasonable "room-like" structure as requested:

```
y=5  . . . # . .
y=4  . . . # . .
y=3  . . . . . .      <- doorway at (3,3)
y=2  . . . # . .
y=1  . . . # . .
y=0  . . . # . .
     x=0 1 2 3 4 5
```
`blocked = {(3,0),(3,1),(3,2),(3,4),(3,5)}`, `(3,3)` open (doorway).

**Room 6x6-2a** (K=2) — one agent in each room, goals swapped, forcing both through the
doorway (this is exactly the "forced through doorways" behavior the paper's §V-B3
discusses):

| Agent | Start | Goal |
|---|---|---|
| 0 | (0, 0) | (5, 5) |
| 1 | (5, 5) | (0, 0) |

Both grids and all start/goal sets are defined once in `grids.py` as the single source
of truth used by both solvers and all α/β sweeps, so CBS and QA are compared on
**identical instances**.

---

## 5. Turn Cost & Direction-Augmented Graph (maps to paper §III)

### 5.1 Per-step angle → penalty
```
angle(d, d') -> θ ∈ {0°, 90°, 180°}          # 4-connected grid
sk = sin(θ)                                  # -> {0, 1, 0}
```
Implemented as a lookup table keyed by `(d, d')` pairs (8 cases: straight, 4 left/right
90° turns, 1 U-turn, plus wait-related identities) rather than trig calls, to avoid
floating point angle computation on a grid where only 3 discrete angles are possible.

### 5.2 Aggregation
```
Sp   = sum(sk for each consecutive move pair along the path)
τp   = exp(Sp) - 1
ℓp   = sum of edge weights (unit weight per move, 0 or small wait cost, per grid)
```
Normalization `ℓ̂p = ℓp/ℓmax`, `τ̂p = τp/τmax` is computed **per restricted path set P̄**
(i.e., recomputed every time P̄ grows — Algorithm 1 line 19), using the max over the
*current* pool, not a global constant.

### 5.3 Direction-augmented graph (paper §IV-E1)
Node = `(v, t, d)` where `d` is the direction of the edge that was just traversed to
arrive at `v` (or a special `START` direction with no penalty for the first move).
Edge weight from `(v,t,d)` to `(v′,t+1,d′)`:
```
w̃ = (α/ℓmax) * w(v,v') + ω(v,v') + (β/τmax) * sin(angle(d, d'))
```
This is exactly eq. (21). Implemented via `networkx.DiGraph` (or a custom adjacency
dict, TBD in implementation phase based on grid size — 6×6×T×4-directions is small
enough that Dijkstra via `heapq` by hand is also fine and avoids a networkx dependency).

`ωe` are the Lagrangian multipliers from the conflict constraints, updated once per
outer iteration of Algorithm 1 (§7.5, step 9/22).

---

## 6. CBS Baseline (from scratch, distance-only)

Standard two-level Conflict-Based Search, no turn awareness (α=1, β=0 objective, i.e.
plain Manhattan/step-cost minimization), used purely as the reference baseline required
by point (2) of your instructions.

**High level:**
```
class CBSNode:
    constraints: Set[Constraint]     # (agent, vertex/edge, time) forbidden
    paths: Dict[agent_id, Path]
    cost: float                      # sum of path lengths

CBS(scenario):
    root = CBSNode(constraints=∅, paths={a: low_level_astar(a, ∅) for a in agents})
    push root to priority queue (ordered by cost)
    while queue not empty:
        node = pop lowest-cost node
        conflict = find_first_conflict(node.paths)   # vertex or edge conflict
        if conflict is None:
            return node.paths                         # solution found
        for agent in (conflict.agent_a, conflict.agent_b):
            child = copy(node)
            child.constraints.add(new constraint forbidding this agent
                                    from the conflicting vertex/edge/time)
            child.paths[agent] = low_level_astar(agent, child.constraints)
            if child.paths[agent] is not None:
                push child to queue
```
`low_level_astar` = standard space-time A* over `(v, t)` nodes, respecting the agent's
constraint set, using Manhattan-distance heuristic, unit move cost, 0-cost or ε-cost
waiting. This lives in `cbs/low_level.py` and is reused nowhere else (QUBO pricing uses
the direction-augmented Dijkstra instead, since it needs turn costs baked in).

CBS itself has no randomness, so "10 seeds" for CBS means: rerun the identical
deterministic algorithm 10 times purely to (a) match the paper's reporting protocol
uniformly across both solvers and (b) capture wall-clock runtime variance — `ℓ`, `Sp`,
`τ` will have **std = 0** for CBS by construction, which is expected and will be noted
in the report, not treated as a bug.

---

## 7. Turn-Aware QUBO-and-Price (Algorithm 1)

### 7.1 Why a local solver stands in for quantum annealing
This sandbox has no network egress, so a live call to a cloud QPU (D-Wave Leap, etc.)
is not possible here. Per your choice in point (1), `SOLVEQUBO(Q)` will be implemented
with a **classical simulated-annealing QUBO/Ising sampler**, run locally — this mirrors
exactly what the paper itself did for its own experiments ("Our experiments use
simulated annealing as a classical backend to validate the formulation independently of
quantum hardware availability," §IV-F). The module is isolated (`qubo_solver.py`)
behind a single function so a real QPU sampler can be substituted later with no changes
to any other module:
```python
def solve_qubo(Q: np.ndarray, num_reads: int, seed: int) -> np.ndarray:
    ...
```
Candidate libraries (decide at implementation time based on what's installable):
- `dimod.SimulatedAnnealingSampler` / `neal.SimulatedAnnealingSampler` (D-Wave Ocean's
  own classical SA reference sampler — same interface as `DWaveSampler`, so swapping to
  real hardware later is a one-line change: `EmbeddingComposite(DWaveSampler())`).
- Fallback if `dwave-neal`/`dimod` cannot be installed offline: a small hand-rolled
  Metropolis/simulated-annealing sampler over binary vectors, since `n = |P̄|` stays
  small (tens of paths) for these 6×6, ≤4-agent instances — trivial to implement without
  external deps.

### 7.2 QUBO construction (`qubo_builder.py`)
Implements eq. (16)–(18) directly:
```python
def build_qubo(path_set: RestrictedPathSet, alpha, beta,
                theta_oh, theta_c) -> QUBOInstance:
    c = alpha * path_set.ell_hat + beta * path_set.tau_hat        # eq. 9, diagonal costs
    n = len(path_set.paths)
    B = same_agent_matrix(path_set.agent_of)                       # B[p,q]=1 iff a(p)=a(q), p≠q
    Q = np.diag(c - theta_oh)
    Q += 2 * theta_oh * B
    Q += theta_c * path_set.conflict_matrix
    return QUBOInstance(Q=Q, path_set=path_set, theta_oh=theta_oh, theta_c=theta_c)
```
Penalty strengths per eq. (13)/(15): `theta_oh = 1.5` (>1 after normalization is
"always sufficient"), `theta_c = 2.5` (>2). These constants live in `config.py` and are
exposed as run parameters for the small ablation tests in `tests/`.

Conflict matrix `C` is built from pairwise path comparisons: vertex conflict = same
`(v,t)` in both paths; edge conflict = the two agents traverse the same undirected edge
in opposite directions at the same timestep (matches §II-B exactly).

### 7.3 Local SOLVEQUBO
`solve_qubo` returns the best sample (lowest `z^T Q z`) found across `num_reads`
independent anneals for a given `seed`; `num_reads` fixed per run (e.g. 200), documented
in `config.py`.

### 7.4 Pricing subproblem (`pricing.py`)
Implements §IV-E: for each agent, Dijkstra over the direction-augmented graph with
weights from eq. (21), using the current `ω` (Lagrange multipliers extracted from the
QUBO solve — see below). Returns the path with minimum reduced cost `c̄p` not already in
`P̄`. `Proposition 4`'s monotonicity argument is exactly why plain Dijkstra suffices
(no need to search over `τp` directly, only over the linear `Sp` proxy inside the edge
weights).

**Extracting ω from a QUBO solve (Algorithm 1, line 9/22 — "Extract ω via (23)"):**
The paper references an eq. (23) for this that isn't included in the excerpt provided.
We will implement this as a **dual-value approximation** standard to QUBO-based
column generation: treat `θ_c` weighted row-sums of the current binary solution `z` as a
proxy for constraint activity per edge, i.e.
```
ω_e ← θ_c * Σ_{p: e∈p} Σ_{q: e∈q, conflict(p,q)} z_p * z_q   (edges appearing in active conflicts get a positive multiplier)
```
This will be flagged clearly in code comments as **"reconstructed from context, not
verified against the paper's own eq. (23)"** and validated empirically in
`test_branch_and_price_small.py` by checking that (a) `ω_e ≥ 0` always, (b) increasing
`ω_e` on a conflicted edge changes pricing to avoid it, and (c) the outer loop still
terminates and returns conflict-free solutions on toy instances — i.e. correctness is
verified behaviorally, not by matching an unseen formula.

### 7.5 Branch-and-Price driver (`branch_and_price/solver.py`)
Direct translation of Algorithm 1's pseudocode:
```python
def turn_aware_qubo_and_price(scenario, alpha, beta, seed, sa_reads) -> Solution:
    P_bar = init_one_shortest_path_per_agent(scenario)      # line 1
    recompute_turn_costs(P_bar)                              # line 2
    normalize(P_bar)                                          # line 3
    omega = {e: 0.0 for e in scenario.edges}                  # line 4
    active_conflicts = set()                                  # line 5

    while True:                                                # line 6 outer repeat
        Q = build_qubo(P_bar, alpha, beta, theta_oh, theta_c, active_conflicts)  # line 7
        z = solve_qubo(Q.Q, num_reads=sa_reads, seed=seed)     # line 8
        omega = extract_omega(z, P_bar, active_conflicts)      # line 9

        while True:                                             # line 10 inner repeat
            new_path_added = False
            for agent in scenario.agents:                       # line 11
                aug_graph = build_direction_augmented_graph(     # line 12
                    scenario, alpha, beta, omega, ell_max, tau_max)
                p_star = dijkstra_min_reduced_cost(aug_graph, agent)  # line 13
                if p_star not in P_bar:                          # line 14
                    P_bar.add(p_star)                            # line 15
                    compute_turn_cost(p_star)                    # line 16
                    new_path_added = True
            renormalize(P_bar)                                    # line 19
            Q = build_qubo(P_bar, alpha, beta, theta_oh, theta_c, active_conflicts)  # line 20
            z = solve_qubo(Q.Q, num_reads=sa_reads, seed=seed)    # line 21
            omega = extract_omega(z, P_bar, active_conflicts)     # line 22
            if stopping_criterion_met(z, omega) or not new_path_added:  # line 23
                break

        conflicts = detect_conflicts(selected_paths(z, P_bar))    # line 24
        if not conflicts:                                          # line 25/28
            return Solution(selected_paths(z, P_bar))
        active_conflicts |= conflicts                              # line 26
```
`stopping_criterion_met` implements eq. (22) directly (`min reduced cost of unselected ≥
v̂ − L(ω)`). A hard iteration cap (e.g. 50 outer iterations) is added defensively so a
bug in the ω-extraction heuristic (§7.4) cannot hang the experiment runner — logged as a
warning + solution flagged `converged=False` if hit, so the experiment report can
distinguish "true optimum found" from "iteration cap reached."

---

## 8. Experiment Harness

### 8.1 Config matrix (`experiment/config.py`)
Single declarative list generated from §1's table, expanded into individual `RunConfig`
objects: 3 scenarios × (1 or 3 or 2 α/β configs) × 10 seeds × 2 solvers (with CBS's
α/β fixed at (1,0) regardless of which β-configs QA runs, per point 4). Seeds fixed as
`0..9` for reproducibility across both solvers.

### 8.2 Runner (`experiment/runner.py`)
For each `RunConfig`:
1. Build scenario from `grids.py`.
2. Dispatch to `cbs.solve()` or `branch_and_price.solve()`.
3. Record: `ℓ` (sum over agents), `Sp` (sum over agents), `τ` (sum over agents, using
   the *unnormalized* exponential formula on the final selected paths — matching how
   Table I reports these as whole-solution totals), wall-clock runtime, `converged` flag.
4. Serialize the full per-agent path (as a list of `(v,t)`) and solver metadata to
   `results/<scenario>/raw/<solver>_<alpha>_<beta>_<seed>.json` for later inspection/
   debugging, independent of the aggregated CSVs.

### 8.3 Aggregation (`experiment/metrics.py`)
Groups the 10 seeds per (scenario, solver, α, β) and writes one row per config to the
appropriate CSV with `mean ± std` columns, matching Table I's schema exactly:
```
scenario,K,alpha,beta,ell_mean,ell_std,Sp_mean,Sp_std,tau_mean,tau_std,time_mean,time_std,converged_rate
```
`cbs_results.csv` will have `alpha=1.0, beta=0.0` fixed on every row (labeling which
scenario, K, and the 10-seed runtime stats); `qa_results.csv` has one row per α/β
config actually run for that scenario.

---

## 9. Plots (per scenario, in `results/<scenario>/plots/`)

1. **`length_vs_turncost.png`** — scatter/line of `ℓ_mean` (x) vs `τ_mean` (y, log
   scale) across the β sweep for QA, with the single CBS point overlaid for reference —
   directly visualizes the trade-off curve the paper's §V-B1/B2 describes.
2. **`sp_by_beta.png`** — bar chart of `Sp_mean` (with error bars = std) per β,
   reproducing the "increasing β ⇒ decreasing Sp" trend (§V-B1).
3. **`runtime.png`** — bar chart of `time_mean` (± std) per solver/config, reproducing
   §V-B4's runtime discussion.

For `Empty 6x6-4a` and `Room 6x6-2a` (multi-config scenarios), all three plots overlay
every α/β config on one figure; for `Empty 6x6-2a` (single config), the plots reduce to
CBS-vs-QA single-point comparisons.

---

## 10. Testing Plan

- `test_turncost.py`: verify `sk` table (0°→0, 90°→1, 180°→0), verify `τp=e^Sp−1`
  against hand-computed values from Table I's own rows (e.g. `Sp=6 → τ≈402.4?`,
  `Sp=4 → τ≈53.6`) as regression fixtures.
- `test_cbs.py`: verify CBS returns conflict-free paths on a 2-agent head-on scenario
  and a swap-conflict scenario; verify optimality on trivial no-conflict cases.
- `test_qubo_builder.py`: verify `Q` is symmetric, diagonal reproduces `c^{α,β}_p −
  θ_oh`, off-diagonal reproduces eq. (17) on a 2-path, 1-agent toy example by hand.
- `test_pricing.py`: verify Dijkstra on the direction-augmented graph picks a longer,
  straighter path over a shorter, zig-zaggy one once β is large enough — a direct,
  minimal check of the core trade-off mechanism.
- `test_branch_and_price_small.py`: full Algorithm 1 run on a 3×3 grid, 2-agent toy
  instance small enough to brute-force-verify the optimum by exhaustive enumeration;
  compares QA-solver output against the brute-force optimum for correctness, and
  exercises the `extract_omega` heuristic (§7.4) via the behavioral checks listed there.

---

## 11. Dependencies (proposed `requirements.txt`)

```
numpy
pandas
matplotlib
networkx        # only if used for the direction-augmented graph; optional, see §5.3
dimod           # QUBO data structures, sampler interface
dwave-neal       # classical SA sampler with the same interface as DWaveSampler
pytest
```
All are pip-installable offline-compatible packages (no cloud credentials needed) —
`dwave-neal`/`dimod` specifically chosen so that swapping in real QPU access later is a
matter of installing `dwave-system` and changing one sampler-construction line, not a
rewrite of `qubo_builder.py`, `pricing.py`, or `solver.py`.

---

## 12. Open Items / Assumptions to Flag in the Write-Up

1. **Table I exact numeric reproduction is not guaranteed** — the paper does not publish
   its grid layouts or start/goal coordinates, so our fixed instances (§4) will show the
   same *qualitative* trends (β↑ ⇒ Sp↓ & ℓ↑; exponential ≫ linear amplification) but not
   necessarily identical `ℓ, Sp, τ` numbers.
2. **Eq. (23) (ω-extraction) is not in the provided excerpt** — §7.4 documents our
   reconstructed heuristic and how it will be validated behaviorally rather than by
   formula matching.
3. **CBS has zero variance across seeds by construction** — this will be called out in
   the results write-up rather than silently reported as if it were meaningful spread.
4. Local SA is used as the literal quantum-annealing stand-in per your instruction; the
   module boundary (`qubo_solver.py`) is designed so a real QPU call is a drop-in
   replacement later.

---

## 13. Build Order (for the follow-up implementation session)

1. `graph/grids.py` + `turncost/turncost.py` (+ tests) — no solver dependencies, fastest
   to validate against Table I's own numbers.
2. `cbs/` (+ tests) — fully self-contained baseline, unblocks early end-to-end runs.
3. `graph/spatiotemporal.py` (direction-augmented graph) + `qubo/qubo_builder.py` (+
   tests) — static construction, no solver needed yet.
4. `qubo/qubo_solver.py` (local SA wrapper) — isolated, testable on toy `Q` matrices.
5. `qubo/pricing.py` + `branch_and_price/solver.py` (+ small-instance test) — the
   Algorithm 1 driver, the highest-risk module given the eq. (23) gap.
6. `experiment/` harness + `report/plots.py` — wire everything into the config matrix,
   run all 3 scenarios × configs × 10 seeds × 2 solvers, produce the per-scenario
   folders exactly as specified in §2.
