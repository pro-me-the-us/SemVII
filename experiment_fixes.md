# Required Changes: Making the Turn-Aware MAPF Experiments Sound

Based on verification of `cbs_results_1/2/3.csv` and `qa_results_1/2/3.csv` against the
paper's formulas. τ math and the CBS-vs-QA smoothness improvement both check out
internally (see §5); the items below are what still need fixing before the results are
trustworthy for a write-up.

---

## 1. Fix the scenario instances so there's an actual trade-off to measure
**Priority: highest — this currently invalidates the trade-off plot.**

**Problem:** With corner-to-corner start/goal pairs, the Manhattan-shortest path and the
smoothest (fewest-turn) path are the same path. Every β config in every scenario
returned identical `ℓ_mean` (std=0) — β never had to cost anything, so the
`length_vs_turncost` plot is a vertical line, not a curve.

**Changes to make:**
- **Empty 6x6-2a / 4a:** move goals off the clean diagonal so the shortest path is not a
  simple 1-turn L-shape — e.g. agent 0: `(0,2) → (5,3)`, agent 1: `(5,2) → (0,3)` for the
  2-agent case, and equivalent off-diagonal offsets for the 4-agent case — so a
  minimum-length path requires ≥2 turns, and a smoother alternative only exists at +2
  length or more.
- **Room 6x6-2a:** move the doorway off-center, e.g. to `(3,1)` instead of `(3,3)`, so
  the shortest approach to it requires a sharp turn right before entering, while a
  longer path can approach it straight-on — this actually exercises the "approach the
  doorway at a favorable angle" behavior described in the paper's §V-B3, which the
  current symmetric doorway placement does not.
- Re-run all 3 scenarios × all β configs with the new instances and confirm `ℓ_mean`
  actually changes across the β sweep before trusting anything downstream.

---

## 2. Debug the `converged_rate = 0.0` case (Room 6x6-2a, α=0.5/β=0.5)
**Priority: high — do not report these numbers until this is resolved.**

- Log the stopping-criterion gap (`min_p c̄_p − (v̂ − L(ω))`) at cap-out for a few seeds.
  - Gap near zero but never crossing → tolerance/threshold issue, add an epsilon slack.
  - Gap far from zero → the reconstructed ω-extraction heuristic (stand-in for the
    paper's unseen eq. 23) is wrong for this instance.
- Explicitly confirm capped-out solutions are still conflict-free
  (`detect_conflicts(...) == ∅`) before including them in any results table.
- Try raising the outer-loop iteration cap (e.g. 50 → 150) for this config to check
  whether it's genuinely stuck vs. just slow to converge.
- If possible, locate the actual eq. (23) from Gerlach et al. (arXiv:2501.14568, ref
  [13] in the paper) and replace the reconstructed ω heuristic with the real formula.

---

## 3. Instrument runtime instead of only reporting the total
**Priority: medium — needed to explain the runtime sign-flip between scenarios.**

Currently Empty 6x6-4a gets *slower* as β decreases, while Room 6x6-2a gets *faster* as
β decreases — opposite trends. Add per-run logging of:
- number of outer iterations
- number of inner (column-generation) iterations
- final `|P̄|` (restricted path set size)
- whether the iteration cap was hit

This will show whether both trends are explained by "more iterations needed" (with
K=4 vs K=2 conflict-graph size dominating), or whether something scenario-specific is
going on.

---

## 4. Change how τ is reported, not how it's computed
**Priority: medium — reporting fix only, the underlying math is already correct.**

`τ = e^{Sp} − 1` amplifies any per-seed variance in `Sp` into heavy-tailed variance in
`τ` (e.g. Room 6x6-2a α=1/β=0: `τ_std` (1201) exceeds `τ_mean` (784)). Mean±std alone
will look broken to a reader even though it's mathematically expected.
- Add `tau_median` and `tau_geomean` columns alongside `tau_mean`/`tau_std`.
- Report mean±std for `Sp` (well-behaved) and median/IQR for `τ` (heavy-tailed) in the
  write-up narrative.

---

## 5. Housekeeping
**Priority: low — cleanup, no effect on correctness.**

- Rename output files by scenario instead of index, e.g.
  `cbs_results_empty_6x6_4a.csv`, `qa_results_room_6x6_2a.csv`, and route them into
  `results/<scenario>/` per the original folder plan.
- Increase `num_reads` for the SA solver on configs that showed `Sp_std > 0` (the β=0
  rows) — a higher read count should tighten the stochastic spread across seeds.

---

## 6. What's already verified correct (no change needed)

- τ is computed as `Σ_agent (e^{Sp_agent} − 1)` (per-path, then summed) — confirmed by
  reverse-solving every reported `τ_mean` against integer per-agent `Sp` splits; every
  row matched to 5+ significant figures.
- The core claim holds: at matched path length, QA finds meaningfully smoother paths
  than CBS (e.g. Empty 6x6-2a: Sp 4→2; Room 6x6-2a: Sp 6→5, with a smaller improvement
  under the doorway bottleneck, matching the paper's own qualitative claim in §V-B3).

## Suggested order of work
1. §1 (fix instances) → re-run everything
2. §2 (debug non-convergence) on the new instances
3. §3 (runtime instrumentation) while re-running
4. §4 (reporting) and §5 (housekeeping) once the numbers are trustworthy
