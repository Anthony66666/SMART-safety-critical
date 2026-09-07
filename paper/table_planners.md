# Planner evaluation on nuPlan benchmarks (our runs)

Layout follows Bosch Table I. Every number below is from our own harness
(official `run_simulation.py`, `two_stage_controller`, official metrics and
aggregator); nothing is copied from published tables.

Columns: OLS = open-loop (not run by us). CLS-NR = closed-loop, log-replay
traffic. CLS-R = closed-loop, IDM traffic. CLS-SR = closed-loop, SMART traffic.
The parenthesis after CLS-SR is CLS-SR − CLS-R, to one decimal.

CLS-SR appears twice because we have two SMART variants and want both in the
paper: **closed** = population fixed at t=0 (the runs archived on 2026-09-07 as
`val14_closed_20260907` / `local_test14hard_closed_20260907`); **open** =
vehicles enter and leave, pedestrians replayed (commits `1c84650`..`4bdb2c4`),
val14 rerun in progress on the server, test14-hard not yet run.

Val14: n = 1118 scenarios per cell. Test14-hard: n = 272. `—` = not run.

| Type | Paradigm | Method | OLS | Val14 CLS-NR | Val14 CLS-R | Val14 CLS-SR closed | Val14 CLS-SR open | T14-hard CLS-R | T14-hard CLS-SR closed | T14-hard CLS-SR open |
|---|---|---|---|---|---|---|---|---|---|---|
| Expert Log | Human | Log Replay | — | — | — | — | — | — | — | — |
| Rule-Based | Rule | IDM Planner | — | 75.60 | 77.33 | — | — | — | — | — |
| Rule-Based | Rule | PDM-Closed | — | 92.84 | 92.13 | 85.16 (−7.0) | — | — | 63.62 | — |
| Hybrid | IL + Rule | PDM-Hybrid | — | 92.77 | 92.11 | — | — | — | — | — |
| Hybrid | IL + Rule | GameFormer | — | — | — | — | — | — | — | — |
| Hybrid | IL + Rule | DTPP | — | — | — | 52.42 | — | — | — | — |
| Hybrid | IL + Rule | PLUTO | — | — | — | — | — | — | — | — |
| Learned | IL | Urban Driver | — | 53.05 | 50.42 | 38.13 (−12.3) | — | — | 27.60 | — |
| Learned | IL | GC-PGP | — | 57.12 | 55.55 | 47.41 (−8.1) | — | — | 31.44 | — |
| Learned | IL | PlanCNN | — | — | — | — | — | — | — | — |
| Learned | IL | PlanTF | — | — | — | — | — | — | — | — |
| Learned | IL | PDM-Open | — | 50.24 | 54.25 | — | — | — | — | — |
| Learned | IL | Diffusion Planner | — | 89.53 | 82.83 | 70.29 (−12.5) | — | — | 52.31 | — |
| Learned | IL | Flow Planner | — | 88.08 | 81.55 | 71.14 (−10.4) | — | — | — | — |
| Learned | RL | CaRL | — | — | — | 82.99 | — | — | — | — |

## What is missing and why

- **OLS**: never run; not part of this benchmark's question.
- **IDM / PDM-Hybrid / PDM-Open, CLS-SR**: dropped from the SMART sweep on
  purpose (IDM is immune to occlusion at 0.000000 m, PDM-Open is open-loop with
  a ±0.00 gap, PDM-Hybrid is within 0.07 of PDM-Closed).
- **DTPP / CaRL, CLS-NR and CLS-R**: only their SMART condition was run.
- **PlanCNN**: every Ray run died of CUDA OOM (5/5); it is being rerun in
  sequential mode in the open sweep.
- **GameFormer / PLUTO / PlanTF**: not installed on either machine.
- **Test14-hard CLS-R**: never run locally (only SMART was).
- **Test14-hard Flow Planner**: the local run was killed by the OOM killer
  after 4 scenarios; excluded rather than reported on n = 4.
- **Expert Log**: no log-replay-as-planner run.

## Caveats that belong in the caption

- The occlusion gaps (baseline − occluded within each traffic model) are the
  paper's actual measurement and live in a separate table; this one is the
  absolute-score companion.
- CLS-SR closed was measured on traffic that thins over the scenario (population
  fixed at t=0, 24–33 vehicles near the ego by 14 s against the log's 46–68).
  Its SR − R drops of −7 to −12.5 are far larger than Bosch's −2 to −8 and
  should not be read as a property of SMART alone until the open column is in.
- Our Flow Planner CLS-NR baseline is 88.08 (87.48 through the authors' own
  script) against the published 90.43; treat published-vs-ours comparisons as
  cross-checks of the harness, never as inputs to any gap.
- No noise floor has been established for the stochastic planners (Flow,
  Diffusion); a Flow baseline replicate once differed by −0.78.
