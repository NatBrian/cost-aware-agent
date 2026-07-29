# Foundation run report — 2026-07-29 (git 9c50b2e)

## 1. What we tested
TODO(3 sentences): trained stopping vs prompt-only vs enforcement on HotpotQA under step budgets.

## 2. Gate verdict: **GO** (medium budget B=4)
- cond1_utility: PASS — {'passed': True, 'a3': 0.2894, 'a1': 0.2052, 'a2': 0.1796}
- cond2_self_stop: PASS — {'passed': True, 'a3': 0.775, 'threshold': 0.7}
- cond3_no_collapse: PASS — {'passed': True, 'a3_f1': 0.5602, 'a2_f1': 0.411, 'margin': 0.05}

## 3. Arm results at B=4

| arm | F1 (95% CI) | steps | utility | self-stop |
|---|---|---|---|---|
| a0 | 0.415 (0.361–0.475) | 3.92 | 0.121 | 78% |
| a1 | 0.471 (0.412–0.528) | 3.54 | 0.205 | 78% |
| a2 | 0.411 (0.353–0.469) | 3.08 | 0.180 | 76% |
| a3 | 0.560 (0.502–0.621) | 3.61 | 0.289 | 78% |

Paired a3−a1 utility: +0.084 (CI +0.025…+0.147, a3 wins 35% of tasks)

Paired a3−a2 utility: +0.110 (CI +0.048…+0.171, a3 wins 34% of tasks)

## 4. Surprises & qualitative examples
TODO: 2–3 quoted trajectories (incl. six-dimension diagnostic verdict: did RL improve only stopping, or also search skill?)

## 5. Judge behavior
- divergence: judge 1.000→0.784, F1 0.232→0.622 over 301 batches (TODO: reading — parallel rise = healthy; judge-up/F1-flat = hacked)

## 6. What this means for paper_plan_v2_1
TODO: adjustment list — confirmed / contradicted / investigate.

## 7. Run costs
TODO: GPU-hours, judge calls (JudgeStats), wall-clock per stage.
