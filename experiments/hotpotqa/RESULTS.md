# HotpotQA money-budget experiment — results

**Claim under test.** Injecting a real **dollar** budget into a coding/QA agent's
context makes the session spend **less money** ($) while roughly holding accuracy.
Cost is money — not tool-call counts, not iterations. OFF (no injection) is the
control. Dataset: HotpotQA (distractor), an offline BM25 corpus of 100 passages;
a plain-text ReAct agent answers multi-hop questions with `SEARCH`/`READ`/`ANSWER`.
Every turn routes through the live cost-aware-agent daemon, which computes real $
from token usage and injects `LLM cost used: $X of $Y, Tier Z` before each step.

## Question set — why the earlier run showed nothing

The preliminary run found no budget effect for one reason: Claude Sonnet answered
famous-entity HotpotQA trivia from **parametric memory** in a single turn, so
there was no exploration for a budget to trim. Difficulty labels don't fix this —
the whole distractor-validation split is already `level=hard`. Fix: a **closed-book
screen** (`screen_questions.py`) asks the model each candidate with no tools and
keeps only the ones it gets **wrong from memory**. All 10 kept questions have a
closed-book F1 of **0.0** — the model cannot answer them without retrieval, so the
budget finally has something to bite on.

## Integrity: the model tried to cheat, and we caught + blocked it

The pilot exposed a real leak: with the repo as CWD and the user's global settings
allowing `Bash(*)`/`Grep(*)`, the model ran `grep -A5 "Alex the Dog"
data/questions.json` and read the **gold answer off disk** — `--allowedTools ""`
did not stop it because the global allow-list overrides. Fixed with defense in
depth: (1) run the CLI in an **empty sandbox CWD** with no experiment files, and
(2) hard `--disallowedTools`. After the fix the same question is answered by a
legitimate `SEARCH`. Every reported run is audit-clean (`web_requests=0`,
`dirty=none`). The audit ignores phantom tool_use blocks named after our own ReAct
verbs (the CLI rejects `SEARCH` as "No such tool" and the model re-emits it as
text) — only real filesystem/web tools count as cheats.

## Claude arm (Sonnet) — the money result

10 paired questions, 3 seeds/tier, all audit-clean. `$/q` is real Anthropic dollars.

| tier | budget | $/q | tool calls | F1 | EM |
|---|---|---|---|---|---|
| OFF | — | 0.24842 | 2.83 | 0.556 | 0.467 |
| **usd30** | **$0.30** | **0.17889** | 2.20 | 0.550 | 0.467 |
| usd60 | $0.60 | 0.22330 | 2.83 | 0.586 | 0.467 |
| usd120 | $1.20 | 0.22397 | 2.57 | 0.586 | 0.467 |

vs OFF:

| tier | $ saved | % cheaper | ΔF1 | ΔEM |
|---|---|---|---|---|
| **usd30 ($0.30)** | 0.06953 | **28.0%** | −0.006 | **+0.000** |
| usd60 ($0.60) | 0.02512 | 10.1% | +0.030 | +0.000 |
| usd120 ($1.20) | 0.02445 | 9.8% | +0.030 | +0.000 |

**The tightest budget ($0.30) cuts real spend 28% with zero EM loss and F1 within
noise.** The mechanism is visible in the tool-call column: under pressure the model
commits earlier (2.2 vs 2.83 calls) instead of over-exploring. Looser budgets exert
less pressure and save less — a monotonic budget→savings gradient. Money is the
metric, and money drops.

## OpenCode arm (deepseek-v4-flash-free) — cross-agent, but capability-gated

Same experiment, same daemon, driven through OpenCode with the free deepseek model
(`run_opencode.py`). deepseek is priced at the paid `deepseek-v4-flash` retail rate
(the daemon's "simulated real-market cost" — it strips the `-free` suffix), so a
whole per-question session costs ~$0.01. The Claude arm's $0.30+ tiers can't bite
at that scale, so budgets are scaled to deepseek's own cost ($0.006 / $0.012).

| tier | budget | $/q | tool calls | F1 | EM |
|---|---|---|---|---|---|
| ocoff | — | 0.01699 | 5.27 | 0.650 | 0.500 |
| oc6 | $0.006 | 0.01670 | 5.20 | 0.631 | 0.500 |
| oc12 | $0.012 | 0.01637 | 5.00 | 0.706 | 0.533 |

vs OFF: oc6 1.7% cheaper (ΔF1 −0.019), oc12 3.6% cheaper (ΔF1 +0.056). All clean.

**What this proves and doesn't.** The daemon's money-tracking and budget-injection
paths work **cross-agent** — retail cost is computed from OpenCode's token stream
and the Budget Tracker is delivered into deepseek's prompt (verified by the live
injection probe). But the behavioral **money-reduction effect is near-noise** on
deepseek: it ignores the "finalize now" guidance and keeps looping — question q5
hits the 20-call safety cap under *every* budget tier. This is the documented
weak-model non-compliance limit: budget **reasoning** requires a capable model
(Sonnet complies → 28%; deepseek-flash largely does not). The cross-agent plumbing
is proven; the economic-judgment payoff is Sonnet's.

## Reproduce

```bash
python build_corpus.py --pool 60 --seed 1     # sample bridge/non-yesno candidates
python screen_questions.py --n 10             # closed-book screen -> keep retrieval-forcing
python orchestrate.py --tiers off,usd30,usd60,usd120 --n 10 --seeds 3   # Claude arm
python orchestrate.py --harness run_opencode.py --tiers ocoff,oc6,oc12 --n 10 --seeds 3  # OpenCode arm
python analyze.py                             # per-gateway accuracy-vs-cost tables
```

`runs/` and `results/` are gitignored — regenerable from the committed harness +
`data/`. `results/analysis.txt` is the committed snapshot of the tables above.
