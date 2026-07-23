#!/usr/bin/env bash
# Temp-1.0 policy health probe (post-round gate; round-1 lesson: temp-0 val
# checks miss sampling-distribution damage). Gate: malformed <10%, cap <15%.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: probe_policy_health.sh <out.jsonl>}
rm -f "$OUT"
.venv/bin/python -m collect.run_collection --task-file data/hotpotqa_val_50.jsonl \
    --limit 20 --arm a3 --mode none --budget medium --g 2 --temperature 1.0 \
    --out "$OUT"
.venv/bin/python - "$OUT" <<'PYEOF'
import json, sys
eps = [json.loads(l) for l in open(sys.argv[1])]
tot = sum(len(e["steps"]) for e in eps)
mal = sum(1 for e in eps for s in e["steps"] if s["action_type"] == "malformed") / tot
cap = sum(1 for e in eps if e["answered_at"] is None) / len(eps)
f1 = sum(e["final_f1"] for e in eps) / len(eps)
steps = sum(e["steps_used"] for e in eps) / len(eps)
print(f"probe: malformed={mal:.1%} hit_cap={cap:.1%} F1={f1:.3f} steps={steps:.2f}")
ok = mal < 0.10 and cap < 0.15
print("PROBE " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PYEOF
