"""Calibration tooling (F3): labeling sheet for Brian + judge-vs-human agreement.

Workflow (E-b):
  1. make_labeling_sheet(pilot_jsonl, out_csv)  -> stratified 50 steps, one row
     each, with the exact context the judge sees and empty label columns.
  2. Brian fills the label columns (0/1) — plain instructions in the CSV header.
  3. agreement(sheet_csv, judge)                -> per-bit agreement + confusion;
     gate per config (>=0.80 per bit, floor 0.70).
"""

import csv
import json
import random
from pathlib import Path

from reward.rubric import (ANSWER_BITS, STEP_BITS, render_answer_prompt,
                           render_step_prompt)

INSTRUCTIONS = ("Fill every label_* column with 0 or 1 per the bit definitions "
                "in reward/rubric.py (answer steps use supported/nothing_left; "
                "leave the other columns blank). Judge only from the context "
                "column — do not look up answers.")


def _strata_of(ep: dict, idx: int) -> str:
    s = ep["steps"][idx]
    if s["action_type"] == "answer":
        return "answer"
    t, tmax = s["t"], max(e["t"] for e in ep["steps"])
    return "early" if t <= tmax / 3 else ("late" if t > 2 * tmax / 3 else "mid")


def make_labeling_sheet(pilot_jsonl: str | Path, out_csv: str | Path,
                        n: int = 50, seed: int = 42) -> int:
    episodes = [json.loads(l) for l in open(pilot_jsonl) if l.strip()]
    pool = []  # (stratum, ep, idx)
    for ep in episodes:
        for i, s in enumerate(ep["steps"]):
            if s["action_type"] in ("search", "answer"):
                pool.append((_strata_of(ep, i), ep, i))
    rng = random.Random(seed)
    by_stratum: dict[str, list] = {}
    for row in pool:
        by_stratum.setdefault(row[0], []).append(row)
    quota = {k: max(1, round(n * len(v) / len(pool)))
             for k, v in by_stratum.items()}
    picked = []
    for k, rows in sorted(by_stratum.items()):
        picked.extend(rng.sample(rows, min(quota[k], len(rows))))
    picked = picked[:n]

    fields = (["sheet_id", "stratum", "task_id", "t", "action_type", "context"]
              + [f"label_{b}" for b in STEP_BITS + ANSWER_BITS])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        f.write(f"# {INSTRUCTIONS}\n")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sid, (stratum, ep, i) in enumerate(picked):
            s = ep["steps"][i]
            prompt = (render_answer_prompt(ep, i) if s["action_type"] == "answer"
                      else render_step_prompt(ep, i))
            w.writerow({"sheet_id": sid, "stratum": stratum,
                        "task_id": ep["task_id"], "t": s["t"],
                        "action_type": s["action_type"], "context": prompt})
    return len(picked)


def agreement(sheet_csv: str | Path, judge, cfg_rubric: dict) -> dict:
    """Judge every labeled row; per-bit agreement + confusion + gate verdict."""
    with open(sheet_csv) as f:
        lines = [l for l in f if not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    per_bit: dict[str, list[tuple[int, int]]] = {}
    neutral: dict[str, int] = {}
    for row in rows:
        bits = (ANSWER_BITS if row["action_type"] == "answer" else STEP_BITS)
        got = judge.judge(row["context"], bits)
        for b in bits:
            human = row.get(f"label_{b}", "").strip()
            if human not in ("0", "1"):
                continue
            v = float(got[b])
            # A neutral 0.5 means the judge had NO OPINION (parse/transport
            # failure). int(round(0.5)) == 0 in Python's banker's rounding, so
            # the old path silently scored "no opinion" as a confident NO and
            # folded judge outages into the agreement number. Count them
            # separately instead. (audit 2026-07-28)
            if v not in (0.0, 1.0):
                neutral[b] = neutral.get(b, 0) + 1
                continue
            per_bit.setdefault(b, []).append((int(human), int(v)))
    gate, floor = cfg_rubric["calibration"]["per_bit_gate"], cfg_rubric["calibration"]["per_bit_floor"]
    report: dict = {"bits": {}, "gate": gate, "floor": floor}
    scores = []
    for b, pairs in sorted(per_bit.items()):
        agree = sum(h == j for h, j in pairs) / len(pairs)
        conf = {"h1_j1": sum(h == 1 and j == 1 for h, j in pairs),
                "h0_j0": sum(h == 0 and j == 0 for h, j in pairs),
                "h1_j0": sum(h == 1 and j == 0 for h, j in pairs),
                "h0_j1": sum(h == 0 and j == 1 for h, j in pairs)}
        report["bits"][b] = {"n": len(pairs), "agreement": round(agree, 3),
                             "confusion": conf, "neutral_dropped": neutral.get(b, 0)}
        scores.append(agree)
    report["mean_agreement"] = round(sum(scores) / len(scores), 3) if scores else 0.0
    report["neutral_dropped_total"] = sum(neutral.values())
    # TWO readings, both reported, because they disagree (audit 2026-07-28):
    #  - strict:     plan §5 / F3 line 118 literally say ">=80% PER BIT".
    #  - mean+floor: what this function used to gate on alone (mean >= .80 and
    #                no bit < .70) — looser, and it is what the 2026-07-22 run
    #                passed (mean .848 with new_info .792, nothing_left .769,
    #                i.e. two bits that the strict reading fails).
    # `passed` follows the SPEC. Shipping the looser one as the only number is
    # how a gate quietly stops being a gate.
    report["passed_strict_per_bit"] = bool(scores) and min(scores) >= gate
    report["passed_mean_and_floor"] = (bool(scores)
                                       and report["mean_agreement"] >= gate
                                       and min(scores) >= floor)
    report["passed"] = report["passed_strict_per_bit"]
    return report
