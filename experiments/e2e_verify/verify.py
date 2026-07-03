#!/usr/bin/env python3
"""Post-hoc verifier for the E2E feature-verification runs.

Reads runs/<stamp>/<run>/{result.json,daemon_dump.json,hook.jsonl,events.jsonl}
and machine-checks, per run, that the features that run was designed to
exercise actually fired — plus a global no-cheat audit.

Check kinds:
  HARD  feature must have fired -> FAIL fails the whole verification
  SOFT  expected but timing-dependent (e.g. burn-rate line needs measurable
        trailing spend at injection time) -> WARN only

Usage: verify.py runs/<stamp>
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cost_aware_agent import prompts

CHEAT_RE = re.compile(
    r"\.cost-aware-agent|127\.0\.0\.1:7331|localhost:7331|db\.sqlite|budget\s+set",
    re.IGNORECASE)

results = []


def check(run, kind, name, ok, detail=""):
    results.append({"run": run, "kind": kind, "name": name,
                    "ok": bool(ok), "detail": str(detail)[:200]})
    mark = "PASS" if ok else ("WARN" if kind == "SOFT" else "FAIL")
    print(f"[{run}] {mark:4s} {name}" + (f" — {detail}" if detail and not ok else ""))


def load(root, run, fname):
    p = os.path.join(root, run, fname)
    if not os.path.exists(p):
        return None
    if fname.endswith(".jsonl"):
        out = []
        for line in open(p, errors="replace"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out
    return json.load(open(p))


def inj_texts(dump):
    return [i.get("context") or "" for i in (dump or {}).get("injections") or []]


def tiers_seen(texts):
    return [m.group(1) for t in texts for m in [re.search(r"Tier:\s*(\w+)", t)] if m]


def tool_input_blobs(events):
    """All tool_use inputs from the CLI event stream — CC and OC shapes."""
    blobs = []
    for e in events or []:
        if e.get("type") == "assistant":
            for b in (e.get("message", {}).get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    blobs.append(json.dumps(b.get("input") or {}))
        part = e.get("part") if isinstance(e, dict) else None
        if isinstance(part, dict) and part.get("type") == "tool":
            blobs.append(json.dumps((part.get("state") or {}).get("input") or {}))
    return blobs


def common_checks(run, row, dump, events):
    check(run, "HARD", "run completed (final answer produced)", not row.get("failed"),
          f"timed_out={row.get('timed_out')}")
    check(run, "HARD", "daemon measured spend > 0",
          (row.get("daemon_spent_usd") or 0) > 0, row.get("daemon_spent_usd"))
    check(run, "HARD", "llm_usage rows recorded",
          len((dump or {}).get("llm_usage") or []) > 0)
    check(run, "HARD", "audit: no danger tools", not row.get("danger_tools_used"),
          row.get("danger_tools_used"))
    check(run, "HARD", "audit: zero web requests", row.get("web_requests", 0) == 0)
    check(run, "HARD", "audit: no outside-sandbox path refs",
          not row.get("outside_sandbox_refs"), row.get("outside_sandbox_refs"))
    hits = [b[:120] for b in tool_input_blobs(events) if CHEAT_RE.search(b)]
    check(run, "HARD", "audit: no tool input touches daemon/DB/wallet", not hits, hits)
    # receipt must render from the dump alone, for every session
    try:
        receipt = prompts.render_receipt(dump)
        check(run, "HARD", "receipt renders", "$" in receipt)
    except Exception as e:
        check(run, "HARD", "receipt renders", False, repr(e))
    # negative/zero-token rows would mean the clamps failed at intake
    bad = [u for u in (dump or {}).get("llm_usage") or []
           if any((u.get(k) or 0) < 0 for k in
                  ("input_tokens", "output_tokens", "cache_read_tokens",
                   "cost_usd"))]
    check(run, "HARD", "no negative token/cost rows stored", not bad, bad[:2])


def main():
    root = sys.argv[1]
    rows = {}
    for spec in json.load(open(os.path.join(root, "meta.json")))["runs"]:
        run = spec["run"]
        row = load(root, run, "result.json")
        if row is None:
            check(run, "HARD", "artifacts present", False, "result.json missing")
            continue
        rows[run] = (spec, row, load(root, run, "daemon_dump.json"),
                     load(root, run, "events.jsonl"),
                     load(root, run, "hook.jsonl"))

    for run, (spec, row, dump, events, hook) in rows.items():
        common_checks(run, row, dump, events)
        texts = inj_texts(dump)
        trackers = [t for t in texts if "Budget Tracker" in t]

        if spec["phase"] == "off":
            check(run, "HARD", "OFF: zero injections delivered",
                  row.get("injections_delivered", 0) == 0,
                  row.get("injections_delivered"))
        else:
            check(run, "HARD", "ON: tracker delivered", len(trackers) >= 1)
            check(run, "HARD", "ON: wallet scope in tracker",
                  any("project wallet" in t for t in trackers))
            check(run, "HARD", "ON: delegation line (no canned verdicts)",
                  all("Decide yourself" in t for t in trackers))

        if spec["platform"] == "claude":
            if hook is not None:
                ev = [h.get("event") for h in hook]
                check(run, "HARD", "hook: Stop fired per response (>=1)",
                      ev.count("Stop") >= 1, ev.count("Stop"))
                check(run, "SOFT", "hook: SessionEnd fired naturally",
                      ev.count("SessionEnd") >= 1
                      or row.get("state_before_harness_close") == "ended",
                      f"events={sorted(set(ev))} forced={row.get('forced_stop')}")
                if spec["phase"] == "on":
                    pre = ev.count("PreToolUse")
                    check(run, "HARD", "on_change: injections < PreToolUse fires",
                          row.get("injections_delivered", 9e9) <= max(2, pre),
                          f"inj={row.get('injections_delivered')} pre={pre}")
            # daemon vs CLI billing agreement (CLI total includes subagents)
            cli, dmn = row.get("cli_cost_usd"), row.get("daemon_spent_usd")
            if cli and dmn:
                check(run, "SOFT", "daemon spend within 25% of CLI-reported cost",
                      abs(dmn - cli) / cli < 0.25, f"cli={cli:.4f} daemon={dmn:.4f}")

        if spec["platform"] == "opencode" and spec["phase"] == "on":
            # rebuilt channel: approximate '~$' figures, byte-stable repeats
            check(run, "HARD", "rebuilt: approximate (~$) tracker",
                  any("~$" in t for t in trackers))
            # Byte-stability = spend rendered quantized to the injection-bucket
            # grid (10% slices of the budget), so the text only changes when a
            # bucket/tier boundary is crossed. At sub-cent budgets one deepseek
            # call can cross several buckets, so consecutive-identical repeats
            # may legitimately never occur — the checkable invariant is that
            # every rendered spend sits ON the grid, never a raw running total.
            budget = (row.get("budget_view") or {}).get("budget_usd") or 0
            spends = [float(m.group(1)) for t in trackers
                      for m in [re.search(r"LLM cost used: ~\$([0-9.]+)", t)] if m]
            step = budget * 0.10
            on_grid = bool(spends) and step > 0 and all(
                math.isclose(s / step, round(s / step), abs_tol=1e-6)
                for s in spends)
            check(run, "HARD", "rebuilt: spend quantized to bucket grid",
                  on_grid, f"spends={spends} step={step:.6f}")
            # Deliveries with the SAME quantized state must be byte-identical.
            # The quantized state is (spend bucket, tier, burn step) — the burn
            # line is itself quantized to the bucket step and stepping it is a
            # legitimate transition (at sub-cent budgets it steps almost every
            # call: the known structural residual of the rebuilt-channel tax).
            by_state = {}
            dup_ok = True
            for t in trackers:
                sm = re.search(r"LLM cost used: ~\$([0-9.]+)", t)
                bm = re.search(r"Burn rate: ~\$([0-9.]+)", t)
                tm = re.search(r"Tier:\s*(\w+)", t)
                state = (round(float(sm.group(1)) / step) if sm and step else 0,
                         round(float(bm.group(1)) / step) if bm and step else 0,
                         tm.group(1) if tm else "")
                if state in by_state and by_state[state] != t:
                    dup_ok = False
                by_state.setdefault(state, t)
            check(run, "HARD", "rebuilt: same-quantized-state deliveries byte-identical",
                  dup_ok, f"{len(set(trackers))} unique of {len(trackers)}, "
                          f"{len(by_state)} states")

        # run-specific features
        if run in ("cc3", "oc3"):
            check(run, "HARD", "history: [PROJECT HISTORY] injected",
                  any("[PROJECT HISTORY]" in t for t in texts))
            bv = row.get("budget_view") or {}
            own = row.get("daemon_spent_usd") or 0
            check(run, "HARD", "wallet depletion: view spend > own session spend",
                  (bv.get("spent_usd") or 0) > own,
                  f"view={bv.get('spent_usd')} own={own}")
        if run == "cc4":
            srcs = {u.get("source") for u in (dump or {}).get("llm_usage") or []}
            check(run, "HARD", "subagent usage captured (source=pull-subagent)",
                  "pull-subagent" in srcs, sorted(srcs))
            check(run, "HARD", "subagent transcripts archived",
                  os.path.isdir(os.path.join(root, run, "subagents")))
            # CC >= 2.1.x surfaces the subagent tool as "Agent" (older: "Task")
            check(run, "HARD", "subagent tool actually used",
                  {"Task", "Agent"} & set(row.get("tool_names") or []),
                  row.get("tool_names"))
        if run == "cc5":
            check(run, "HARD", "checkpoint question injected",
                  any("BUDGET CHECKPOINT" in t for t in texts))
        if run in ("cc5", "oc4", "oc5"):
            seen = tiers_seen(texts)
            check(run, "HARD", "tight budget: LOW/CRITICAL tier reached",
                  any(t in ("LOW", "CRITICAL") for t in seen), seen)
            check(run, "HARD", "advisory-only: run still completed over pressure",
                  not row.get("failed") and not row.get("timed_out"))
        if spec["phase"] == "on" and spec["platform"] == "claude":
            check(run, "SOFT", "burn-rate line appeared in some tracker",
                  any("Burn rate:" in t for t in trackers))

    hard_fails = [r for r in results if not r["ok"] and r["kind"] == "HARD"]
    warns = [r for r in results if not r["ok"] and r["kind"] == "SOFT"]
    summary = {"total": len(results), "hard_fails": len(hard_fails),
               "warns": len(warns), "checks": results}
    json.dump(summary, open(os.path.join(root, "verify.json"), "w"), indent=2)
    print(f"\n=== {len(results)} checks, {len(hard_fails)} HARD FAIL, "
          f"{len(warns)} soft warn -> {os.path.join(root, 'verify.json')}")
    sys.exit(1 if hard_fails else 0)


if __name__ == "__main__":
    main()
