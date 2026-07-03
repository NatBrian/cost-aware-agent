#!/usr/bin/env python3
"""Re-derive OpenCode tool counts + cheat audit from the authoritative daemon
dumps, patching result.json + results.jsonl in place.

OpenCode's --format json stdout is buffered and often truncated (seen: 1 tool
in the parsed events vs 20 in the daemon), so counting/auditing from it is
unreliable — and for the cheat audit, auditing only the parsed subset could
miss a tool that reached the network or escaped the sandbox. The plugin pushed
every tool call to the daemon live, so the daemon dump is the source of truth.
Idempotent: safe to run repeatedly.
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
import run_swe  # reuse NET_RE, CHEAT_RE, DANGER, audit


def reaudit_run(run_dir):
    rj = os.path.join(run_dir, "result.json")
    dj = os.path.join(run_dir, "daemon_dump.json")
    if not (os.path.exists(rj) and os.path.exists(dj)):
        return None
    r = json.load(open(rj))
    if r.get("platform") != "opencode":
        return None
    dump = json.load(open(dj))
    tools = [{"name": tc.get("tool_name"), "input": tc.get("tool_input")}
             for tc in (dump.get("tool_calls") or [])]
    # the sandbox project dir is gone by now; reuse the one recorded in the row
    proj = r.get("project") or ""
    a = run_swe.audit(tools, proj)
    r["tool_calls"] = len(tools)
    r["tool_names"] = [t["name"] for t in tools]
    r["tools_source"] = "daemon"
    r.update(a)
    json.dump(r, open(rj, "w"), indent=2)
    return r


def main():
    patched = {}
    for rj in glob.glob(os.path.join(HERE, "runs", "opencode", "*", "*", "result.json")):
        r = reaudit_run(os.path.dirname(rj))
        if r:
            patched[(r["arm"], r["seed"], r["iid"])] = r
    # rewrite each results.jsonl from the patched result.json files so the
    # aggregate rows match
    for rp in glob.glob(os.path.join(HERE, "runs", "opencode", "*", "results.jsonl")):
        arm_dir = os.path.dirname(rp)
        rows = []
        for line in open(rp):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("arm"), row.get("seed"), row.get("iid"))
            single = os.path.join(arm_dir, row["iid"], "result.json")
            if key in patched and os.path.exists(single):
                rows.append(json.load(open(single)))
            else:
                rows.append(row)
        with open(rp, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"re-audited {len(patched)} opencode runs from daemon dumps")
    for (arm, seed, iid), r in sorted(patched.items()):
        flag = "" if r.get("audit_clean") else "  <-- AUDIT FLAG"
        print(f"  {arm}-s{seed} {iid:26s} tools={r['tool_calls']:3d} "
              f"clean={r.get('audit_clean')}{flag}")


if __name__ == "__main__":
    main()
