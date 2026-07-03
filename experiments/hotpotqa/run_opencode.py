#!/usr/bin/env python3
"""Cross-agent arm: the SAME multi-hop QA / money-budget harness driven through
OpenCode (model = deepseek-v4-flash-free via the `opencode` zen gateway) instead
of the Claude CLI. Purpose: show the cost-aware-agent daemon's money-tracking +
budget-injection path works cross-agent, exactly as the README claims.

It reuses run_claude.py's corpus, retrieval tools, grading, prompt builder,
ReAct parser, and daemon routing verbatim — only the model gateway changes. Every
turn still goes through the LIVE daemon (/session/start, /tool/pre, /tool/post,
/llm/usage), and the daemon's injected budget text is fed into the prompt, so the
identical mechanism is under test on a different agent.

MONEY on a FREE model — priced at retail. The user's OpenCode run of
deepseek-v4-flash-free bills $0, but the daemon prices it at the *paid*
deepseek-v4-flash retail rate (cost.price_for_model strips the `-free` suffix by
design: "simulated real-market cost, so the agent gets genuine budget pressure
regardless of which account it runs under"). So the money-tracking path IS live
here — but deepseek is ~20x cheaper per token than Sonnet, so a whole session
costs only cents. The Claude arm's $0.30/$0.60/$1.20 tiers would never bite on it
(spend stays far under $0.30, tier pinned HIGH); the OpenCode budgets must be
scaled down to deepseek's own cost so the tier actually moves. The orchestration
picks a biting budget from an OFF run's measured mean cost. This arm proves the
daemon tracks money and injects budget CROSS-AGENT, and — with a scaled budget —
that the same money-reduction effect reproduces on a second, cheaper agent.

Gateway: `opencode run --pure --format json -m opencode/deepseek-v4-flash-free`.
--pure disables OpenCode's own plugins/tool scaffolding so it acts as a clean
text-completion gateway (we route through the daemon ourselves, just like the
Claude arm does not use Claude Code's hooks). --format json exposes the model
text parts and a step-finish `tokens` block we map onto /llm/usage.

Usage: run_opencode.py --tag oc_off --n 5
"""
import argparse
import datetime
import json
import os
import signal
import subprocess
import time

import run_claude as rc  # reuse corpus, tools, grading, prompt, parsing, daemon I/O
from cost_aware_agent import cost as costeng  # same retail pricing the daemon uses

HERE = rc.HERE
RUNS_DIR = rc.RUNS_DIR
OC_MODEL = "opencode/deepseek-v4-flash-free"
PRICED_MODEL = "deepseek-v4-flash-free"  # daemon pricing key; cost.py strips -free
                                         # and prices at paid deepseek-v4-flash retail


def _run_killable(cmd, timeout, cwd):
    """Same process-group kill guard as the Claude arm (opencode also spawns a
    node child tree that can wedge the stdout pipe past a naive timeout)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, start_new_session=True, cwd=cwd)
    try:
        out, _ = p.communicate(timeout=timeout)
        return out or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        p.communicate()
        raise


def opencode_call(prompt, tries=4):
    """Run one deepseek turn through OpenCode headless. Returns
    (text, cost_usd, usage, meta) mirroring rc.claude_call's contract (minus the
    Claude-specific session id / event stream). usage is normalized to the field
    names report_llm_usage expects."""
    last = None
    for i in range(tries):
        try:
            out = _run_killable(
                ["opencode", "run", "--pure", "--format", "json",
                 "-m", OC_MODEL, prompt],
                timeout=180, cwd=rc._SANDBOX)
            events = []
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
            text_parts, tokens, cost = [], {}, 0.0
            for e in events:
                part = e.get("part", {}) if isinstance(e, dict) else {}
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "step-finish":
                    tokens = part.get("tokens", {}) or tokens
                    cost += part.get("cost", 0.0) or 0.0
            text = "".join(text_parts).strip()
            if not text and not tokens:
                raise RuntimeError(f"no text/tokens; out head: {out[:160]!r}")
            cache = tokens.get("cache", {}) or {}
            usage = {
                "input_tokens": tokens.get("input", 0),
                # deepseek reasoning tokens are billed as output — fold them in
                "output_tokens": tokens.get("output", 0) + tokens.get("reasoning", 0),
                "cache_read_input_tokens": cache.get("read", 0),
                "cache_creation_input_tokens": cache.get("write", 0),
            }
            meta = {"tokens": tokens, "cost": cost,
                    "tool_uses": [], "permission_denials": []}
            return text, cost, usage, meta, events
        except Exception as e:
            last = e
            time.sleep(3 + 3 * i)
    raise RuntimeError(f"opencode call failed after {tries}: {last}")


def run_question(q, run_id, traces_dir):
    sid = f"{run_id}-{q['id']}"
    tf = open(os.path.join(traces_dir, f"{q['id']}.jsonl"), "w")

    def trace(kind, **kw):
        tf.write(json.dumps({"kind": kind, "wall": time.time(), **kw}) + "\n")
        tf.flush()

    transcript = []
    trace("session", session_id=sid, question=q["question"], gold=q["answer"],
          gold_titles=q.get("gold_titles"))
    plan = rc.daemon_post("/session/start", {
        "session_id": sid, "cli": "qa-opencode", "task": q["question"],
        "model": PRICED_MODEL})
    trace("session_start", plan_seed=plan, fed_to_model=False)

    tool_calls_used = 0
    cost = 0.0
    out_tok = 0
    hit_cap = False
    answer = "(no answer)"

    for step in range(rc.MAX_STEPS):
        tracker = rc.daemon_post("/tool/pre", {"session_id": sid, "tool_name": "search",
                                               "channel": "rebuilt"})
        prompt = rc.build_prompt(q["question"], transcript, tracker)
        text, _oc_cost, usage, meta, events = opencode_call(prompt)
        # MONEY = simulated retail cost. opencode's own `cost` is $0 (free tier);
        # price the same token usage at deepseek-v4-flash retail (exactly what the
        # daemon does internally for its budget tier), so the row's cost_usd is
        # the real money this usage would cost and the budget has something to bite.
        c = costeng.cost_llm_usage(
            PRICED_MODEL, usage["input_tokens"], usage["output_tokens"],
            usage["cache_read_input_tokens"], usage["cache_creation_input_tokens"], 0)
        cost += c
        out_tok += (usage or {}).get("output_tokens", 0)
        trace("cli_raw", step=step, events=events)
        # Price the daemon's budget against the model that ACTUALLY ran (deepseek),
        # not run_claude's default Sonnet key — otherwise the injected budget
        # spend is ~20x too high and every sub-cent OpenCode budget pins to
        # CRITICAL on call 1 (oc6 and oc12 become indistinguishable).
        rc.report_llm_usage(sid, {**usage, "cache_creation": {}},
                             message_id=f"{sid}-s{step}", priced_model=PRICED_MODEL)
        verb, arg = rc.parse_action(text)
        trace("step", step=step, injected_tracker=tracker,
              sim_budget=rc.parse_tracker(tracker), prompt=prompt,
              model_raw=text, action=verb, arg=arg, usage=usage, cost_usd=c,
              tokens=meta.get("tokens"), tool_calls_used=tool_calls_used)

        if verb == "ANSWER":
            answer = arg
            break
        if verb == "RETRY":
            transcript.append("[SYSTEM] Invalid output. Reply with exactly one "
                              "line: SEARCH:/READ:/ANSWER:.")
            trace("retry", step=step)
            continue
        if tool_calls_used >= rc.MAX_TOOLS:
            hit_cap = True
            answer = "(hit tool cap, no answer)"
            break
        tool_calls_used += 1

        if verb == "SEARCH":
            obs = json.dumps(rc.tool_search(arg))
        else:
            obs = rc.tool_read(arg)
        transcript.append(f"> {verb}: {arg}")
        transcript.append(f"OBSERVATION: {obs[:1200]}")
        trace("observation", step=step, verb=verb, arg=arg, observation=obs)
        post = rc.daemon_post("/tool/post", {
            "session_id": sid, "tool_name": verb.lower(),
            "tool_input": {"arg": arg}, "tool_result": obs[:500]})
        trace("tool_post", step=step, daemon_context=post)
    else:
        answer = "(max steps, no answer)"

    em, f1 = rc.score(answer, q["answer"])
    row = {"id": q["id"], "session_id": sid, "question": q["question"],
           "gold": q["answer"], "answer": answer, "em": em, "f1": round(f1, 3),
           "tool_calls": tool_calls_used, "cost_usd": round(cost, 6),
           "out_tok": out_tok, "hit_cap": hit_cap,
           # free model: no CLI tool channel exists in --pure mode, so clean by construction
           "audit_clean": True, "cli_tool_uses": [], "cli_denials": 0,
           "web_requests": 0, "trace": f"traces/{q['id']}.jsonl"}
    trace("result", **row)
    tf.close()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    questions = json.load(open(os.path.join(HERE, "data", "questions.json")))
    questions = questions[args.start:args.start + args.n]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.tag}-{stamp}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    condition = rc.read_daemon_condition()
    meta = {"run_id": run_id, "tag": args.tag, "model": OC_MODEL,
            "gateway": "opencode-cli", "priced_model": PRICED_MODEL,
            "dataset": "HotpotQA (distractor, screened retrieval-forcing subset)",
            "condition": condition, "git_sha": rc.git_sha(), "args": vars(args),
            "questions": [q["id"] for q in questions], "started": stamp}
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=2)
    if os.path.exists(rc.CONFIG_PATH):
        import shutil
        shutil.copy(rc.CONFIG_PATH, os.path.join(run_dir, "config_snapshot.json"))
    print(f"[{run_id}] condition={condition} model={OC_MODEL}", flush=True)

    rows = []
    with open(os.path.join(run_dir, "results.jsonl"), "w") as f:
        for q in questions:
            try:
                r = run_question(q, run_id, traces_dir)
            except Exception as e:
                r = {"id": q["id"], "session_id": f"{run_id}-{q['id']}",
                     "question": q["question"], "gold": q["answer"],
                     "answer": f"(ERROR: {str(e)[:80]})", "em": 0.0, "f1": 0.0,
                     "tool_calls": 0, "cost_usd": 0.0, "out_tok": 0,
                     "hit_cap": False, "failed": True}
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{run_id}] {r['id']} calls={r['tool_calls']} f1={r['f1']} "
                  f"em={r['em']} cost={r['cost_usd']} tok={r.get('out_tok')} "
                  f"fail={r.get('failed', False)} ans={r['answer'][:40]!r}", flush=True)

    ok = [r for r in rows if not r.get("failed")]
    n = len(ok) or 1
    summary = {"run_id": run_id, "n_ok": len(ok), "n_total": len(rows),
               "condition": condition,
               "mean_tool_calls": round(sum(r["tool_calls"] for r in ok) / n, 3),
               "mean_out_tok": round(sum(r["out_tok"] for r in ok) / n, 1),
               "mean_f1": round(sum(r["f1"] for r in ok) / n, 3),
               "mean_em": round(sum(r["em"] for r in ok) / n, 3),
               "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
               "finished": datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    import shutil
    shutil.copy(os.path.join(run_dir, "results.jsonl"),
                os.path.join(HERE, "results", f"{args.tag}.jsonl"))
    print(f"\n=== {run_id} summary ===")
    for k in ("mean_tool_calls", "mean_out_tok", "mean_f1", "mean_em", "total_cost_usd"):
        print(f"{k:16} {summary[k]}")
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
