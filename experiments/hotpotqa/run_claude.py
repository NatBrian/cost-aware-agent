#!/usr/bin/env python3
"""Multi-hop QA A/B harness (Claude Sonnet gateway) — BATS replication.

Model = Claude Sonnet via the local `claude -p` CLI (headless JSON). Instead of
OpenAI function-calling we drive a plain-text ReAct protocol: each turn the model
emits one line — SEARCH:/READ:/ANSWER: — which we parse, execute against the
offline HotpotQA distractor corpus, and feed back as an observation. This avoids
the OpenAI `content:null` tool-result-dropping bug entirely.

Every turn is routed through the LIVE cost-aware-agent daemon
(/session/start, /tool/pre, /tool/post) so the daemon's real injection +
tool-call-budget logic is what is under test. Condition (inject_enabled /
session_budget_estimate_usd) is set on the daemon BEFORE this runs.

FULL TRACEABILITY. Every run writes a self-describing directory:

  runs/<run_id>/
    meta.json             tag, model, condition, git sha, args, timestamps
    config_snapshot.json  daemon config.json copied at run start
    results.jsonl         one final row per question
    summary.json          aggregates
    traces/<qid>.jsonl    EVERY step: injected budget text, full prompt sent,
                          raw model output, parsed action, observation, usage, cost

session_id is unique per run (<run_id>-<qid>) so daemon-side rows never collide
across runs. results/<tag>.jsonl is also refreshed as a convenience pointer for
analyze.py (latest run of that tag).

Usage:
  run_claude.py --tag off   --n 5
  run_claude.py --tag on10  --n 5
"""
import argparse
import datetime
import json
import os
import re
import shutil
import string
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
CONFIG_PATH = os.path.expanduser("~/.cost-aware-agent/config.json")
MODEL = "sonnet"              # value for the claude CLI --model flag
PRICED_MODEL = "claude-sonnet-5"  # LiteLLM pricing key the daemon costs against
DAEMON = "http://127.0.0.1:7331"

# Hard safety ceiling on tool executions per question. Above the largest budget
# tier so the budget stays advisory — we observe whether the model stops on its
# own, not force it. A run that hits this cap is flagged.
MAX_TOOLS = 20
MAX_STEPS = 25

# --- corpus + retrieval tools (BM25 over the offline HotpotQA passages) ---
_corpus = json.load(open(os.path.join(HERE, "data", "corpus.json")))
_titles = list(_corpus)
_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
from rank_bm25 import BM25Okapi  # noqa: E402
_bm25 = BM25Okapi([_tok(f"{t} {_corpus[t]}") for t in _titles])


def tool_search(query, k=5):
    scores = _bm25.get_scores(_tok(query))
    order = sorted(range(len(_titles)), key=lambda i: scores[i], reverse=True)[:max(1, int(k))]
    return [{"title": _titles[i], "snippet": _corpus[_titles[i]][:200]} for i in order]


def tool_read(title):
    if title in _corpus:
        return _corpus[title]
    for t in _titles:
        if t.lower().strip() == str(title).lower().strip():
            return _corpus[t]
    return f"ERROR: no passage titled {title!r}. Use SEARCH to find exact titles."


# --- model gateway: claude -p headless ---

def claude_call(prompt, tries=4):
    """Call the Claude CLI headless in STREAM mode and return
    (text, cost_usd, usage, session_id, meta).

    Uses --output-format stream-json --verbose so every intermediate message is
    emitted, not just the final aggregate. We parse each assistant message's
    content blocks and record the NAME + input of every tool_use, plus any
    permission_denials from the result event. This makes tool usage auditable by
    name — the check for whether the model cheated (web/local tools) instead of
    relying only on the aggregate web_search counter. With --allowedTools "" the
    expected tool_uses list is EMPTY; anything in it is a real signal."""
    last = None
    for i in range(tries):
        try:
            p = subprocess.run(
                ["claude", "-p", prompt, "--model", MODEL,
                 "--output-format", "stream-json", "--verbose", "--allowedTools", ""],
                capture_output=True, text=True, timeout=180)
            events = []
            for line in p.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
            result = next((e for e in reversed(events) if e.get("type") == "result"), None)
            if result is None:
                raise RuntimeError(f"no result event; stdout head: {p.stdout[:160]!r}")
            if result.get("is_error"):
                raise RuntimeError(str(result.get("result"))[:120])
            # Every tool the model actually invoked, by name.
            tool_uses = []
            for e in events:
                if e.get("type") == "assistant":
                    for blk in (e.get("message", {}).get("content") or []):
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            tool_uses.append({"name": blk.get("name"), "input": blk.get("input")})
            meta = {k: result.get(k) for k in
                    ("duration_ms", "num_turns", "stop_reason", "session_id", "total_cost_usd")}
            meta["tool_uses"] = tool_uses
            meta["permission_denials"] = result.get("permission_denials") or []
            # events = the COMPLETE verbatim CLI event stream for this call
            # (system init, every assistant/thinking/tool_use block, every
            # tool_result, the final result). Returned so the trace can log it
            # in full — nothing the model did is dropped.
            return result.get("result", ""), result.get("total_cost_usd", 0.0), \
                result.get("usage", {}), result.get("session_id"), meta, events
        except Exception as e:
            last = e
            time.sleep(3 + 3 * i)
    raise RuntimeError(f"claude call failed after {tries}: {last}")


import urllib.request  # noqa: E402


def report_llm_usage(session_id, usage, message_id):
    """Feed a model call's REAL token usage to the daemon so its dollar-mode
    budget tracker reflects true accumulated spend. Without this, spent_usd stays
    $0 and the budget injection exerts no pressure. Maps the claude-CLI usage
    shape onto the daemon's expected field names; message_id dedups retries."""
    u = usage or {}
    cc = u.get("cache_creation", {}) or {}
    body = {
        "session_id": session_id,
        "model": PRICED_MODEL,
        "message_id": message_id,
        "usage": {
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "cache_read_tokens": u.get("cache_read_input_tokens", 0),
            "cache_creation_tokens": u.get("cache_creation_input_tokens", 0),
            "cache_creation_1h_tokens": cc.get("ephemeral_1h_input_tokens", 0),
        },
    }
    daemon_post("/llm/usage", body)


def daemon_post(path, body):
    """Best-effort daemon call; returns additionalContext string or None. The
    daemon is under test but must never crash the experiment."""
    try:
        req = urllib.request.Request(
            DAEMON + path, json.dumps(body).encode(),
            {"content-type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=10))
        return d.get("additionalContext")
    except Exception:
        return None


# --- grading (SQuAD/HotpotQA normalization) ---

def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score(pred, gold):
    np, ng = _norm(pred), _norm(gold)
    em = float(np == ng)
    pt, gt = np.split(), ng.split()
    if not pt or not gt:
        return em, float(np == ng)
    common = {}
    for w in pt:
        if w in gt:
            common[w] = min(pt.count(w), gt.count(w))
    nsame = sum(common.values())
    if nsame == 0:
        return em, 0.0
    prec, rec = nsame / len(pt), nsame / len(gt)
    return em, 2 * prec * rec / (prec + rec)


BASE_SYS = (
    "You are a research assistant answering a multi-hop question. The answer is "
    "found by combining facts from MULTIPLE passages in an offline corpus, which "
    "you access with two tools. On each turn output EXACTLY ONE line, one of:\n"
    "  SEARCH: <query>       (BM25 search, returns up to 5 {title, snippet})\n"
    "  READ: <exact title>   (returns the full passage text)\n"
    "  ANSWER: <short answer> (a name, date, or phrase — only when confident)\n"
    "Output nothing else — no reasoning, no extra lines. Once you can answer, "
    "give ANSWER and stop."
)


def build_prompt(question, transcript, tracker):
    parts = [BASE_SYS, "", f"QUESTION: {question}", ""]
    if tracker:
        parts += [tracker, ""]
    if transcript:
        parts.append("TRANSCRIPT SO FAR:")
        parts += transcript
        parts.append("")
    parts.append("Your next single line:")
    return "\n".join(parts)


ACT_RE = re.compile(r"\b(SEARCH|READ|ANSWER)\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


# A model turn that carries none of the ReAct verbs but IS structured markup
# (e.g. a planning checklist leaking in) must never be accepted as a final answer.
_NONANSWER_RE = re.compile(r"<\s*(checklist|item\b)", re.IGNORECASE)


def parse_action(text):
    t = (text or "").strip()
    m = ACT_RE.search(t)
    if not m:
        # No explicit verb. Only treat as an answer if it's plausibly a short
        # answer, not markup/empty. Otherwise signal a malformed turn so the loop
        # can re-prompt instead of scoring garbage.
        if not t or _NONANSWER_RE.search(t):
            return "RETRY", ""
        return "ANSWER", t
    verb = m.group(1).upper()
    arg = m.group(2).strip().splitlines()[0].strip()  # first line only
    if verb == "ANSWER" and (not arg or _NONANSWER_RE.search(arg)):
        return "RETRY", ""
    return verb, arg


def parse_tracker(tracker):
    """Parse the daemon's injected budget text into structured numbers so a run
    is machine-reasonable (remaining calls, used, tier). None when injection off."""
    if not tracker:
        return None
    out = {}
    # Budget is money: "LLM cost used: $X, remaining (of session estimate): $Y"
    m = re.search(r"LLM cost used:\s*\$([\d.]+),\s*remaining[^$]*\$([\d.]+)", tracker)
    if m:
        out["spent_usd"], out["remaining_usd"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Tool calls used:\s*(\d+)", tracker)
    if m:
        out["tool_calls_used"] = int(m.group(1))
    m = re.search(r"Tier:\s*(\w+)", tracker)
    if m:
        out["tier"] = m.group(1)
    return out or None


def run_question(q, run_id, tag, traces_dir):
    sid = f"{run_id}-{q['id']}"
    trace_path = os.path.join(traces_dir, f"{q['id']}.jsonl")
    tf = open(trace_path, "w")

    def trace(kind, **kw):
        tf.write(json.dumps({"kind": kind, "wall": time.time(), **kw}) + "\n")
        tf.flush()

    transcript = []
    trace("session", session_id=sid, question=q["question"], gold=q["answer"],
          gold_titles=q.get("gold_titles"))

    # Register the session with the daemon (for logging + budget accounting) but
    # do NOT feed its plan-seed into the model prompt. The seed is a *coding-agent*
    # planning template ("output your decomposition, nothing else in this block");
    # a QA ReAct model obeys it and emits the checklist INSTEAD of playing the
    # SEARCH/READ/ANSWER protocol, hijacking the run. This experiment isolates the
    # variable under test — the money budget tracker — so the seed is recorded in
    # the trace for provenance but kept out of the model's context. (Testing the
    # plan-seed feature itself is a separate experiment with a QA-appropriate seed.)
    plan = daemon_post("/session/start", {
        "session_id": sid, "cli": "qa-exp", "task": q["question"], "model": PRICED_MODEL})
    trace("session_start", plan_seed=plan, fed_to_model=False)

    tool_calls_used = 0
    cost = 0.0
    out_tok = 0
    hit_cap = False
    answer = "(no answer)"
    # cheat-audit accumulators across all model calls for this question
    all_tool_uses = []          # every real tool the CLI invoked, by name
    all_denials = []            # every blocked tool attempt
    web_reqs = 0                # Anthropic server-side web_search + web_fetch

    for step in range(MAX_STEPS):
        tracker = daemon_post("/tool/pre", {"session_id": sid, "tool_name": "search"})
        prompt = build_prompt(q["question"], transcript, tracker)
        text, c, usage, cli_sid, meta, events = claude_call(prompt)
        cost += c
        out_tok += (usage or {}).get("output_tokens", 0)
        # COMPLETE verbatim CLI event stream for this call — logged in full so
        # every action claude took (thinking, tool_use, tool_result) is on record.
        trace("cli_raw", step=step, events=events)
        all_tool_uses += meta.get("tool_uses") or []
        all_denials += meta.get("permission_denials") or []
        stu = (usage or {}).get("server_tool_use", {}) or {}
        web_reqs += stu.get("web_search_requests", 0) + stu.get("web_fetch_requests", 0)
        # Report real spend to the daemon BEFORE the next /tool/pre so the next
        # step's injected budget reflects money already spent this session.
        report_llm_usage(sid, usage, message_id=f"{sid}-s{step}")
        verb, arg = parse_action(text)

        # Full per-step record: exactly what was injected, sent, and returned.
        # sim_budget = the daemon's advisory state parsed from the injected text,
        # so a run can be reasoned about numerically (remaining/tier) without
        # re-parsing prose later.
        trace("step", step=step, injected_tracker=tracker,
              sim_budget=parse_tracker(tracker), prompt=prompt,
              model_raw=text, action=verb, arg=arg, usage=usage, cost_usd=c,
              cli_session_id=cli_sid, cli_meta=meta,
              # explicit cheat-audit fields: which real tools the CLI invoked
              cli_tool_uses=meta.get("tool_uses"),
              cli_permission_denials=meta.get("permission_denials"),
              web_search_requests=(usage or {}).get("server_tool_use", {}).get("web_search_requests", 0),
              web_fetch_requests=(usage or {}).get("server_tool_use", {}).get("web_fetch_requests", 0),
              tool_calls_used=tool_calls_used)

        if verb == "ANSWER":
            answer = arg
            break

        if verb == "RETRY":
            # Malformed turn (empty or leaked markup, not a real action). Nudge
            # the model back onto the protocol; does not consume a tool call.
            transcript.append("[SYSTEM] Invalid output. Reply with exactly one "
                              "line: SEARCH:/READ:/ANSWER:.")
            trace("retry", step=step, reason="nonanswer_or_empty")
            continue

        if tool_calls_used >= MAX_TOOLS:
            hit_cap = True
            answer = "(hit tool cap, no answer)"
            break
        tool_calls_used += 1

        if verb == "SEARCH":
            res = tool_search(arg)
            obs = json.dumps(res)
        else:  # READ
            obs = tool_read(arg)
        transcript.append(f"> {verb}: {arg}")
        transcript.append(f"OBSERVATION: {obs[:1200]}")
        trace("observation", step=step, verb=verb, arg=arg, observation=obs)

        post = daemon_post("/tool/post", {
            "session_id": sid, "tool_name": verb.lower(), "tool_input": {"arg": arg},
            "tool_result": obs[:500]})
        trace("tool_post", step=step, daemon_context=post)
    else:
        answer = "(max steps, no answer)"

    em, f1 = score(answer, q["answer"])
    # cheat audit: the model is CLEAN iff it invoked no real CLI tools, no tool
    # attempt was denied, and no server-side web request fired. Any nonzero here
    # means it may have reached outside our offline corpus.
    tool_use_names = [t.get("name") for t in all_tool_uses]
    clean = not tool_use_names and not all_denials and web_reqs == 0
    row = {"id": q["id"], "session_id": sid, "question": q["question"],
           "gold": q["answer"], "answer": answer, "em": em, "f1": round(f1, 3),
           "tool_calls": tool_calls_used, "cost_usd": round(cost, 4),
           "out_tok": out_tok, "hit_cap": hit_cap,
           "audit_clean": clean, "cli_tool_uses": tool_use_names,
           "cli_denials": len(all_denials), "web_requests": web_reqs,
           "trace": f"traces/{q['id']}.jsonl"}
    trace("result", **row)
    tf.close()
    return row


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def read_daemon_condition():
    try:
        c = json.load(open(CONFIG_PATH))
        return {"inject_enabled": c.get("inject_enabled"),
                "session_budget_estimate_usd": c.get("session_budget_estimate_usd")}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    questions = json.load(open(os.path.join(HERE, "data", "questions.json")))
    questions = questions[args.start:args.start + args.n]

    # Unique, sortable, human-readable run id.
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.tag}-{stamp}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    condition = read_daemon_condition()
    meta = {"run_id": run_id, "tag": args.tag, "model": MODEL,
            "gateway": "claude-cli",
            "dataset": "HotpotQA (distractor setting, offline subset)",
            "dataset_files": {"questions": "data/questions.json",
                              "corpus": "data/corpus.json",
                              "n_corpus_passages": len(_corpus)},
            "condition": condition,
            "git_sha": git_sha(), "args": vars(args),
            "max_tools": MAX_TOOLS, "max_steps": MAX_STEPS,
            "questions": [q["id"] for q in questions],
            "started": stamp}
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=2)
    if os.path.exists(CONFIG_PATH):
        shutil.copy(CONFIG_PATH, os.path.join(run_dir, "config_snapshot.json"))
    print(f"[{run_id}] condition={condition} git={meta['git_sha'][:8] if meta['git_sha'] else '?'}",
          flush=True)

    results_path = os.path.join(run_dir, "results.jsonl")
    rows = []
    with open(results_path, "w") as f:
        for q in questions:
            try:
                r = run_question(q, run_id, args.tag, traces_dir)
            except Exception as e:
                r = {"id": q["id"], "session_id": f"{run_id}-{q['id']}",
                     "question": q["question"], "gold": q["answer"],
                     "answer": f"(ERROR: {str(e)[:80]})", "em": 0.0, "f1": 0.0,
                     "tool_calls": 0, "cost_usd": 0.0, "out_tok": 0,
                     "hit_cap": False, "failed": True}
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{run_id}] {r['id']} calls={r['tool_calls']} "
                  f"f1={r['f1']} em={r['em']} cap={r['hit_cap']} "
                  f"clean={r.get('audit_clean', '?')} web={r.get('web_requests', 0)} "
                  f"fail={r.get('failed', False)} ans={r['answer'][:40]!r}", flush=True)

    ok = [r for r in rows if not r.get("failed")]
    n = len(ok)
    summary = {"run_id": run_id, "n_ok": n, "n_total": len(rows),
               "condition": condition,
               "mean_tool_calls": round(sum(r["tool_calls"] for r in ok) / n, 3) if n else None,
               "mean_out_tok": round(sum(r["out_tok"] for r in ok) / n, 1) if n else None,
               "mean_f1": round(sum(r["f1"] for r in ok) / n, 3) if n else None,
               "mean_em": round(sum(r["em"] for r in ok) / n, 3) if n else None,
               "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
               "hit_cap": sum(bool(r["hit_cap"]) for r in ok),
               # cheat audit rollup: how many questions were clean, and every
               # tool/web signal seen across the run (empty = nobody cheated).
               "audit_all_clean": all(r.get("audit_clean", True) for r in ok),
               "audit_dirty_ids": [r["id"] for r in ok if not r.get("audit_clean", True)],
               "audit_tool_uses": sorted({n for r in ok for n in (r.get("cli_tool_uses") or [])}),
               "audit_total_web_requests": sum(r.get("web_requests", 0) for r in ok),
               "audit_total_denials": sum(r.get("cli_denials", 0) for r in ok),
               "finished": datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)

    # convenience pointer for analyze.py: latest run of this tag
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    shutil.copy(results_path, os.path.join(HERE, "results", f"{args.tag}.jsonl"))
    with open(os.path.join(HERE, "results", f"{args.tag}.latest"), "w") as f:
        f.write(run_id + "\n")

    print(f"\n=== {run_id} summary ===")
    for k in ("mean_tool_calls", "mean_out_tok", "mean_f1", "mean_em",
              "total_cost_usd", "hit_cap"):
        print(f"{k:16} {summary[k]}")
    print(f"{'audit_clean':16} {summary['audit_all_clean']} "
          f"(tool_uses={summary['audit_tool_uses']}, "
          f"web={summary['audit_total_web_requests']}, "
          f"denials={summary['audit_total_denials']}, "
          f"dirty={summary['audit_dirty_ids']})")
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
