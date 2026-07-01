# cost-aware-agent

A background daemon + hook adapters that give CLI coding agents (Claude Code, OpenCode)
real-time visibility into how much a session is actually costing, so the agent can factor
cost into its own decisions instead of running blind.

**Advisory only.** The daemon never blocks a tool call, never aborts a session, and never
sets a hard budget ceiling. It computes real dollar cost from actual token/tool-call usage
and injects that information into the model's context. What the agent does with it — keep
going, wrap up, try a cheaper approach — is entirely the agent's call.

## Why

Coding agents today have no sense of what they're spending. A five-minute task and a
five-dollar detour look identical from inside the conversation. This gives the agent the
missing half of the trade-off: not just "can I do this" but "is this still worth it."

## How it works

A local FastAPI + SQLite daemon (`~/.cost-aware-agent/`) tracks cost per session:

- **Cost Engine** — real `$` from token counts and model pricing (vendored LiteLLM price
  map), not synthetic units. Handles Anthropic's split pricing for 5-minute vs 1-hour
  prompt-cache writes.
- **Budget Tracker** — injects current spend, tier (HIGH/MEDIUM/LOW/CRITICAL), and
  guidance text before tool calls / LLM turns.
- **Planning** — asks the agent to decompose the task into an exploration/verification
  checklist up front, tracked for the rest of the session.
- **Self-Verification** — on milestone actions (edits, test runs), asks the agent to check
  its checklist against the budget and decide: finish up, keep going, or change approach.

Two adapters, two capture mechanisms:

| | Claude Code | OpenCode |
|---|---|---|
| Mechanism | pull — daemon parses the session transcript on hook fire | push — plugin posts usage straight from hook payloads |
| Install | `cost-aware-agent install --for claude-code [--project-dir .\|--global]` | `cost-aware-agent install --for opencode [--project-dir .\|--global]` |

The daemon auto-spawns on first hook fire if it isn't already running, and fails open —
if it's ever unreachable, adapters silently no-op rather than blocking the agent.

## Install

```bash
pip install -e .
cost-aware-agent install --for claude-code --project-dir .
# or: cost-aware-agent install --for opencode --project-dir .
```

## Status

- Claude Code adapter — built, live-verified against real sessions.
- OpenCode adapter — built against the real `@opencode-ai/plugin` API, not yet run against
  a live OpenCode session.
- Known open gap: the model reliably reads the injected Planning/Self-Verification prompts
  but doesn't yet reliably reply in the expected format — under active investigation.
