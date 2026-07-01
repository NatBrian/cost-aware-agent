// cost-aware-agent OpenCode adapter. Installed by `cost-aware-agent install --for opencode`.
//
// Corrected against the real @opencode-ai/plugin@1.17.12 Hooks type (verified by unpacking
// the npm tarball, not the docs) — several hooks the original design sketch assumed do not
// exist as written:
//   - "tool.execute.before" output is `{ args: any }` only — there is NO text/context field.
//     It can rewrite tool arguments, not inject advisory text. Not used here.
//   - "message.updated" and "session.idle" are not top-level hooks. They are members of the
//     `Event` union, delivered through the single generic `event` hook. That hook's signature
//     is `(input: { event: Event }) => Promise<void>` — no `output` param, so it is
//     notify-only and cannot inject anything back into the model.
// The only hooks that can hand text back to OpenCode are:
//   - "experimental.chat.system.transform": output.system: string[] — runs before every LLM
//     call, so it doubles as the Budget Tracker injection point (higher frequency than Claude
//     Code's PreToolUse-only tracker, since OpenCode has no equivalent hook per tool call).
//   - "tool.execute.after": output.output: string — the tool result text handed to the model.
//     Self-Verification prompt is appended here on milestone tool calls, since there's no
//     other channel to reach the model at that point in the lifecycle.

import type { Plugin } from "@opencode-ai/plugin";

const DAEMON = "http://127.0.0.1:7331";
const seenSessions = new Set<string>();

async function post(path: string, body: unknown): Promise<any> {
  try {
    const res = await fetch(`${DAEMON}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null; // fail-open — daemon down must never block the agent
  }
}

export const CostAwareAgentPlugin: Plugin = async ({ $ }) => {
  async function ensureDaemon(): Promise<void> {
    try {
      const res = await fetch(`${DAEMON}/health`, { signal: AbortSignal.timeout(1000) });
      if (res.ok) return;
    } catch {
      // fall through to spawn attempt
    }
    try {
      await $`cost-aware-agent daemon start`.quiet().nothrow();
      await new Promise((r) => setTimeout(r, 500));
    } catch {
      // spawn failed too — fail open, every post() call below already tolerates a dead daemon
    }
  }

  return {
    "experimental.chat.system.transform": async (input, output) => {
      await ensureDaemon();
      const sessionID = input.sessionID ?? "";
      if (!sessionID) return;
      if (!seenSessions.has(sessionID)) {
        seenSessions.add(sessionID);
        const res = await post("/session/start", {
          session_id: sessionID,
          cli: "opencode",
          model: input.model?.modelID ?? "",
        });
        if (res?.additionalContext) output.system.push(res.additionalContext);
        return;
      }
      const res = await post("/tool/pre", { session_id: sessionID, tool_name: "" });
      if (res?.additionalContext) output.system.push(res.additionalContext);
    },

    "tool.execute.after": async (input, output) => {
      const res = await post("/tool/post", {
        session_id: input.sessionID,
        tool_name: input.tool,
        tool_input: input.args ?? {},
        tool_result: String(output.output ?? ""),
      });
      if (res?.additionalContext) {
        output.output = `${output.output}\n\n[cost-aware-agent]\n${res.additionalContext}`;
      }
    },

    event: async ({ event }) => {
      if (event.type === "message.updated" && event.properties.info.role === "assistant") {
        const info = event.properties.info as any;
        const t = info.tokens ?? { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } };
        await post("/llm/usage", {
          session_id: info.sessionID,
          model: info.modelID ?? "",
          usage: {
            input_tokens: t.input ?? 0,
            output_tokens: (t.output ?? 0) + (t.reasoning ?? 0), // reasoning bills as output, same as Claude Code
            cache_read_tokens: t.cache?.read ?? 0,
            cache_creation_tokens: t.cache?.write ?? 0,
            // no 1h/5m split exposed here (AssistantMessage.tokens.cache is a flat {read, write}) —
            // daemon falls back to pricing the whole write amount at the 5m rate, a known,
            // documented undercount, not a bug in this adapter.
          },
        });
        return;
      }
      if (event.type === "message.part.updated") {
        const part = event.properties.part as any;
        if (part.type === "text" && typeof part.text === "string" && part.text.includes("<verification>")) {
          await post("/verification/result", {
            session_id: part.sessionID,
            raw_response: part.text,
          });
        }
        return;
      }
      if (event.type === "session.idle") {
        await post("/session/stop", { session_id: event.properties.sessionID });
      }
    },
  };
};

export default CostAwareAgentPlugin;
