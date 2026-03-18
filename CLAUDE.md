# CLAUDE.md

## Project overview

LangGraph version of multi-agent-org. Same concept — hierarchical agent teams with distinct roles — but instead of spawning `claude -p` subprocesses, each agent is a LangGraph node with its own persona, tools, and system prompt.

The original `../multi-agent-org` creates directory structures and runs agents as subprocesses. This version runs everything in a single Python process using LangGraph's StateGraph for orchestration.

## Architecture

- `main.py` — Entry point. Loads YAML config, builds the LangGraph orchestration graph, runs it.
- `orchestrator.py` — Builds the LangGraph graph. Fan-out (children → parent synthesis) and delegated (decompose → assign → execute → synthesize) modes.
- `tools.py` — Agent tools: base tools (read, write, edit, grep, glob, run_command) + RMS tools (add, retract, challenge, defend, explain, etc.).
- `config.py` — YAML parsing into data models.
- `models.py` — Data classes: Agent, OrgConfig.

## Key differences from multi-agent-org

| Aspect | multi-agent-org | This version |
|--------|----------------|--------------|
| Agent execution | `claude -p` subprocess | LangGraph node (in-process) |
| State | Filesystem (.tasks/) | StateGraph state dict |
| Orchestration | Python loops in orchestrator.py | LangGraph edges and conditional routing |
| Tools | Via `--allowedTools` flag | Bound directly to model via `bind_tools()` |
| Beliefs | `beliefs` CLI subprocess | RMS tools called directly via `rms_lib.api` |
| Observability | None (or git log) | Langfuse callbacks |

## Coordination modes

- **fan-out** (default): All children receive the same task, execute sequentially, parent synthesizes.
- **delegated**: Parent decomposes task → assigns subtasks to children → children execute → parent synthesizes.

## Environment variables

- `GOOGLE_CLOUD_PROJECT` — Required. GCP project ID.
- `GOOGLE_CLOUD_REGION` — Optional. Vertex AI region (default: us-east5).
- `ANTHROPIC_MODEL` — Optional. Model name (default: claude-sonnet-4-20250514).
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` — Optional. Enables tracing.

## Commands

```bash
# Run with a config and task
python main.py examples/review-org.yaml "Review the authentication module"
python main.py examples/research-org.yaml "What are the failure modes of LLM belief tracking?"

# Install deps
pip install -r requirements.txt
```

## RMS integration

All agents have access to RMS tools (if rms is installed):
- `rms_add` / `rms_retract` / `rms_assert` — manage beliefs
- `rms_explain` / `rms_trace` — debug justification chains
- `rms_challenge` / `rms_defend` — dialectical argumentation
- `rms_compact` — token-budgeted summaries
- `rms_nogood` — record contradictions with automatic backtracking

Agents can use these to track findings, challenge each other's claims, and maintain a shared belief network.
