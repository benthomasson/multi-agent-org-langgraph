# multi-agent-org-langgraph

LangGraph version of [multi-agent-org](https://github.com/benthomasson/multi-agent-org). Hierarchical agent teams running as LangGraph nodes instead of `claude -p` subprocesses.

## Quick Start

```bash
pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT=your-project
python main.py examples/review-org.yaml "Review the authentication module"
```

## How It Works

Define your agent team in YAML:

```yaml
org: code-review
coordination: fan-out

agents:
  lead-reviewer:
    role: Lead Code Reviewer
    persona: You synthesize feedback from specialist reviewers.
    children:
      security-reviewer:
        role: Security Specialist
        persona: You review code for security vulnerabilities.
      performance-reviewer:
        role: Performance Specialist
        persona: You review code for performance bottlenecks.
```

Run it:

```bash
python main.py config.yaml "Review the auth module for security and performance"
```

The orchestrator builds a LangGraph StateGraph where:
1. Each child agent is a node with its own persona and tools
2. Children execute the task (fan-out) or assigned subtasks (delegated)
3. The parent agent synthesizes all child outputs into a final result

## Coordination Modes

**Fan-out** — All children receive the same task, parent synthesizes:
```
task → [security-reviewer, performance-reviewer] → lead-reviewer → result
```

**Delegated** — Parent decomposes task, assigns subtasks:
```
task → lead-reviewer (decompose) → [security: subtask-1, perf: subtask-2] → lead-reviewer (synthesize) → result
```

## RMS Integration

All agents have access to [RMS](https://github.com/benthomasson/rms) tools for belief tracking:

```
rms_add, rms_retract, rms_assert, rms_explain, rms_search,
rms_trace, rms_challenge, rms_defend, rms_nogood, rms_compact
```

Agents can track findings as beliefs, challenge each other's claims, and maintain a shared belief network with automatic retraction cascades.

## vs multi-agent-org

| | multi-agent-org | This version |
|---|---|---|
| Agent execution | `claude -p` subprocess per agent | LangGraph node (in-process) |
| Communication | Filesystem (`.tasks/` directory) | StateGraph state dict |
| Orchestration | Python loops | LangGraph edges |
| Observability | Git log | Langfuse tracing |
| Beliefs | `beliefs` CLI subprocess | RMS tools (direct API) |
