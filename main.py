"""
Multi-agent org orchestrator using LangGraph.

This is the LangGraph version of multi-agent-org. Instead of spawning
`claude -p` subprocesses in separate directories, each agent is a
LangGraph node with its own persona, tools, and system prompt.

The orchestration graph replaces the manual orchestration loops:
- Fan-out: children execute in sequence, parent synthesizes
- Delegated: parent decomposes → children execute → parent synthesizes

Usage:
    python main.py <config.yaml> "Your task here"
    python main.py examples/review-org.yaml "Review the authentication module"
"""

import os
import sys

from config import load_config
from orchestrator import build_graph


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <config.yaml> <task>")
        print()
        print("Example:")
        print('  python main.py examples/review-org.yaml "Review the auth module"')
        sys.exit(1)

    config_path = sys.argv[1]
    task = " ".join(sys.argv[2:])

    # Validate environment
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("Error: set GOOGLE_CLOUD_PROJECT environment variable")
        sys.exit(1)

    # Load config
    config = load_config(config_path)
    print(f"Organization: {config.name}")
    print(f"Coordination: {config.coordination}")
    print(f"Root agent: {config.root.name} ({config.root.role})")
    if config.root.children:
        children = ", ".join(f"{c.name}" for c in config.root.children)
        print(f"Children: {children}")
    print()

    # Build and compile the graph
    graph = build_graph(config)
    app = graph.compile()

    # Langfuse tracing (optional)
    langfuse_config = {}
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        from langfuse.langchain import CallbackHandler
        langfuse_config["callbacks"] = [CallbackHandler()]
        print("(Langfuse tracing enabled)")

    # Run
    print(f"Task: {task}")
    print("=" * 60)
    print()

    initial_state = {
        "task": task,
        "context": "",
        "results": {},
        "final_output": "",
    }

    result = app.invoke(initial_state, {**langfuse_config, "recursion_limit": 50})

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for agent_name, output in result.get("results", {}).items():
        print(f"\n--- {agent_name} ---")
        if output:
            print(output[:500])
            if len(output) > 500:
                print(f"... ({len(output)} chars total)")
        print()

    if result.get("final_output"):
        print("=" * 60)
        print("FINAL OUTPUT")
        print("=" * 60)
        print(result["final_output"])


if __name__ == "__main__":
    main()
