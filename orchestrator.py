"""LangGraph-based multi-agent orchestrator.

Replaces the subprocess-based orchestrator from multi-agent-org.
Instead of spawning `claude -p` subprocesses, each agent is a LangGraph
node with its own persona, tools, and system prompt. The orchestration
graph handles fan-out to children, synthesis by parents, and delegation.

Key LangGraph concepts:
- Each agent is a node in the graph (not a subprocess)
- State flows through the graph as messages
- Fan-out: parallel branches for each child agent
- Synthesis: parent node receives all child outputs
- Delegation: parent decomposes, assigns, children execute, parent synthesizes
"""

import os
from typing import TypedDict

from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langgraph.graph import END, StateGraph

from models import Agent, OrgConfig
from tools import ALL_TOOLS


class AgentState(TypedDict):
    """State that flows through the orchestration graph."""
    task: str
    context: str
    results: dict[str, str]  # agent_name -> output
    final_output: str


def _make_model(tools: list | None = None):
    """Create a ChatAnthropicVertex model, optionally with tools bound."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "us-east5")
    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    model = ChatAnthropicVertex(
        model_name=model_name,
        project=project,
        location=region,
    )

    if tools:
        model = model.bind_tools(tools)

    return model


def _build_system_prompt(agent: Agent) -> str:
    """Build a system prompt from an agent's persona and role."""
    parts = [
        f"You are {agent.name}, a {agent.role}.",
    ]
    if agent.persona:
        parts.append(agent.persona)

    if agent.children:
        child_names = ", ".join(f"{c.name} ({c.role})" for c in agent.children)
        parts.append(f"Your direct reports are: {child_names}.")

    if agent.parent:
        parts.append(f"You report to {agent.parent.name} ({agent.parent.role}).")

    parts.append(
        "You have tools available to read files, write files, run commands, "
        "and manage beliefs (rms tools). Use them when needed."
    )

    return "\n\n".join(parts)


def _run_agent_node(agent: Agent, state: AgentState) -> dict:
    """Run a single agent as a LangGraph node.

    The agent receives the task + any context, runs the tool loop
    via LangGraph's built-in ToolNode pattern, and returns its output.
    """
    from langgraph.graph import MessagesState, StateGraph as InnerGraph
    from langgraph.prebuilt import ToolNode

    model = _make_model(ALL_TOOLS)
    system_prompt = _build_system_prompt(agent)

    # Build the inner agent graph (same pattern as claude_code_langgraph)
    def agent_node(inner_state: MessagesState):
        response = model.invoke(inner_state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    def should_continue(inner_state: MessagesState):
        last = inner_state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END

    inner = InnerGraph(MessagesState)
    inner.add_node("agent", agent_node)
    inner.add_node("tools", tool_node)
    inner.set_entry_point("agent")
    inner.add_conditional_edges("agent", should_continue)
    inner.add_edge("tools", "agent")
    app = inner.compile()

    # Build the prompt
    task = state["task"]
    context = state.get("context", "")
    prompt = f"## Your Task\n\n{task}"
    if context:
        prompt = f"## Context\n\n{context}\n\n---\n\n{prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    result = app.invoke({"messages": messages}, {"recursion_limit": 25})

    # Extract the final text response
    output = ""
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            output = msg.content
            break

    return {"results": {**state.get("results", {}), agent.name: output}}


def build_fan_out_graph(config: OrgConfig) -> StateGraph:
    """Build a fan-out orchestration graph.

    Leaves execute the task, parent synthesizes results.

    Graph structure:
      start -> [child_1, child_2, ...] -> synthesize -> end
    """
    root = config.root
    graph = StateGraph(AgentState)

    if root.is_leaf:
        # Single agent, no children
        def solo_node(state: AgentState) -> dict:
            result = _run_agent_node(root, state)
            return {**result, "final_output": result["results"][root.name]}

        graph.add_node("solo", solo_node)
        graph.set_entry_point("solo")
        graph.add_edge("solo", END)
        return graph

    # Add child nodes
    for child in root.children:
        def make_child_node(agent=child):
            def node(state: AgentState) -> dict:
                return _run_agent_node(agent, state)
            return node

        graph.add_node(child.name, make_child_node())

    # Add synthesis node
    def synthesize(state: AgentState) -> dict:
        results = state.get("results", {})

        # Build synthesis context from child results
        context_parts = []
        for child in root.children:
            if child.name in results:
                context_parts.append(
                    f"## Report from {child.name} ({child.role})\n\n{results[child.name]}"
                )
        synthesis_context = "\n\n---\n\n".join(context_parts)

        synthesis_state = {
            "task": (
                f"Your direct reports have completed their analysis. "
                f"Synthesize the following reports into a unified result.\n\n"
                f"Original task: {state['task']}"
            ),
            "context": synthesis_context,
            "results": results,
        }

        result = _run_agent_node(root, synthesis_state)
        return {
            **result,
            "final_output": result["results"][root.name],
        }

    graph.add_node("synthesize", synthesize)

    # Fan-out: start -> all children in parallel
    # LangGraph doesn't have native parallel fan-out, so we chain them.
    # For true parallelism, each child would be a separate branch.
    # Sequential is simpler and correct — parallel is an optimization.
    first_child = root.children[0]
    graph.set_entry_point(first_child.name)

    for i, child in enumerate(root.children[:-1]):
        next_child = root.children[i + 1]
        graph.add_edge(child.name, next_child.name)

    last_child = root.children[-1]
    graph.add_edge(last_child.name, "synthesize")
    graph.add_edge("synthesize", END)

    return graph


def build_delegated_graph(config: OrgConfig) -> StateGraph:
    """Build a delegated orchestration graph.

    Phase 1: Parent decomposes task and assigns subtasks
    Phase 2: Children execute their assigned subtasks
    Phase 3: Parent synthesizes results

    Graph structure:
      start -> decompose -> [child_1, child_2, ...] -> synthesize -> end
    """
    root = config.root
    graph = StateGraph(AgentState)

    # Phase 1: Decompose
    def decompose(state: AgentState) -> dict:
        child_info = []
        for child in root.children:
            info = f"- **{child.name}** ({child.role})"
            if child.persona:
                first_sentence = child.persona.strip().split(".")[0].strip()
                info += f": {first_sentence}"
            child_info.append(info)

        decompose_state = {
            "task": (
                f"You have received the following task:\n\n{state['task']}\n\n"
                f"## Your Job\n\n"
                f"Decompose this task into subtasks and assign each to the "
                f"best-fit direct report.\n\n"
                f"Your direct reports:\n" + "\n".join(child_info) + "\n\n"
                f"Format each assignment as:\n"
                f"### ASSIGN: agent-name\n"
                f"<subtask description>\n"
            ),
            "context": state.get("context", ""),
            "results": {},
        }

        result = _run_agent_node(root, decompose_state)
        # Store decomposition output for children to parse
        return {
            "results": result["results"],
            "context": result["results"].get(root.name, ""),
            "task": state["task"],
        }

    graph.add_node("decompose", decompose)

    # Phase 2: Children execute (sequentially for simplicity)
    for child in root.children:
        def make_child_node(agent=child):
            def node(state: AgentState) -> dict:
                # Extract this child's assignment from decomposition output
                decomp_output = state.get("context", "")
                import re
                pattern = rf"###\s*ASSIGN:\s*{re.escape(agent.name)}\s*\n(.*?)(?=###\s*ASSIGN:|$)"
                match = re.search(pattern, decomp_output, re.DOTALL | re.IGNORECASE)
                subtask = match.group(1).strip() if match else state["task"]

                child_state = {
                    "task": subtask,
                    "context": "",
                    "results": state.get("results", {}),
                }
                return _run_agent_node(agent, child_state)
            return node

        graph.add_node(child.name, make_child_node())

    # Phase 3: Synthesize (same as fan-out)
    def synthesize(state: AgentState) -> dict:
        results = state.get("results", {})
        context_parts = []
        for child in root.children:
            if child.name in results:
                context_parts.append(
                    f"## Report from {child.name} ({child.role})\n\n{results[child.name]}"
                )
        synthesis_context = "\n\n---\n\n".join(context_parts)

        synthesis_state = {
            "task": (
                f"Your direct reports have completed their assigned subtasks. "
                f"Synthesize the following reports into a unified result.\n\n"
                f"Original task: {state['task']}"
            ),
            "context": synthesis_context,
            "results": results,
        }
        result = _run_agent_node(root, synthesis_state)
        return {**result, "final_output": result["results"][root.name]}

    graph.add_node("synthesize", synthesize)

    # Wire: decompose -> children -> synthesize
    graph.set_entry_point("decompose")
    first_child = root.children[0]
    graph.add_edge("decompose", first_child.name)

    for i, child in enumerate(root.children[:-1]):
        next_child = root.children[i + 1]
        graph.add_edge(child.name, next_child.name)

    last_child = root.children[-1]
    graph.add_edge(last_child.name, "synthesize")
    graph.add_edge("synthesize", END)

    return graph


def build_graph(config: OrgConfig) -> StateGraph:
    """Build the orchestration graph based on coordination mode."""
    if config.coordination == "delegated":
        return build_delegated_graph(config)
    return build_fan_out_graph(config)
