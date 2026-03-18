"""Parse org YAML config into data models."""

import yaml
from pathlib import Path

from models import Agent, OrgConfig


def load_config(path: str | Path) -> OrgConfig:
    """Load and parse an org YAML config file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    root = _parse_agent(data["agents"], level=0)
    return OrgConfig(
        name=data.get("org", "unnamed"),
        description=data.get("description", ""),
        root=root,
        coordination=data.get("coordination", "fan-out"),
    )


def _parse_agent(agents_dict: dict, level: int, parent: Agent | None = None) -> Agent:
    """Recursively parse agent hierarchy from YAML."""
    # There should be exactly one key at each level (the agent name)
    if len(agents_dict) != 1 and level == 0:
        # Root level: first key is root agent
        pass

    for name, spec in agents_dict.items():
        if isinstance(spec, str):
            spec = {"role": spec, "persona": ""}

        agent = Agent(
            name=name,
            role=spec.get("role", name),
            persona=spec.get("persona", ""),
            level=level,
            parent=parent,
        )

        children_dict = spec.get("children", {})
        for child_name, child_spec in children_dict.items():
            child = _parse_agent(
                {child_name: child_spec}, level=level + 1, parent=agent
            )
            agent.children.append(child)

        return agent
