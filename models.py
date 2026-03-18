"""Data models for multi-agent org — same as multi-agent-org but standalone."""

from dataclasses import dataclass, field


@dataclass
class Agent:
    """An agent in the hierarchy."""

    name: str
    role: str
    persona: str
    level: int  # 0=root, 1=supervisor, 2=expert
    children: list["Agent"] = field(default_factory=list)
    parent: "Agent | None" = field(default=None, repr=False)

    @property
    def level_name(self) -> str:
        names = {0: "root", 1: "supervisor", 2: "expert"}
        return names.get(self.level, "agent")

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class OrgConfig:
    """Parsed org configuration."""

    name: str
    description: str
    root: Agent
    coordination: str = "fan-out"  # "fan-out" | "delegated"
