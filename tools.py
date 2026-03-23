"""Tools for multi-agent org agents.

Base tools (read, write, edit, grep, glob, run_command) from claude_code_langgraph
plus RMS tools for belief tracking with automatic retraction cascades.
"""

import glob as glob_module
import json
import os
import re
import subprocess

from langchain_core.tools import tool


# --- Base tools (same as claude_code_langgraph) ---

@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path. Creates the file if it doesn't exist."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace an exact string match in a file. old_string must appear exactly once."""
    try:
        with open(path, "r") as f:
            content = f.read()
        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1:
            return f"Error: old_string appears {count} times in {path}"
        new_content = content.replace(old_string, new_string, 1)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {e}"


@tool
def grep(pattern: str, path: str = ".") -> str:
    """Search for a regex pattern across files in a directory."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"
    matches = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".venv")]
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{filepath}:{i}: {line.rstrip()}")
            except (OSError, IsADirectoryError):
                continue
    if not matches:
        return "No matches found"
    return "\n".join(matches[:100])


@tool
def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern (e.g. '**/*.py')."""
    full_pattern = os.path.join(path, pattern)
    matches = glob_module.glob(full_pattern, recursive=True)
    skip = {".git", ".venv", "node_modules", "__pycache__"}
    filtered = []
    for m in matches:
        parts = m.split(os.sep)
        if any(p in skip or (p.startswith(".") and p != ".") for p in parts):
            continue
        if os.path.isfile(m):
            filtered.append(os.path.relpath(m, path))
    if not filtered:
        return "No files found"
    return "\n".join(sorted(filtered))


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


BASE_TOOLS = [read_file, write_file, edit_file, grep, glob_files, run_command]


# --- RMS tools ---

try:
    from reasons_lib import api as reasons_api
    _REASONS_AVAILABLE = True
except ImportError:
    _REASONS_AVAILABLE = False


@tool
def reasons_status(db_path: str = "rms.db") -> str:
    """Show all beliefs in the RMS network with truth values (IN or OUT)."""
    return json.dumps(reasons_api.get_status(db_path=db_path), indent=2)


@tool
def reasons_add(node_id: str, text: str, sl: str = "", unless: str = "",
            label: str = "", source: str = "", db_path: str = "rms.db") -> str:
    """Add a belief to the RMS network. Use sl for dependencies, unless for outlist."""
    return json.dumps(reasons_api.add_node(node_id, text, sl=sl, unless=unless,
                                        label=label, source=source, db_path=db_path))


@tool
def reasons_retract(node_id: str, db_path: str = "rms.db") -> str:
    """Retract a belief and cascade to all dependents."""
    return json.dumps(reasons_api.retract_node(node_id, db_path=db_path))


@tool
def reasons_assert(node_id: str, db_path: str = "rms.db") -> str:
    """Assert a belief (mark IN) and cascade restoration."""
    return json.dumps(reasons_api.assert_node(node_id, db_path=db_path))


@tool
def reasons_explain(node_id: str, db_path: str = "rms.db") -> str:
    """Explain why a belief is IN or OUT by tracing its justification chain."""
    return json.dumps(reasons_api.explain_node(node_id, db_path=db_path), indent=2)


@tool
def reasons_search(query: str, db_path: str = "rms.db") -> str:
    """Search beliefs by text or ID (case-insensitive)."""
    return json.dumps(reasons_api.search(query, db_path=db_path), indent=2)


@tool
def reasons_trace(node_id: str, db_path: str = "rms.db") -> str:
    """Trace backward to find all premises a belief rests on."""
    return json.dumps(reasons_api.trace_assumptions(node_id, db_path=db_path))


@tool
def reasons_challenge(target_id: str, reason: str, db_path: str = "rms.db") -> str:
    """Challenge a belief. Creates a challenge node and the target goes OUT."""
    return json.dumps(reasons_api.challenge(target_id, reason, db_path=db_path))


@tool
def reasons_defend(target_id: str, challenge_id: str, reason: str,
               db_path: str = "rms.db") -> str:
    """Defend a belief against a challenge. Neutralises the challenge, target restored."""
    return json.dumps(reasons_api.defend(target_id, challenge_id, reason, db_path=db_path))


@tool
def reasons_nogood(node_ids: list[str], db_path: str = "rms.db") -> str:
    """Record a contradiction. Uses backtracking to retract the responsible premise."""
    return json.dumps(reasons_api.add_nogood(node_ids, db_path=db_path))


@tool
def reasons_compact(budget: int = 500, db_path: str = "rms.db") -> str:
    """Token-budgeted summary of the belief network."""
    return reasons_api.compact(budget=budget, db_path=db_path)


REASONS_TOOLS = [
    rms_status, rms_add, rms_retract, rms_assert, rms_explain,
    rms_search, rms_trace, rms_challenge, rms_defend, rms_nogood, rms_compact,
]

ALL_TOOLS = BASE_TOOLS + (REASONS_TOOLS if _REASONS_AVAILABLE else [])
