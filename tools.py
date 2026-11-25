"""Starter tools for the Developer Buddy assignment.

Students build the LLM/controller loop and web interface around these functions.
Only bundled classroom cases can be executed; browser-provided code is not run.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CASES = json.loads((ROOT / "case_requirements.json").read_text())


def _case(case_id):
    if case_id not in CASES:
        raise ValueError("Choose calculator, average, or task_counter.")
    return CASES[case_id]


def inspect_case(case_id):
    """Return the selected case's actual source code and behavior requirements."""
    case = _case(case_id)
    return {
        "case_id": case_id,
        "requirements": case["rules"],
        "source_code": (ROOT / case["source"]).read_text(),
        "test_code": (ROOT / case["tests"]).read_text(),
    }


def run_checks(case_id):
    """Run the selected case's supplied pytest tests and return real evidence."""
    case = _case(case_id)
    command = [sys.executable, "-m", "pytest", "-q", case["tests"], "--tb=short"]
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        return {"case_id": case_id, "status": "timeout", "output": "Test run timed out."}
    return {
        "case_id": case_id,
        "status": "passed" if completed.returncode == 0 else "not_passed",
        "exit_code": completed.returncode,
        "command": "python -m pytest -q " + case["tests"] + " --tb=short",
        "output": (completed.stdout + completed.stderr)[-10000:],
    }


# Framework-neutral descriptions. Adapt these to your provider's tool API,
# or use them in a JSON-action prompt and dispatch actions in your controller.
TOOL_DESCRIPTIONS = [
    {
        "name": "inspect_case",
        "description": "Read a bundled classroom case's requirements, source, and tests.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {
                    "type": "string",
                    "enum": ["calculator", "average", "task_counter"],
                }
            },
            "required": ["case_id"],
        },
    },
    {
        "name": "run_checks",
        "description": "Execute supplied tests for one bundled case and return actual pytest output.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {
                    "type": "string",
                    "enum": ["calculator", "average", "task_counter"],
                }
            },
            "required": ["case_id"],
        },
    },
]
