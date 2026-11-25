"""Bounded agent loop for Inspector Assert (the Developer Buddy).

The language model chooses actions; this controller validates them, executes the
two supplied tools from tools.py, records what happened, and sends the real
observations back to the model. The loop is provider-neutral: any object with
``send_text`` and ``send_tool_results`` methods (see ``ChatModel``) can drive it.
"""
from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tools import CASES, TOOL_DESCRIPTIONS, inspect_case, run_checks

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = ROOT / "system_prompt.txt"

MAX_TOOL_EXECUTIONS = 4
MAX_MODEL_CALLS = 5

TOOLS = {"inspect_case": inspect_case, "run_checks": run_checks}
STATUSES = ("passed", "tests_failed", "tool_problem")
REPORT_FIELDS = (
    "relevant_requirement",
    "observed_status",
    "finding",
    "beginner_explanation",
    "next_step",
    "limitation",
)


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Tool declarations sent to the model
# --------------------------------------------------------------------------
SUBMIT_REPORT_DECLARATION = {
    "name": "submit_report",
    "description": (
        "Finish the investigation with an evidence-based report. Only call this "
        "after you have received results from both inspect_case and run_checks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "relevant_requirement": {
                "type": "string",
                "description": "The written requirement most related to the finding, quoted.",
            },
            "observed_status": {
                "type": "string",
                "enum": list(STATUSES),
                "description": "Must equal the controller_note classification of run_checks.",
            },
            "finding": {"type": "string", "description": "Most important result, citing test evidence."},
            "beginner_explanation": {"type": "string", "description": "2-4 short beginner-friendly sentences."},
            "next_step": {
                "type": "string",
                "description": "One suggested next step labeled behavior change, refactor, or new test idea.",
            },
            "limitation": {"type": "string", "description": "One honest limitation of this investigation."},
        },
        "required": list(REPORT_FIELDS),
    },
}


def tool_declarations() -> list[dict[str, Any]]:
    """Supplied tool descriptions plus a required short ``reason`` and the finish action."""
    declarations = []
    for description in TOOL_DESCRIPTIONS:
        declaration = copy.deepcopy(description)
        params = declaration["parameters"]
        params["properties"]["reason"] = {
            "type": "string",
            "description": "One short sentence: why this tool is the right next step.",
        }
        params["required"] = ["case_id", "reason"]
        declarations.append(declaration)
    declarations.append(copy.deepcopy(SUBMIT_REPORT_DECLARATION))
    return declarations


# --------------------------------------------------------------------------
# Model interface
# --------------------------------------------------------------------------
@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class ModelReply:
    text: str
    calls: list[ToolCall]


class ModelError(Exception):
    """The model service could not produce a reply.

    kind is one of: not_configured, auth, quota, timeout, unavailable,
    model_not_found, bad_request, empty.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class ChatModel(Protocol):
    def send_text(self, text: str) -> ModelReply: ...

    def send_tool_results(self, results: list[tuple[str, dict[str, Any]]]) -> ModelReply: ...


# --------------------------------------------------------------------------
# Evidence helpers
# --------------------------------------------------------------------------
def classify_check(result: dict[str, Any]) -> str:
    """Separate real test failures from problems starting or running pytest."""
    if result.get("status") == "timeout":
        return "tool_problem"
    exit_code = result.get("exit_code")
    output = result.get("output", "")
    if exit_code == 0 and re.search(r"\b\d+ passed\b", output):
        return "passed"
    if exit_code == 1 and re.search(r"\b\d+ failed\b", output):
        return "tests_failed"
    return "tool_problem"


CHECK_NOTES = {
    "passed": "All supplied tests passed (exit code 0). This does not prove the program is perfect.",
    "tests_failed": "pytest ran and at least one supplied test FAILED (exit code 1). The failure is real evidence.",
    "tool_problem": (
        "The test tool did not complete normally (timeout, missing pytest, or collection/usage error). "
        "Do not report this as a confirmed defect in the sample program."
    ),
}


def output_excerpt(output: str, max_lines: int = 12) -> str:
    """Short, real excerpt of pytest output: failures, assertion lines, and the summary."""
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    keep = [
        line
        for line in lines
        if re.match(r"^(FAILED|ERROR)\b", line)
        or line.startswith("E ")
        or re.search(r"\b\d+ (passed|failed|error)", line)
        or "timed out" in line
        or "No module named" in line
    ]
    if not keep:
        keep = lines[-max_lines:]
    return "\n".join(keep[:max_lines])


def summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    if "error" in result:
        return f"Error: {result['error']}"
    if name == "inspect_case":
        return (
            f"Read {len(result.get('requirements', []))} requirements, "
            f"{len(result.get('source_code', '').splitlines())} lines of source, "
            f"{len(result.get('test_code', '').splitlines())} lines of tests."
        )
    if name == "run_checks":
        lines = [line for line in result.get("output", "").splitlines() if line.strip()]
        summary = lines[-1].strip() if lines else "(no output)"
        return f"exit code {result.get('exit_code', 'n/a')} -> {result.get('controller_note_kind')}: {summary}"
    return "Result recorded."


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# --------------------------------------------------------------------------
# Run record
# --------------------------------------------------------------------------
@dataclass
class RunRecord:
    case_id: str
    task: str
    model_name: str
    started_at: str = field(default_factory=_now)
    finished_at: str = ""
    outcome: str = "running"  # completed | stopped_limit | model_error | invalid_case
    error_kind: str = ""
    error_message: str = ""
    model_calls: int = 0
    tool_executions: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    inspect_result: dict[str, Any] | None = None
    check_result: dict[str, Any] | None = None
    check_kind: str = ""
    report: dict[str, str] | None = None
    limits: dict[str, int] = field(
        default_factory=lambda: {"max_tool_executions": MAX_TOOL_EXECUTIONS, "max_model_calls": MAX_MODEL_CALLS}
    )

    def add(self, kind: str, title: str, reason: str = "", observation: str = "", next_step: str = "", raw=None):
        self.trace.append(
            {
                "step": len(self.trace) + 1,
                "kind": kind,
                "title": title,
                "reason": reason,
                "observation": observation,
                "next": next_step,
                "raw": raw,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# The bounded loop
# --------------------------------------------------------------------------
def _validate_tool_call(call: ToolCall, case_id: str, record: RunRecord) -> str:
    requested = call.args.get("case_id")
    if not isinstance(requested, str) or not requested:
        return "Missing case_id."
    if requested not in CASES:
        return f"Unsupported case_id '{requested}'. Only bundled cases can be inspected."
    if requested != case_id:
        return f"This run investigates '{case_id}' only; '{requested}' was not selected."
    if record.tool_executions >= MAX_TOOL_EXECUTIONS:
        return "Tool budget used up. Call submit_report with the evidence you already have."
    return ""


def _validate_report(args: dict[str, Any], seen: set[str], record: RunRecord) -> tuple[str, dict[str, str]]:
    missing_evidence = [name for name in ("inspect_case", "run_checks") if name not in seen]
    if missing_evidence:
        return (
            "Report rejected: you have not yet received results from "
            + " and ".join(missing_evidence)
            + ". Call it first.",
            {},
        )
    report = {key: str(args.get(key, "")).strip() for key in REPORT_FIELDS}
    empty = [key for key, value in report.items() if not value]
    if empty:
        return f"Report rejected: missing fields {', '.join(empty)}.", {}
    if report["observed_status"] != record.check_kind:
        return (
            f"Report rejected: observed_status '{report['observed_status']}' does not match the real "
            f"run_checks result '{record.check_kind}'. Use the returned evidence.",
            {},
        )
    return "", report


def run_investigation(case_id: str, task: str, model: ChatModel, model_name: str) -> RunRecord:
    record = RunRecord(case_id=case_id, task=task, model_name=model_name)
    if case_id not in CASES:
        record.outcome = "invalid_case"
        record.error_message = f"'{case_id}' is not a bundled case."
        record.finished_at = _now()
        return record

    seen_by_model: set[str] = set()  # tools whose results the model has actually received
    opening = (
        f"Selected case_id: {case_id}\n"
        f"User task: {task.strip() or '(empty) Investigate this program and explain the most important result.'}\n"
        f"Limits: at most {MAX_TOOL_EXECUTIONS} tool executions and {MAX_MODEL_CALLS} model turns. "
        "Respond only by calling a function."
    )
    record.add("start", "Task sent to the model", observation=opening, next_step="Model chooses the first action.")
    pending: tuple[str, Any] = ("text", opening)

    while True:
        if record.model_calls >= MAX_MODEL_CALLS:
            record.outcome = "stopped_limit"
            record.add(
                "stop",
                "Model-call limit reached",
                observation=f"{MAX_MODEL_CALLS} model calls used without an accepted report.",
                next_step="Run stopped. Only the raw evidence below is shown; no conclusion is claimed.",
            )
            break
        try:
            if pending[0] == "text":
                reply = model.send_text(pending[1])
            else:
                reply = model.send_tool_results(pending[1])
        except ModelError as error:
            record.outcome = "model_error"
            record.error_kind = error.kind
            record.error_message = str(error)
            record.add("error", "Model service problem", observation=str(error), next_step="Run stopped safely.")
            break
        record.model_calls += 1

        if not reply.calls:
            record.add(
                "rejected",
                "Model replied without choosing an action",
                observation=(reply.text[:400] or "(empty reply)"),
                next_step="Controller asks the model to call a function instead.",
            )
            pending = ("text", "Please continue by calling exactly one function (inspect_case, run_checks, or submit_report).")
            continue

        responses: list[tuple[str, dict[str, Any]]] = []
        executed: set[str] = set()
        accepted_report = None
        for call in reply.calls:
            reason = str(call.args.get("reason", "")).strip()
            if call.name == "submit_report":
                problem, report = _validate_report(call.args, seen_by_model, record)
                if problem:
                    responses.append((call.name, {"accepted": False, "error": problem}))
                    record.add(
                        "rejected",
                        "Model tried to finish: report rejected",
                        observation=problem,
                        next_step="Rejection sent back; the model must correct itself.",
                        raw=call.args,
                    )
                else:
                    accepted_report = report
                    record.add(
                        "final",
                        "Model submitted the final report",
                        observation=f"Claimed status '{report['observed_status']}' matches the real test result.",
                        next_step="Report accepted and shown below.",
                        raw=call.args,
                    )
                continue

            if call.name not in TOOLS:
                problem = f"Unknown action '{call.name}'. Available: inspect_case, run_checks, submit_report."
                responses.append((call.name, {"error": problem}))
                record.add("rejected", f"Malformed action: {call.name}", observation=problem,
                           next_step="Error sent back to the model.", raw=call.args)
                continue

            problem = _validate_tool_call(call, case_id, record)
            if problem:
                responses.append((call.name, {"error": problem}))
                record.add("rejected", f"Rejected {call.name}", reason=reason, observation=problem,
                           next_step="Error sent back to the model.", raw=call.args)
                continue

            record.add("action", f"Model chose {call.name}('{case_id}')", reason=reason,
                       next_step="Controller executes the supplied tool.")
            try:
                result = TOOLS[call.name](case_id)
            except Exception as error:  # tool failed to start: not a program defect
                result = {"case_id": case_id, "status": "tool_error", "error": f"{type(error).__name__}: {error}"}
            record.tool_executions += 1
            executed.add(call.name)

            if call.name == "run_checks":
                kind = classify_check(result)
                result = dict(result, controller_note_kind=kind, controller_note=CHECK_NOTES[kind])
                record.check_result = result
                record.check_kind = kind
            elif "error" not in result:
                record.inspect_result = result

            responses.append((call.name, result))
            record.add(
                "observation",
                f"{call.name} returned",
                observation=summarize_tool_result(call.name, result),
                next_step="Real result sent back to the model; it decides what to do next.",
                raw=result,
            )

        if accepted_report is not None:
            record.report = accepted_report
            record.outcome = "completed"
            break

        seen_by_model.update(executed)  # these results are delivered with the next model call
        pending = ("tools", responses)

    record.finished_at = _now()
    return record


FALLBACK_ERRORS = ("quota", "model_not_found", "unavailable")


def run_with_fallback(case_id: str, task: str, model_names: list[str], make_model) -> RunRecord:
    """Run with the first model; if its quota is used up (or it is unavailable for the key),
    start a fresh run with the next configured model. Each attempt is noted honestly."""
    notes: list[str] = []
    record = None
    for index, name in enumerate(model_names):
        try:
            model = make_model(name)
        except ModelError as error:
            record = RunRecord(case_id=case_id, task=task, model_name=name, outcome="model_error",
                               error_kind=error.kind, error_message=str(error), finished_at=_now())
        else:
            record = run_investigation(case_id, task, model, name)
        if record.outcome != "model_error" or record.error_kind not in FALLBACK_ERRORS:
            break
        if index + 1 < len(model_names):
            notes.append(f"{name}: {record.error_message} Switched to backup model {model_names[index + 1]}.")
    if notes and record is not None:
        record.trace.insert(0, {"step": 0, "kind": "note", "title": "Backup model used",
                                "reason": "", "observation": " ".join(notes),
                                "next": "This run restarted from scratch with the backup model.", "raw": None})
    return record
