"""Offline checks of the controller loop using a scripted stand-in model.

These do not call Gemini. They verify the controller's guarantees: real tools run,
results reach the model, limits hold, and unsupported finishes are rejected.
Run with:  python -m pytest -q dev_tests
"""
from agent import MAX_MODEL_CALLS, ModelError, ModelReply, ToolCall, run_investigation


class ScriptedModel:
    """Replies with a fixed list of actions and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.received = []

    def _next(self):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def send_text(self, text):
        self.received.append(("text", text))
        return self._next()

    def send_tool_results(self, results):
        self.received.append(("tools", results))
        return self._next()


def call(name, case_id="calculator", **extra):
    return ModelReply("", [ToolCall(name, {"case_id": case_id, "reason": "test", **extra})])


def report(status, **overrides):
    args = {
        "relevant_requirement": "divide(7, 2) must return 3.5",
        "observed_status": status,
        "finding": "finding",
        "beginner_explanation": "explanation",
        "next_step": "behavior change: use true division",
        "limitation": "only supplied tests",
        "reason": "done",
    }
    args.update(overrides)
    return ModelReply("", [ToolCall("submit_report", args)])


def test_happy_path_calculator_uses_real_failing_evidence():
    model = ScriptedModel([call("inspect_case"), call("run_checks"), report("tests_failed")])
    record = run_investigation("calculator", "Investigate", model, "scripted")
    assert record.outcome == "completed"
    assert record.check_kind == "tests_failed"
    assert "1 failed, 6 passed" in record.check_result["output"]
    # The real pytest output was delivered to the model before it finished.
    delivered = [payload for kind, items in model.received if kind == "tools" for _, payload in items]
    assert any("assert 3 == 3.5" in p.get("output", "") for p in delivered)
    assert record.tool_executions == 2 and record.model_calls == 3


def test_claiming_success_on_failing_case_is_rejected_then_corrected():
    model = ScriptedModel([call("inspect_case"), call("run_checks"), report("passed"), report("tests_failed")])
    record = run_investigation("calculator", "Investigate", model, "scripted")
    assert record.outcome == "completed"
    assert any(e["kind"] == "rejected" and "does not match" in e["observation"] for e in record.trace)


def test_report_before_evidence_is_rejected():
    model = ScriptedModel([report("passed"), call("inspect_case", "task_counter"),
                           call("run_checks", "task_counter"), report("passed")])
    record = run_investigation("task_counter", "Investigate", model, "scripted")
    assert record.outcome == "completed"
    assert record.trace[1]["kind"] == "rejected"
    assert record.check_kind == "passed"


def test_report_in_same_turn_as_tools_is_not_accepted():
    both = ModelReply("", [ToolCall("inspect_case", {"case_id": "task_counter", "reason": "r"}),
                           ToolCall("run_checks", {"case_id": "task_counter", "reason": "r"}),
                           ToolCall("submit_report", report("passed").calls[0].args)])
    model = ScriptedModel([both, report("passed")])
    record = run_investigation("task_counter", "Investigate", model, "scripted")
    assert record.outcome == "completed"
    assert any(e["kind"] == "rejected" for e in record.trace)
    assert record.model_calls == 2


def test_wrong_and_unsupported_case_ids_are_not_executed():
    model = ScriptedModel([call("inspect_case", "average"), call("run_checks", "../../etc"),
                           call("delete_files"), call("inspect_case"), call("run_checks"),
                           report("tests_failed")])
    record = run_investigation("calculator", "Investigate", model, "scripted")
    rejected = [e for e in record.trace if e["kind"] == "rejected"]
    assert len(rejected) == 3  # other case, path-like id, unknown tool
    assert record.tool_executions == 2  # only the valid calls ran
    assert record.outcome == "stopped_limit"  # 5 model calls used before a report
    assert record.model_calls == MAX_MODEL_CALLS


def test_tool_budget_is_enforced():
    model = ScriptedModel([call("run_checks")] * 5)
    record = run_investigation("calculator", "Investigate", model, "scripted")
    assert record.tool_executions == 4
    assert record.outcome == "stopped_limit"
    assert record.report is None


def test_model_outage_is_reported_not_faked():
    model = ScriptedModel([call("inspect_case"), ModelError("quota", "HTTP 429")])
    record = run_investigation("calculator", "Investigate", model, "scripted")
    assert record.outcome == "model_error" and record.error_kind == "quota"
    assert record.report is None


def test_quota_on_primary_switches_to_backup_and_says_so():
    from agent import run_with_fallback

    scripts = {
        "primary": ScriptedModel([ModelError("quota", "HTTP 429")]),
        "backup": ScriptedModel([call("inspect_case"), call("run_checks"), report("tests_failed")]),
    }
    record = run_with_fallback("calculator", "Investigate", ["primary", "backup"], scripts.__getitem__)
    assert record.outcome == "completed" and record.model_name == "backup"
    assert record.trace[0]["kind"] == "note" and "backup" in record.trace[0]["observation"]


def test_non_quota_error_does_not_switch_model():
    from agent import run_with_fallback

    scripts = {"primary": ScriptedModel([ModelError("auth", "bad key")]), "backup": ScriptedModel([])}
    record = run_with_fallback("calculator", "Investigate", ["primary", "backup"], scripts.__getitem__)
    assert record.outcome == "model_error" and record.model_name == "primary"


def test_text_reply_without_action_is_nudged():
    model = ScriptedModel([ModelReply("Everything looks fine!", []), call("inspect_case"),
                           call("run_checks"), report("tests_failed")])
    record = run_investigation("calculator", "Investigate", model, "scripted")
    assert record.outcome == "completed"
    assert record.trace[1]["kind"] == "rejected"
