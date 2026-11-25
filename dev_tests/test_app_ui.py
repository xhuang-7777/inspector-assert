"""Offline UI checks with Streamlit's AppTest (no model key needed)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def fresh_app(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30)
    at.secrets["GEMINI_API_KEY"] = ""  # replaces the real secrets file: no live calls in tests
    at.run()
    return at


def button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


def test_home_page_controls(monkeypatch):
    at = fresh_app(monkeypatch)
    assert not at.exception
    assert at.selectbox(key="case_id").value == "calculator"
    assert "Investigate this program" in at.text_area(key="task").value
    labels = [b.label for b in at.button]
    assert any("Run my buddy" in label for label in labels)
    assert any("Reset" in label for label in labels)


def test_missing_key_gives_helpful_error_not_fake_success(monkeypatch):
    at = fresh_app(monkeypatch)
    button(at, "🔎 Run").click().run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "never need to provide a key" in errors
    assert "Case report" not in [s.value for s in at.subheader]


def test_simulated_outage_is_labeled(monkeypatch):
    at = fresh_app(monkeypatch)
    at.sidebar.checkbox[0].check().run()
    button(at, "🔎 Run").click().run()
    assert "Simulated outage" in " ".join(e.value for e in at.error)


def test_completed_report_renders_with_replay_label(monkeypatch):
    from agent import run_investigation
    from test_agent_loop import ScriptedModel, call, report

    model = ScriptedModel([call("inspect_case"), call("run_checks"), report("tests_failed")])
    record = run_investigation("calculator", "Investigate", model, "scripted").to_dict()
    at = fresh_app(monkeypatch)
    at.session_state["run"] = record
    at.session_state["run_mode"] = "replay"
    at.run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "SAVED REPLAY" in page and "not a fresh run" in page
    assert "TESTS FAILED" in page
    assert any("assert 3 == 3.5" in c.value for c in at.code)
    assert "Activity trace" in " ".join(s.value for s in at.subheader)


def test_switching_case_and_reset_clear_previous_result(monkeypatch):
    at = fresh_app(monkeypatch)
    button(at, "🔎 Run").click().run()
    assert at.error  # a result (the error) is on screen
    at.selectbox(key="case_id").set_value("task_counter").run()
    assert not at.error  # old case result removed when the case changes
    at.text_area(key="task").set_value("Suggest one next test.").run()
    button(at, "🔎 Run").click().run()
    button(at, "↺ Reset").click().run()
    assert not at.error
    assert at.selectbox(key="case_id").value == "calculator"
    assert "Investigate this program" in at.text_area(key="task").value
