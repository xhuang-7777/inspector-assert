"""Inspector Assert: a Bug Detective Developer Buddy (Streamlit app).

Run locally with:  python -m streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path

import streamlit as st

from agent import (
    MAX_MODEL_CALLS,
    MAX_TOOL_EXECUTIONS,
    load_system_prompt,
    output_excerpt,
    run_investigation,
    run_with_fallback,
    tool_declarations,
)
from llm import DEFAULT_FALLBACK_MODEL, DEFAULT_MODEL, GeminiChat, UnavailableModel
from tools import CASES

ROOT = Path(__file__).resolve().parent
REPLAY_DIR = ROOT / "replays"
DEFAULT_TASK = "Investigate this program and explain the most important result to a beginner."
CASE_LABELS = {
    "calculator": "calculator (Calculator class)",
    "task_counter": "task_counter (count completed tasks)",
    "average": "average (optional extra: lecture exercise)",
}

st.set_page_config(page_title="Inspector Assert: Bug Detective", page_icon="🔎", layout="centered")

st.markdown(
    """
<style>
.ia-card {border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0 0.9rem 0; line-height: 1.5;}
.ia-evidence {border: 2px solid #3b6ea5; background: rgba(59,110,165,0.08);}
.ia-finding {border: 2px solid #5b4a8b; background: rgba(91,74,139,0.07);}
.ia-suggestion {border: 2px dashed #c48a12; background: rgba(196,138,18,0.08);}
.ia-limit {border: 2px solid #8a8a8a; background: rgba(138,138,138,0.08);}
.ia-label {font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.8;}
.ia-banner {border-radius: 8px; padding: 0.6rem 0.9rem; margin: 0.6rem 0; font-weight: 600;}
.ia-live {background: rgba(46,139,87,0.15); border-left: 6px solid #2e8b57;}
.ia-cache {background: rgba(59,110,165,0.13); border-left: 6px solid #3b6ea5;}
.ia-replay {background: repeating-linear-gradient(45deg, rgba(128,90,213,0.16), rgba(128,90,213,0.16) 12px,
            rgba(128,90,213,0.07) 12px, rgba(128,90,213,0.07) 24px); border-left: 6px solid #805ad5;}
.ia-step {border-left: 3px solid rgba(128,128,128,0.45); padding: 0.15rem 0 0.15rem 0.8rem; margin: 0.35rem 0;}
.ia-step small {opacity: 0.75;}
.ia-badge {display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px; font-weight: 700; color: white;}
.ia-passed {background: #2e8b57;} .ia-tests_failed {background: #c0392b;} .ia-tool_problem {background: #7f8c8d;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Configuration and helpers
# --------------------------------------------------------------------------
def get_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file locally
        value = None
    return str(value or os.environ.get(name, default))


MODEL_NAME = get_setting("GEMINI_MODEL", DEFAULT_MODEL)
FALLBACK_MODEL = get_setting("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
MODEL_NAMES = [MODEL_NAME] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL_NAME else [])
SYSTEM_PROMPT = load_system_prompt()
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:10]


@st.cache_resource
def run_cache() -> dict:
    """Completed live runs keyed by identical request, shared across sessions."""
    return {}


def esc(text: str) -> str:
    return html.escape(str(text)).replace("\n", "<br>")


def card(css: str, label: str, body_html: str) -> None:
    st.markdown(
        f'<div class="ia-card {css}"><div class="ia-label">{label}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def load_replays() -> dict[str, dict]:
    replays = {}
    for path in sorted(REPLAY_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            replays[path.stem] = data
        except (OSError, json.JSONDecodeError):
            continue
    return replays


def clear_run() -> None:
    for key in ("run", "run_mode", "run_note"):
        st.session_state.pop(key, None)


def reset_all() -> None:
    clear_run()
    st.session_state["case_id"] = "calculator"
    st.session_state["task"] = DEFAULT_TASK


ERROR_HELP = {
    "not_configured": "The buddy's model key is not set up on this server. That is the app owner's job; "
                      "you never need to provide a key.",
    "auth": "The server's model credential was rejected. The app owner needs to update it.",
    "quota": "The free model quota is used up for the moment. Wait a minute and try again.",
    "timeout": "The model service took too long to answer.",
    "unavailable": "The model service is unavailable right now.",
    "model_not_found": "The configured model name is not available for this key.",
    "bad_request": "The model service could not process the request.",
    "empty": "The model sent back an empty reply.",
}


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🔎 Inspector Assert")
st.caption("Your Bug Detective buddy for small Python programs")
st.markdown(
    "Inspector Assert investigates one small classroom Python program by reading its written requirements "
    "and running its real tests. Then it explains in beginner-friendly language what the evidence shows, "
    "and keeps proven facts separate from suggestions."
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.session_state.setdefault("case_id", "calculator")
st.session_state.setdefault("task", DEFAULT_TASK)

with st.sidebar:
    st.header("🕵️ Case file")
    st.markdown(
        f"- **Identity:** Bug Detective\n"
        f"- **Model:** `{MODEL_NAME}` (Gemini API, free tier)"
        + (f", backup `{MODEL_NAMES[1]}` if the daily quota runs out\n" if len(MODEL_NAMES) > 1 else "\n") +
        f"- **Limits per run:** {MAX_TOOL_EXECUTIONS} tool executions, {MAX_MODEL_CALLS} model calls\n"
        f"- **Tools:** `inspect_case`, `run_checks`"
    )

    st.header("🎞️ Replay a saved run")
    replays = load_replays()
    if replays:
        choice = st.selectbox(
            "Saved runs",
            list(replays),
            format_func=lambda key: f"{replays[key]['record']['case_id']}: saved {replays[key].get('saved_at', '?')}",
        )
        if st.button("Show saved replay"):
            st.session_state["run"] = replays[choice]["record"]
            st.session_state["run_mode"] = "replay"
            st.session_state["run_note"] = replays[choice].get("saved_at", "")
        st.caption("A replay is an earlier recorded run. It is never presented as a fresh run.")
    else:
        st.caption("No saved replays are bundled yet.")

    st.header("🧪 Testing switches")
    simulate_outage = st.checkbox(
        "Simulate model outage", help="Error-handling demo: the next run behaves as if the model service is down."
    )
    force_fresh = st.checkbox(
        "Always run fresh (skip cache)", help="Identical requests normally reuse the last live result to save quota."
    )

    with st.expander("📖 Detective's glossary"):
        st.markdown(
            "- **Requirement:** a written rule the program must follow.\n"
            "- **Test:** a small check: *given* inputs, *when* the function runs, *then* a result or error must happen.\n"
            "- **Assertion (`assert`):** the line in a test that compares the actual result with the required one.\n"
            "- **pytest:** the tool that runs the tests and reports passed/failed.\n"
            "- **Exit code:** a number pytest returns: 0 = all passed, 1 = some tests failed, other = the run itself had a problem.\n"
            "- **Behavior change vs refactor:** a behavior change makes the program do something different (e.g. fix a bug); "
            "a refactor only tidies the code and keeps behavior the same."
        )

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
case_id = st.selectbox(
    "1. Pick a classroom case",
    list(CASES),
    format_func=lambda key: CASE_LABELS.get(key, key),
    key="case_id",
    on_change=clear_run,
)
task = st.text_area("2. What should the buddy do? (you can edit this)", key="task", height=90)
col_run, col_reset = st.columns([3, 1])
run_clicked = col_run.button("🔎 Run my buddy", type="primary")
col_reset.button("↺ Reset", on_click=reset_all)

if run_clicked:
    clear_run()
    cache_key = (case_id, task.strip(), tuple(MODEL_NAMES), PROMPT_HASH)
    cached = run_cache().get(cache_key)
    if cached and not force_fresh and not simulate_outage:
        st.session_state["run"] = cached
        st.session_state["run_mode"] = "cache"
    else:
        with st.spinner("Inspector Assert is examining the clues..."):
            if simulate_outage:
                record = run_investigation(case_id, task, UnavailableModel(), MODEL_NAME).to_dict()
            else:
                record = run_with_fallback(
                    case_id,
                    task,
                    MODEL_NAMES,
                    lambda name: GeminiChat(
                        api_key=get_setting("GEMINI_API_KEY"),
                        model_name=name,
                        system_prompt=SYSTEM_PROMPT,
                        declarations=tool_declarations(),
                    ),
                ).to_dict()
        if record["outcome"] == "completed" and not simulate_outage:
            run_cache()[cache_key] = record
        st.session_state["run"] = record
        st.session_state["run_mode"] = "simulated" if simulate_outage else "live"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
STEP_ICONS = {"note": "🔁", "start": "📝", "action": "🧭", "observation": "🔬", "rejected": "⛔", "final": "✅", "stop": "⏹️", "error": "⚠️"}
STATUS_TEXT = {"passed": "PASSED", "tests_failed": "TESTS FAILED", "tool_problem": "TOOL PROBLEM"}


def render_banner(record: dict, mode: str) -> None:
    if mode == "replay":
        css, text = "ia-replay", (
            f"🎞️ SAVED REPLAY: this is an earlier recorded run (originally run {record.get('started_at')}), "
            "not a fresh run."
        )
    elif mode == "cache":
        css, text = "ia-cache", (
            f"♻️ Reused result: an identical request ran live at {record.get('started_at')}. "
            "Tick 'Always run fresh' in the sidebar to call the model again."
        )
    elif mode == "simulated":
        css, text = "ia-cache", "🧪 Simulated outage run (testing switch is on). No model was called."
    else:
        css, text = "ia-live", f"🟢 Live run with {record.get('model_name')} at {record.get('started_at')}."
    st.markdown(f'<div class="ia-banner {css}">{esc(text)}</div>', unsafe_allow_html=True)


def render_trace(record: dict) -> None:
    st.subheader("🗂️ Activity trace")
    st.caption(
        f"Model calls used: {record.get('model_calls', 0)}/{MAX_MODEL_CALLS} · "
        f"Tool executions: {record.get('tool_executions', 0)}/{MAX_TOOL_EXECUTIONS}"
    )
    for entry in record.get("trace", []):
        icon = STEP_ICONS.get(entry["kind"], "•")
        lines = [f"<b>{icon} Step {entry['step']}: {esc(entry['title'])}</b>"]
        if entry.get("reason"):
            lines.append(f"<small>Reason (from the model):</small> {esc(entry['reason'])}")
        if entry.get("observation") and entry["kind"] != "start":
            lines.append(f"<small>Result:</small> {esc(entry['observation'])}")
        if entry.get("next"):
            lines.append(f"<small>Next:</small> {esc(entry['next'])}")
        st.markdown('<div class="ia-step">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
        if entry["kind"] == "observation" and entry.get("raw"):
            with st.expander(f"Raw tool result (step {entry['step']})"):
                st.json(entry["raw"], expanded=False)


def render_evidence(record: dict) -> None:
    check = record.get("check_result")
    kind = record.get("check_kind")
    if not check:
        card("ia-evidence", "Observed evidence: from run_checks, not written by the AI",
             "The tests were <b>not run</b> in this session, so no test result is claimed.")
        return
    body = (
        f'<span class="ia-badge ia-{kind}">{STATUS_TEXT.get(kind, kind)}</span><br>'
        f"<b>Command:</b> <code>{esc(check.get('command', 'n/a'))}</code> · "
        f"<b>Exit code:</b> {esc(check.get('exit_code', 'n/a'))}<br>{esc(check.get('controller_note', ''))}"
    )
    card("ia-evidence", "Observed evidence: from run_checks, not written by the AI", body)
    st.code(output_excerpt(check.get("output", check.get("error", ""))), language="text")


def render_report(record: dict) -> None:
    report = record["report"]
    st.subheader("📋 Case report")
    st.markdown(f"**Case:** `{record['case_id']}`")
    card("ia-finding", "Relevant requirement", esc(report["relevant_requirement"]))
    with st.expander("All written requirements for this case"):
        for rule in CASES[record["case_id"]]["rules"]:
            st.markdown(f"- {rule}")
    render_evidence(record)
    card("ia-finding", "🕵️ Detective's finding (AI explanation of the evidence above)",
         f"<b>{esc(report['finding'])}</b><br><br>{esc(report['beginner_explanation'])}")
    card("ia-suggestion", "💡 Suggested next step: a suggestion, not verified. You decide.",
         esc(report["next_step"]))
    card("ia-limit", "⚠️ Limitation", esc(report["limitation"]))


def render_failure(record: dict, mode: str) -> None:
    outcome = record.get("outcome")
    if outcome == "model_error":
        help_text = ERROR_HELP.get(record.get("error_kind"), "The model service had a problem.")
        st.error(
            f"**The buddy could not finish this run.** {help_text}\n\n"
            f"Details: {record.get('error_message')}\n\n"
            "Nothing below is a fresh AI conclusion. You can open **Replay a saved run** in the sidebar "
            "to see an earlier recorded run (clearly labeled)."
        )
    elif outcome == "stopped_limit":
        st.warning(
            "**Run stopped at the safety limit** before the model produced an acceptable report. "
            "Only raw evidence collected so far is shown; no conclusion is claimed."
        )
    else:
        st.error(record.get("error_message") or "The run did not complete.")
    if record.get("check_result"):
        st.subheader("Raw evidence collected before stopping")
        render_evidence(record)


if "run" in st.session_state:
    record = st.session_state["run"]
    mode = st.session_state.get("run_mode", "live")
    st.divider()
    render_banner(record, mode)
    st.markdown(f"**Task:** {esc(record.get('task', ''))}", unsafe_allow_html=True)
    if record.get("outcome") == "completed":
        render_report(record)
    else:
        render_failure(record, mode)
    if record.get("trace"):
        render_trace(record)
    if mode == "live" and record.get("outcome") == "completed":
        st.download_button(
            "⬇️ Download this run as JSON (to save as a replay)",
            data=json.dumps({"saved_at": record.get("started_at"), "record": record}, indent=2),
            file_name=f"{record['case_id']}_run.json",
            mime="application/json",
        )
