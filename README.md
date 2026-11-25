# Inspector Assert: a Bug Detective Developer Buddy

A small Streamlit app in which a Gemini model investigates one bundled classroom Python program.
The model chooses actions, and the app validates and executes the two supplied tools (`inspect_case`, `run_checks`).
The real results go back to the model, which writes a beginner-friendly, evidence-based report.

**Public app:** _add your `https://...streamlit.app` URL here_

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface: case selector, task box, Run/Reset, activity trace, report, replay, testing switches |
| `agent.py` | Bounded agent loop (max 4 tool executions, 5 model calls), action validation, evidence classification |
| `llm.py` | Gemini adapter (tool declarations only, automatic function calling disabled) and error translation |
| `system_prompt.txt` | The buddy's system prompt |
| `tools.py`, `case_requirements.json`, `sample_code/`, `tests/` | Supplied starter tools and cases (unchanged) |
| `replays/` | Saved real runs for the "Replay a saved run" feature |
| `record_replay.py` | Records a live run into `replays/` |
| `dev_tests/` | Offline checks of the loop and UI (scripted model, no API key) |

## Run locally (Windows PowerShell)

```text
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then paste your Gemini key into it
python -m streamlit run app.py
```

Offline checks: `python -m pytest -q dev_tests`. Supplied case checks: `python -m pytest -q tests/test_calculator.py`
(one test is expected to fail).

## Deploy (Streamlit Community Cloud)

1. Push this folder (with `app.py` at the repository root) to a GitHub repository. Never commit `.streamlit/secrets.toml` or `.venv/`.
2. At share.streamlit.io: Create app, then select the repo, branch `main`, and entrypoint `app.py`. Under Advanced settings, pick Python 3.13.
3. In Advanced settings, open Secrets and paste:
   ```toml
   GEMINI_API_KEY = "..."
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
4. In Sharing, choose **This app is public and searchable**. Test the URL in a private browser window.
