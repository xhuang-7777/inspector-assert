"""Record a real live run and save it to replays/ for the app's 'Replay a saved run' feature.

Usage (with GEMINI_API_KEY in the environment or in .streamlit/secrets.toml):
    python record_replay.py calculator
    python record_replay.py task_counter "Suggest one next test for a beginner."
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

from agent import load_system_prompt, run_with_fallback, tool_declarations
from llm import DEFAULT_FALLBACK_MODEL, DEFAULT_MODEL, GeminiChat

ROOT = Path(__file__).resolve().parent
DEFAULT_TASK = "Investigate this program and explain the most important result to a beginner."


def setting(name: str, default: str = "") -> str:
    secrets_file = ROOT / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        value = tomllib.loads(secrets_file.read_text(encoding="utf-8")).get(name)
        if value:
            return str(value)
    return os.environ.get(name, default)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    case_id = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TASK
    names = [setting("GEMINI_MODEL", DEFAULT_MODEL)]
    backup = setting("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
    if backup and backup not in names:
        names.append(backup)
    prompt, declarations, key = load_system_prompt(), tool_declarations(), setting("GEMINI_API_KEY")
    record = run_with_fallback(case_id, task, names, lambda name: GeminiChat(key, name, prompt, declarations))
    for entry in record.trace:
        print(f"[{entry['step']}] {entry['title']} | reason: {entry['reason']} | {entry['observation'][:160]}")
    print("outcome:", record.outcome, record.error_message)
    if record.outcome != "completed":
        print("Not saved: only completed live runs are saved as replays.")
        return 1
    out_dir = ROOT / "replays"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{case_id}_{record.started_at[:10]}.json"
    out_path.write_text(json.dumps({"saved_at": record.started_at, "record": record.to_dict()}, indent=2),
                        encoding="utf-8")
    print("Saved", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
