"""Scan the /testbed directory tree and return a structured JSON report.

Reports all files under /testbed/data, /testbed/emails, and /testbed/calendar
with file types, sizes, and user-level summaries for emails and calendar.
"""
import json
import os
import shlex

import fire


DEMO = (
    "Scan the testbed workspace to see all available files, emails, and calendar entries. "
    "{'app': 'env_explorer', 'action': 'scan_testbed'}"
)


def construct_action(work_dir, args: dict, py_file_path="/apps/env_explorer_app/scan_testbed.py"):
    return f"python3 {shlex.quote(py_file_path)}"


# ── Extension → human-readable type mapping ──────────────────────
_EXT_MAP = {
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel",
    ".csv": "csv", ".tsv": "tsv",
    ".docx": "word", ".doc": "word",
    ".pdf": "pdf",
    ".txt": "text", ".md": "text", ".log": "text",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".gif": "image", ".bmp": "image", ".tiff": "image",
    ".ics": "calendar",
    ".eml": "email",
    ".json": "json", ".xml": "xml", ".yaml": "yaml", ".yml": "yaml",
    ".py": "python", ".sh": "script",
    ".zip": "archive", ".tar": "archive", ".gz": "archive",
}


def _file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _EXT_MAP.get(ext, ext.lstrip(".") or "unknown")


def _scan_dir(root: str) -> list[dict]:
    """Walk a directory and collect file metadata."""
    entries = []
    if not os.path.isdir(root):
        return entries
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                size = 0
            entries.append({
                "path": fpath,
                "type": _file_type(fname),
                "size_kb": round(size / 1024, 1),
            })
    return entries


def _summarise_emails(email_root: str) -> list[dict]:
    """Summarise per-user email counts."""
    users = []
    if not os.path.isdir(email_root):
        return users
    for user_dir in sorted(os.listdir(email_root)):
        user_path = os.path.join(email_root, user_dir)
        if not os.path.isdir(user_path):
            continue
        eml_files = [f for f in os.listdir(user_path) if f.endswith(".eml")]
        users.append({
            "username": user_dir,
            "email_count": len(eml_files),
            "files": [os.path.join(user_path, f) for f in sorted(eml_files)],
        })
    return users


def _summarise_calendar(cal_root: str) -> list[dict]:
    """Summarise calendar entries."""
    entries = []
    if not os.path.isdir(cal_root):
        return entries
    for fname in sorted(os.listdir(cal_root)):
        if fname.endswith(".ics"):
            entries.append(os.path.join(cal_root, fname))
    # Also check user sub-dirs
    for user_dir in sorted(os.listdir(cal_root)):
        user_path = os.path.join(cal_root, user_dir)
        if os.path.isdir(user_path):
            for fname in sorted(os.listdir(user_path)):
                if fname.endswith(".ics"):
                    entries.append(os.path.join(user_path, fname))
    return entries


def main() -> str:
    testbed = "/testbed"
    report = {
        "data_files": _scan_dir(os.path.join(testbed, "data")),
        "email_users": _summarise_emails(os.path.join(testbed, "emails")),
        "calendar_files": _summarise_calendar(os.path.join(testbed, "calendar")),
        "other_files": _scan_dir(testbed),
        "hints": [
            "Use python3 (not python) for scripts.",
            "Write output files to /testbed/data/ unless told otherwise.",
            "Use shell echo/printf to create .txt files, NOT excel.set_cell.",
            "Use email agent send_email for sending emails — do not create .eml files manually.",
            "Use calendar agent create_event for calendar — do not create .ics files manually.",
        ],
    }
    # Filter 'other_files' to only things NOT in data/emails/calendar
    known_prefixes = (
        os.path.join(testbed, "data"),
        os.path.join(testbed, "emails"),
        os.path.join(testbed, "calendar"),
    )
    report["other_files"] = [
        f for f in report["other_files"]
        if not any(f["path"].startswith(p) for p in known_prefixes)
    ]
    return json.dumps(report, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
