"""Read the first N lines of a file to preview its content without full parsing."""
import os
import shlex

import fire


DEMO = (
    "Read the first lines of a file to preview its content: "
    "{'app': 'env_explorer', 'action': 'read_snippet', "
    "'file_path': [THE_PATH_TO_THE_FILE], 'num_lines': 20}"
)


def construct_action(work_dir, args: dict, py_file_path="/apps/env_explorer_app/read_snippet.py"):
    file_path = args["file_path"]
    num_lines = args.get("num_lines", 20)
    return (
        f"python3 {shlex.quote(py_file_path)} "
        f"--file_path {shlex.quote(str(file_path))} "
        f"--num_lines {int(num_lines)}"
    )


def main(file_path: str, num_lines: int = 20) -> str:
    if not os.path.exists(file_path):
        return f"OBSERVATION: File not found: {file_path}"

    if not file_path.startswith("/testbed"):
        return f"OBSERVATION: Access denied — only /testbed paths are allowed."

    size_kb = round(os.path.getsize(file_path) / 1024, 1)
    ext = os.path.splitext(file_path)[1].lower()

    # Binary files
    if ext in (".xlsx", ".xls", ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz"):
        return (
            f"OBSERVATION: {file_path} is a binary file ({ext}, {size_kb} KB). "
            f"Use the appropriate specialist agent (excel/word/pdf/ocr) to read it."
        )

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                lines.append(line.rstrip("\n"))
        total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8", errors="replace"))
        header = f"OBSERVATION: {file_path} ({size_kb} KB, {total_lines} total lines) — first {min(num_lines, total_lines)} lines:"
        return header + "\n" + "\n".join(lines)
    except Exception as e:
        return f"OBSERVATION: Error reading {file_path}: {e}"


if __name__ == "__main__":
    fire.Fire(main)
