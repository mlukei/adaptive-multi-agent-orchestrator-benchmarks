"""Convert a PDF file to Word (.docx) via LibreOffice."""
import os
import fire
import shlex
import subprocess

DEMO = (
    "convert a PDF to a Word document: "
    "{'app': 'file_transformer', 'action': 'pdf_to_word', "
    "'source_path': [THE_PATH_TO_THE_PDF_FILE], 'target_path': [THE_PATH_TO_THE_WORD_FILE]}"
)


def construct_action(work_dir, args: dict, py_file_path='/apps/file_transformer_app/ft_pdf_to_word.py'):
    return (
        f'python3 {shlex.quote(py_file_path)} '
        f'--source_path {shlex.quote(str(args["source_path"]))} '
        f'--target_path {shlex.quote(str(args["target_path"]))}'
    )


def pdf_to_word(source_path: str, target_path: str) -> bool:
    try:
        output_dir = os.path.dirname(target_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        subprocess.call(
            ['libreoffice', '--headless', '--infilter=writer_pdf_import',
             '--convert-to', 'docx', source_path, '--outdir', output_dir]
        )
        auto_name = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(source_path))[0] + '.docx',
        )
        if auto_name != target_path and os.path.exists(auto_name):
            os.rename(auto_name, target_path)
        return os.path.exists(target_path)
    except Exception:
        return False


def main(source_path: str, target_path: str) -> str:
    if not os.path.exists(source_path):
        return f"OBSERVATION: {source_path} does not exist. Conversion aborted."
    success = pdf_to_word(source_path, target_path)
    if success:
        return f"OBSERVATION: Successfully converted {source_path} to {target_path}"
    return f"OBSERVATION: Failed to convert {source_path} to {target_path}"


if __name__ == '__main__':
    fire.Fire(main)
