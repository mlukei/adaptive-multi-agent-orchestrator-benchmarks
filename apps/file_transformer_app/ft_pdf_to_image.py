"""Convert the first page of a PDF file to an image."""
import os
import shlex

import fire
import fitz

DEMO = (
    "convert the first page of a PDF to an image: "
    "{'app': 'file_transformer', 'action': 'pdf_to_image', "
    "'source_path': [THE_PATH_TO_THE_PDF_FILE], 'target_path': [THE_PATH_TO_THE_IMAGE_FILE]}"
)


def construct_action(work_dir, args: dict, py_file_path='/apps/file_transformer_app/ft_pdf_to_image.py'):
    return (
        f'python3 {shlex.quote(py_file_path)} '
        f'--source_path {shlex.quote(str(args["source_path"]))} '
        f'--target_path {shlex.quote(str(args["target_path"]))}'
    )


def pdf_to_image(source_path: str, target_path: str) -> bool:
    try:
        output_dir = os.path.dirname(target_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        with fitz.open(source_path) as doc:
            if doc.page_count == 0:
                return False
            page = doc.load_page(0)
            pix = page.get_pixmap()
            pix.save(target_path)
        return os.path.exists(target_path)
    except Exception:
        return False


def main(source_path: str, target_path: str) -> str:
    if not os.path.exists(source_path):
        return f"OBSERVATION: {source_path} does not exist. Conversion aborted."
    success = pdf_to_image(source_path, target_path)
    if success:
        return f"OBSERVATION: Successfully converted {source_path} to {target_path}"
    return f"OBSERVATION: Failed to convert {source_path} to {target_path}"


if __name__ == '__main__':
    fire.Fire(main)
