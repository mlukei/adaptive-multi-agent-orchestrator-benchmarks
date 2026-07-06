"""Create an image file (.jpg / .png) from text content.

Internally the tool writes the content into a temporary Word document,
converts it to PDF, then renders the first page as an image.  This keeps
all dependencies that are already available in the project (python-docx,
PyMuPDF / fitz) and avoids pulling in heavyweight image-generation
libraries.
"""

import os
import shlex
import subprocess
import tempfile

import fire
import fitz
from docx import Document

DEMO = (
    "create an image file from text content: "
    "{'app': 'image_creator', 'action': 'create_image', "
    "'content': [THE_TEXT_CONTENT_TO_RENDER], "
    "'target_path': [THE_PATH_TO_THE_OUTPUT_IMAGE_FILE]}"
)


def construct_action(
    work_dir,
    args: dict,
    py_file_path="/apps/image_creator_app/create_image.py",
):
    return (
        f"python3 {shlex.quote(py_file_path)} "
        f"--content {shlex.quote(str(args['content']))} "
        f"--target_path {shlex.quote(str(args['target_path']))}"
    )


def _content_to_image(content: str, target_path: str) -> bool:
    """Render *content* as a Word page, convert to PDF, then to image."""
    tmp_dir = tempfile.mkdtemp()
    docx_path = os.path.join(tmp_dir, "tmp.docx")
    pdf_path = os.path.join(tmp_dir, "tmp.pdf")

    try:
        # 1. Create a Word document with the content
        doc = Document()
        for paragraph in content.split("\n"):
            doc.add_paragraph(paragraph)
        doc.save(docx_path)

        # 2. Convert docx → pdf via libreoffice headless
        subprocess.call(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             docx_path, "--outdir", tmp_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not os.path.exists(pdf_path):
            return False

        # 3. Render first PDF page as image
        output_dir = os.path.dirname(target_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        with fitz.open(pdf_path) as pdf_doc:
            if pdf_doc.page_count == 0:
                return False
            page = pdf_doc.load_page(0)
            pix = page.get_pixmap(dpi=150)
            pix.save(target_path)

        return os.path.exists(target_path)
    except Exception:
        return False
    finally:
        for f in [docx_path, pdf_path]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(tmp_dir):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass


def main(content: str, target_path: str) -> str:
    success = _content_to_image(content, target_path)
    if success:
        return f"OBSERVATION: Successfully created image {target_path}"
    return f"OBSERVATION: Failed to create image {target_path}"


if __name__ == "__main__":
    fire.Fire(main)
