"""LangChain tools for GAIA benchmark agents.

Attribution
-----------
The starting point for this module was the GAIA solution
fisherman611/gaia-agent (https://github.com/fisherman611/gaia-agent),
with its ``agent.py``, ``code_interpreter.py`` and ``image_processing.py``.

Adopted from that repo (kept, lightly adapted):
- ``CodeInterpreter`` class + ``execute_code_multilang``
- the math tools (multiply/add/subtract/divide/modulus/power/square_root)
- ``extract_text_from_image`` (pytesseract OCR), ``analyze_csv_file``,
  ``analyze_excel_file``, ``save_and_read_file``, ``download_file_from_url``

Rewritten (same intent, new implementation):
- ``web_search`` now uses DuckDuckGo (ddgs) instead of paid Tavily
- ``wiki_search`` / ``arxiv_search`` now hit the free Wikipedia/arXiv HTTP APIs
  via httpx instead of langchain_community document loaders

Added by us (not in the upstream repo):
- ``fetch_webpage`` (httpx + HTML stripping)
- ``read_pdf`` / ``read_docx`` / ``read_pptx`` (document text extraction)
- ``analyze_image_with_llm`` (Azure vision model) and ``transcribe_audio`` (local Whisper)
- ``_resolve_path`` (resolves task file paths against GAIA_WORKDIR)
- the WEB_SURFER_TOOLS / FILE_SURFER_TOOLS / CODER_TOOLS exports
"""

from __future__ import annotations

import base64
import logging
import os
import resource
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from html.parser import HTMLParser
from typing import Any, Dict

import httpx
import pandas as pd
from langchain_core.tools import tool
from PIL import Image

logger = logging.getLogger(__name__)


# =============== BROWSER TOOLS ===============


@tool
def web_search(query: str) -> str:
    """Search the web for a query using DuckDuckGo and return up to 5 results.

    Args:
        query: The search query.
    
    Returns:
        A string containing the search results, or an error message if the search fails.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return {"error": "ddgs is not installed. Run: pip install ddgs"}

    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return {"web_results": "No results found."}

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', '')}\n"
                f"    {r.get('body', '')[:500]}"
            )
        return {"web_results": "\n\n".join(parts)}
    except Exception as e:
        logger.exception("Web search failed")
        return {"error": f"Web search failed: {str(e)}"}


@tool
def fetch_webpage(url: str) -> str:
    """Fetch and extract text content from a URL.

    Args:
        url: The URL to fetch.
    """
    if not url:
        return {"error": "URL parameter is required."}

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
    }

    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()

        content = resp.text

        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts: list[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "noscript"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "noscript"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    text = data.strip()
                    if text:
                        self._parts.append(text)

            def get_text(self) -> str:
                return "\n".join(self._parts)

        extractor = _TextExtractor()
        extractor.feed(content)
        text = extractor.get_text()

        if len(text) > 15000:
            text = text[:15000] + "\n\n[... truncated at 15000 chars]"

        return {"page_content": text if text.strip() else "(empty page)"}
    except Exception as e:
        logger.exception("Fetch webpage failed")
        return {"error": f"Failed to fetch {url}: {str(e)}"}


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return up to 2 article summaries.

    Args:
        query: The search query.
    """
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            search = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 2,
                    "format": "json",
                },
                headers={"User-Agent": "thesis-gaia-agent/1.0"},
            )
            search.raise_for_status()
            hits = search.json().get("query", {}).get("search", [])
            if not hits:
                return {"wiki_results": "No Wikipedia results found."}

            parts = []
            for hit in hits:
                title = hit["title"]
                summary = client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
                    headers={"User-Agent": "thesis-gaia-agent/1.0"},
                )
                extract = ""
                if summary.status_code == 200:
                    extract = summary.json().get("extract", "")
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                parts.append(f'<Document source="{url}" title="{title}"/>\n{extract}\n</Document>')

        return {"wiki_results": "\n\n---\n\n".join(parts)}
    except Exception as e:
        logger.exception("Wikipedia search failed")
        return {"error": f"Wikipedia search failed: {str(e)}"}


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for a query and return up to 3 paper abstracts.

    Args:
        query: The search query.
    """
    try:
        import xml.etree.ElementTree as ET

        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "start": 0, "max_results": 3},
                headers={"User-Agent": "thesis-gaia-agent/1.0"},
            )
            resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)
        if not entries:
            return {"arxiv_results": "No arXiv results found."}

        parts = []
        for entry in entries:
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            parts.append(
                f'<Document source="{url}" title="{title}"/>\n{summary[:1000]}\n</Document>'
            )

        return {"arxiv_results": "\n\n---\n\n".join(parts)}
    except Exception as e:
        logger.exception("arXiv search failed")
        return {"error": f"arXiv search failed: {str(e)}"}


@tool
def download_file_from_url(url: str, filename: str = None) -> str:
    """Download a file from a URL to a temporary location.

    Args:
        url: The URL of the file to download.
        filename: Optional filename; if not provided, extracted from URL.

    Returns:
        Path to the downloaded file.
    """
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()

            if not filename:
                filename = url.split("/")[-1] or "downloaded_file"

            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)

            with open(file_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return {"file_path": file_path}
    except Exception as e:
        logger.exception("File download failed")
        return {"error": f"Failed to download from {url}: {str(e)}"}


# =============== FILE READING TOOLS ===============


def _resolve_path(file_path: str) -> str:
    """Resolve a (possibly relative) path against the GAIA working directory.

    run_task.py sets GAIA_WORKDIR to the per-run testbed directory where attached
    files live. Absolute paths are returned as-is; relative paths are resolved
    against GAIA_WORKDIR, falling back to a basename search under it.
    """
    cleaned = str(file_path).strip()
    if not cleaned:
        return cleaned

    workdir = os.environ.get("GAIA_WORKDIR", os.getcwd())

    if os.path.isabs(cleaned):
        return cleaned

    # Strip leading testbed prefixes the orchestrator may pass through.
    for prefix in ("/testbed/data/", "/testbed/", "testbed/data/", "testbed/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    candidate = os.path.join(workdir, cleaned)
    if os.path.isfile(candidate):
        return candidate

    # Fall back to searching by basename under the workdir.
    name = os.path.basename(cleaned)
    for root, _dirs, files in os.walk(workdir):
        if name in files:
            return os.path.join(root, name)
    return candidate


@tool
def read_pdf(file_path: str) -> str:
    """Read and extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.
    """
    if not file_path:
        return {"error": "'file_path' parameter is required."}

    file_path = _resolve_path(file_path)
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        import fitz  # pymupdf

        doc = fitz.open(file_path)
        parts = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                parts.append(f"--- Page {i + 1} ---\n{text}")
        doc.close()

        result = "\n\n".join(parts) if parts else "(no text content found in PDF)"
        if len(result) > 20000:
            result = result[:20000] + "\n\n[... truncated at 20000 chars]"
        return {"pdf_content": result}
    except ImportError:
        return {"error": "pymupdf is not installed. Run: pip install pymupdf"}
    except Exception as e:
        logger.exception("PDF read failed")
        return {"error": f"Failed to read PDF: {str(e)}"}


@tool
def read_docx(file_path: str) -> str:
    """Read and extract text from a DOCX file.

    Args:
        file_path: Path to the DOCX file.
    """
    if not file_path:
        return {"error": "'file_path' parameter is required."}

    file_path = _resolve_path(file_path)
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        result = "\n".join(paragraphs) if paragraphs else "(no text content found)"

        if len(result) > 20000:
            result = result[:20000] + "\n\n[... truncated at 20000 chars]"
        return {"docx_content": result}
    except ImportError:
        return {"error": "python-docx is not installed. Run: pip install python-docx"}
    except Exception as e:
        logger.exception("DOCX read failed")
        return {"error": f"Failed to read DOCX: {str(e)}"}


@tool
def read_pptx(file_path: str) -> str:
    """Read and extract text from a PPTX file.

    Args:
        file_path: Path to the PPTX file.
    """
    if not file_path:
        return {"error": "'file_path' parameter is required."}

    file_path = _resolve_path(file_path)
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        from pptx import Presentation

        prs = Presentation(file_path)
        slides_text = []
        for slide_idx, slide in enumerate(prs.slides, 1):
            slide_content = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_content.append(shape.text)
            if slide_content:
                slides_text.append(f"--- Slide {slide_idx} ---\n" + "\n".join(slide_content))

        result = "\n\n".join(slides_text) if slides_text else "(no text content found)"
        if len(result) > 20000:
            result = result[:20000] + "\n\n[... truncated at 20000 chars]"
        return {"pptx_content": result}
    except ImportError:
        return {"error": "python-pptx is not installed. Run: pip install python-pptx"}
    except Exception as e:
        logger.exception("PPTX read failed")
        return {"error": f"Failed to read PPTX: {str(e)}"}


@tool
def analyze_csv_file(file_path: str, query: str = "") -> str:
    """Analyze a CSV file and return summary statistics.

    Args:
        file_path: Path to the CSV file.
        query: Optional question about the data.
    """
    if not file_path:
        return {"error": "'file_path' parameter is required."}

    file_path = _resolve_path(file_path)
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        df = pd.read_csv(file_path)
        result = f"CSV Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        result += "Columns: " + ", ".join(df.columns) + "\n\n"
        result += "First few rows:\n" + df.head(10).to_string() + "\n\n"
        result += "Summary statistics:\n" + df.describe(include="all").to_string()

        if len(result) > 10000:
            result = result[:10000] + "\n\n[... truncated at 10000 chars]"
        return {"csv_analysis": result}
    except Exception as e:
        logger.exception("CSV analysis failed")
        return {"error": f"Failed to analyze CSV: {str(e)}"}


@tool
def analyze_excel_file(file_path: str, sheet: str = None) -> str:
    """Analyze an Excel file and return summary statistics.

    Args:
        file_path: Path to the Excel file.
        sheet: Optional sheet name.
    """
    if not file_path:
        return {"error": "'file_path' parameter is required."}

    file_path = _resolve_path(file_path)
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        df = pd.read_excel(file_path, sheet_name=sheet or 0)
        result = f"Excel Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        result += "Columns: " + ", ".join(df.columns) + "\n\n"
        result += "First few rows:\n" + df.head(10).to_string() + "\n\n"
        result += "Summary statistics:\n" + df.describe(include="all").to_string()

        if len(result) > 10000:
            result = result[:10000] + "\n\n[... truncated at 10000 chars]"
        return {"excel_analysis": result}
    except Exception as e:
        logger.exception("Excel analysis failed")
        return {"error": f"Failed to analyze Excel: {str(e)}"}


@tool
def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image using OCR (pytesseract).

    Args:
        file_path: Path to the image file.
    """
    if not file_path:
        return {"error": "'file_path' parameter is required."}

    file_path = _resolve_path(file_path)
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        import pytesseract

        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return {"extracted_text": text if text.strip() else "(no text found in image)"}
    except ImportError:
        return {"error": "pytesseract is not installed. Run: pip install pytesseract"}
    except pytesseract.pytesseract.TesseractNotFoundError:
        return {
            "error": (
                "Tesseract OCR is not installed or is not on PATH. "
                "Install the native binary, e.g. on Ubuntu: "
                "sudo apt-get update && sudo apt-get install -y tesseract-ocr"
            )
        }
    except Exception as e:
        logger.exception("Text extraction failed")
        return {"error": f"Failed to extract text from image: {str(e)}"}


@tool
def analyze_image_with_llm(file_path: str, question: str = "") -> str:
    """Analyze an image using a vision-capable LLM (Azure GPT-4o).

    Args:
        file_path: Path to the image file.
        question: What to look for in the image (default: detailed description).
    """
    if not file_path:
        return {"error": "'file_path' parameter is required."}

    file_path = _resolve_path(file_path)
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    question = question or "Describe this image in detail."

    try:
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(ext, "image/png")

        try:
            from openai import AzureOpenAI
        except ImportError:
            return {"error": "openai package is not installed"}

        if not os.environ.get("AZURE_OPENAI_API_KEY") or not os.environ.get("AZURE_OPENAI_ENDPOINT"):
            return {
                "error": "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set"
            }

        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        response = client.chat.completions.create(
            model=os.getenv("GAIA_VISION_MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_data}"},
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        return {"image_analysis": response.choices[0].message.content or "(no response)"}
    except Exception as e:
        logger.exception("Image analysis failed")
        return {"error": f"Failed to analyze image: {str(e)}"}


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe audio using local Whisper.

    Args:
        file_path: Path to the audio file.
    """
    if not file_path:
        return {"error": "'file_path' parameter is required."}

    file_path = _resolve_path(file_path)
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return {"transcription": result.get("text", "(no transcription)")}
    except ImportError:
        return {"error": "local Whisper is not installed. Install openai-whisper."}
    except Exception as e:
        logger.exception("Local Whisper transcription failed")
        return {"error": f"Local Whisper transcription failed: {str(e)}"}


@tool
def save_and_read_file(content: str, filename: str = None) -> str:
    """Save content to a file and return the path.

    Args:
        content: Content to save.
        filename: Optional filename.
    """
    try:
        if not filename:
            filename = f"file_{uuid.uuid4().hex[:8]}.txt"
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
        return {"file_path": file_path}
    except Exception as e:
        logger.exception("Save file failed")
        return {"error": f"Failed to save file: {str(e)}"}


# =============== CODE INTERPRETER ===============


class CodeInterpreter:
    """Multi-language code executor: runs code in resource-limited subprocesses."""

    def __init__(self, max_execution_time=30, working_directory=None):
        self.max_execution_time = max_execution_time
        self.working_directory = working_directory or os.getcwd()
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.temp_sqlite_db = os.path.join(tempfile.gettempdir(), "code_exec.db")
        self._ram_bytes = 2 * 1024 * 1024 * 1024  # 2 GB virtual memory cap

    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code in the specified language."""
        self._sync_working_directory()
        language = language.lower()
        execution_id = str(uuid.uuid4())

        result = {
            "execution_id": execution_id,
            "status": "error",
            "stdout": "",
            "stderr": "",
        }

        if language == "python":
            return self._execute_python(code, result)
        elif language == "bash":
            return self._execute_bash(code, result)
        elif language == "sql":
            return self._execute_sql(code, result)
        elif language == "c":
            return self._execute_c(code, result)
        elif language == "java":
            return self._execute_java(code, result)
        else:
            result["stderr"] = f"Unsupported language: {language}"
            return result

    def _sync_working_directory(self) -> None:
        """Keep execution anchored to the active GAIA task workdir."""
        workdir = os.environ.get("GAIA_WORKDIR") or os.getcwd()
        if workdir != self.working_directory:
            self.working_directory = workdir
            if not os.path.exists(self.working_directory):
                os.makedirs(self.working_directory)

    def _limiter(self) -> object:
        cpu = self.max_execution_time
        ram = self._ram_bytes

        def _set_limits():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_AS, (ram, ram))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

        return _set_limits

    def _execute_python(self, code: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code in a subprocess with CPU/RAM limits."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                cwd=self.working_directory,
                preexec_fn=self._limiter(),
            )
            result["status"] = "success" if proc.returncode == 0 else "error"
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
        except subprocess.TimeoutExpired:
            result["stderr"] = f"Python execution timeout ({self.max_execution_time}s)"
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = f"{type(e).__name__}: {str(e)}"

        return result

    def _execute_bash(self, code: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Bash code."""
        try:
            proc = subprocess.run(
                ["bash", "-c", code],
                cwd=self.working_directory,
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                preexec_fn=self._limiter(),
            )
            result["status"] = "success"
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
        except subprocess.TimeoutExpired:
            result["stderr"] = f"Bash execution timeout ({self.max_execution_time}s)"
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = f"Bash error: {str(e)}"

        return result

    def _execute_sql(self, code: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL code against SQLite."""
        try:
            conn = sqlite3.connect(self.temp_sqlite_db)
            cursor = conn.cursor()
            cursor.executescript(code)
            conn.commit()
            result["status"] = "success"
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = f"SQL error: {str(e)}"
        finally:
            conn.close()

        return result

    def _execute_c(self, code: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compile and execute C code."""
        try:
            temp_file = os.path.join(tempfile.gettempdir(), f"prog_{uuid.uuid4().hex[:8]}.c")
            exe_file = temp_file.replace(".c", "")

            with open(temp_file, "w") as f:
                f.write(code)

            proc = subprocess.run(
                ["gcc", temp_file, "-o", exe_file],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if proc.returncode != 0:
                result["stderr"] = f"Compilation error:\n{proc.stderr}"
                return result

            proc = subprocess.run(
                [exe_file],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                preexec_fn=self._limiter(),
            )

            result["status"] = "success"
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr

            os.remove(temp_file)
            os.remove(exe_file)
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = f"C execution error: {str(e)}"

        return result

    def _execute_java(self, code: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compile and execute Java code."""
        try:
            temp_dir = tempfile.gettempdir()
            import_match = code.find("class ")
            if import_match == -1:
                result["stderr"] = "No class definition found"
                return result

            class_name = code[import_match + 6:].split()[0]
            temp_file = os.path.join(temp_dir, f"{class_name}.java")

            with open(temp_file, "w") as f:
                f.write(code)

            proc = subprocess.run(
                ["javac", temp_file], capture_output=True, text=True, timeout=10
            )

            if proc.returncode != 0:
                result["stderr"] = f"Compilation error:\n{proc.stderr}"
                return result

            proc = subprocess.run(
                ["java", "-cp", temp_dir, class_name],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                preexec_fn=self._limiter(),
            )

            result["status"] = "success"
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr

            os.remove(temp_file)
            os.remove(os.path.join(temp_dir, f"{class_name}.class"))
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = f"Java execution error: {str(e)}"

        return result


interpreter_instance = CodeInterpreter()


@tool
def execute_code_multilang(code: str, language: str = "python") -> str:
    """Execute code in multiple languages (Python, Bash, SQL, C, Java).

    Args:
        code: The source code to execute.
        language: Language ("python", "bash", "sql", "c", "java").
    """
    supported_languages = ["python", "bash", "sql", "c", "java"]
    language = language.lower()

    if language not in supported_languages:
        return f"Unsupported language: {language}. Supported: {', '.join(supported_languages)}"

    result = interpreter_instance.execute_code(code, language=language)
    response = []

    if result["status"] == "success":
        response.append(f"✅ Code executed successfully in {language.upper()}")

        if result.get("stdout"):
            response.append("\n**Standard Output:**\n```\n" + result["stdout"].strip() + "\n```")

        if result.get("stderr"):
            response.append(
                "\n**Standard Error:**\n```\n" + result["stderr"].strip() + "\n```"
            )
    else:
        response.append(f"❌ Code execution failed in {language.upper()}")
        if result.get("stderr"):
            response.append("\n**Error:**\n```\n" + result["stderr"].strip() + "\n```")

    return "\n".join(response)


# =============== MATH TOOLS ===============


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        return float("inf")
    return a / b


@tool
def modulus(a: float, b: float) -> float:
    """Return a mod b."""
    if b == 0:
        return float("nan")
    return a % b


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent


@tool
def square_root(n: float) -> float:
    """Return the square root of n."""
    if n < 0:
        return float("nan")
    return n ** 0.5


# =============== TOOL EXPORTS ===============


WEB_SURFER_TOOLS = [web_search, fetch_webpage, wiki_search, arxiv_search, download_file_from_url]

FILE_SURFER_TOOLS = [
    read_pdf,
    read_docx,
    read_pptx,
    analyze_csv_file,
    analyze_excel_file,
    extract_text_from_image,
    analyze_image_with_llm,
    transcribe_audio,
    save_and_read_file,
]

CODER_TOOLS = [execute_code_multilang, multiply, add, subtract, divide, modulus, power, square_root]
