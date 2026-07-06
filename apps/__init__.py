from . import excel_app
from . import ocr_app
from . import pdf_app
from . import shell_app
from . import word_app
from . import calendar_app
from . import email_app
from . import llm_app
from . import file_transformer_app
from . import env_explorer_app
from . import image_creator_app


AVAILABLE_APPS = {
    'calendar': calendar_app,
    'excel': excel_app,
    'ocr': ocr_app,
    'pdf': pdf_app,
    'shell': shell_app,
    'word': word_app,
    'email': email_app,
    'llm': llm_app,
    'file_transformer': file_transformer_app,
    'env_explorer': env_explorer_app,
    'image_creator': image_creator_app,
}

AVAILABLE_AGENTS_INTRO = {
    "calendar": (
        "Calendar specialist: use for scheduling tasks. Can list events, create new events, and delete events. "
        "Choose this agent when the task involves dates, meetings, appointments, or calendar availability."
    ),
    "excel": (
        "Excel specialist: use for spreadsheet tasks. Can read Excel files, set or delete cells, create new Excel files, "
        "and convert Excel to PDF. Choose this agent for calculations, tables, structured data editing, or producing a PDF from Excel."
    ),
    "shell": (
        "Shell specialist: use for file discovery and OS-level commands. Can run shell commands (e.g., list directories, find files) "
        "especially in /testbed/data. Choose this agent when filenames are unknown or you need to inspect the filesystem."
    ),
    "word": (
        "Word specialist: use for creating and editing Word documents. Can create new files, read Word files, write text into Word files, "
        "and convert Word to PDF. Choose this agent for drafting documents, formatting text content, or producing a PDF from Word."
    ),
    "email": (
        "Email specialist: use for email workflows. Can list emails, read specific emails, and send emails. "
        "Choose this agent when the task involves communication, replying/summarizing messages, or sending results via email."
    ),
    "llm": (
        "LLM tool specialist: use for pure text transformation tasks via the 'llm' app (complete_text). "
        "Choose this agent when you need rewriting, summarization, classification, or generating text that does not require other apps/tools."
    ),
    "ocr": (
        "OCR specialist: use to extract text from images or scanned documents. Can run OCR on a file (recognize_file). "
        "Choose this agent when the input is an image/PDF page screenshot/scan and text must be recognized before further processing."
    ),
    "pdf": (
        "PDF specialist: use for PDF reading and conversion. Can read PDF files, convert PDFs to images, and convert PDFs to Word documents. "
        "Choose this agent when the task involves extracting content from PDFs or converting PDFs into editable formats."
    ),
    "file_transformer": (
        "File Transformer specialist: use for converting documents between formats. "
        "Can convert Excel to PDF, Excel to Word, Word to PDF, PDF to Word, and PDF to image. "
        "Choose this agent when the task requires cross-format document conversion "
        "(e.g. export a spreadsheet as a Word table, or turn a report into PDF)."
    ),
    "env_explorer": (
        "Environment Explorer specialist: use FIRST to scan the /testbed workspace before other agents act. "
        "Reports all files in /testbed/data, emails per user in /testbed/emails, and calendar entries "
        "in /testbed/calendar. Also previews text files. Choose this agent when you need to understand "
        "what resources exist before planning a multi-step task."
    ),
    "image_creator": (
        "Image Creator specialist: use to generate image files (.jpg/.png) from text or data content. "
        "Renders the provided content as a visual image. Choose this agent when the task requires creating "
        "an image post, poster, contact card, report image, or any visual output file."
    ),
}


AVAILABLE_ACTIONS = {
    'calendar': {
        'create_event': calendar_app.calendar_create_event,
        'delete_event': calendar_app.calendar_delete_event,
        'list_events': calendar_app.calendar_list_events,
    },
    'excel': {
        # 'add_column': excel_app.excel_add_column,
        # 'add_row': excel_app.excel_add_row,
        # 'delete_cell': excel_app.excel_delete_cell,
        # 'delete_column': excel_app.excel_delete_column,
        # 'delete_row': excel_app.excel_delete_row,
        'read_file': excel_app.excel_read_file,
        'set_cell': excel_app.excel_set_cell,
        'delete_cell': excel_app.excel_delete_cell,
        'create_new_file': excel_app.excel_create_new_file,
        'convert_to_pdf': excel_app.excel_convert_to_pdf,
        # 'write_column': excel_app.excel_write_column,
        # 'write_row': excel_app.excel_write_row,
    },
    'ocr': {
        'recognize_file': ocr_app.ocr_recognize_file
    },
    'pdf': {
        'convert_to_image': pdf_app.pdf_convert_to_image,
        'convert_to_word': pdf_app.pdf_convert_to_word,
        'read_file': pdf_app.pdf_read_file,
    },
    'email': {
        'send_email': email_app.email_send_email,
        'list_emails': email_app.email_list_emails,
        'read_email': email_app.email_read_email,
    },
    'shell': {
        'command': shell_app.command,
    },
    'word': {
        'convert_to_pdf': word_app.word_convert_to_pdf,
        'create_new_file': word_app.word_create_new_file,
        'read_file': word_app.word_read_file,
        'write_to_file': word_app.word_write_to_file,
    },
    'llm': {
        'complete_text': llm_app.llm_query
    },
    'file_transformer': {
        'excel_to_pdf':  file_transformer_app.ft_excel_to_pdf,
        'excel_to_word': file_transformer_app.ft_excel_to_word,
        'word_to_pdf':   file_transformer_app.ft_word_to_pdf,
        'pdf_to_word':   file_transformer_app.ft_pdf_to_word,
        'pdf_to_image':  file_transformer_app.ft_pdf_to_image,
    },
    'env_explorer': {
        'scan_testbed':   env_explorer_app.scan_testbed,
        'read_snippet':   env_explorer_app.read_snippet,
    },
    'image_creator': {
        'create_image': image_creator_app.create_image,
    },
}
