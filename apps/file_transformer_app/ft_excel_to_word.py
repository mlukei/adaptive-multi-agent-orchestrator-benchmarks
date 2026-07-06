"""Convert an Excel file to Word (.docx) — exports data as a table."""
import os
import fire
import shlex

DEMO = (
    "convert an Excel file to a Word document: "
    "{'app': 'file_transformer', 'action': 'excel_to_word', "
    "'source_path': [THE_PATH_TO_THE_EXCEL_FILE], 'target_path': [THE_PATH_TO_THE_WORD_FILE]}"
)


def construct_action(work_dir, args: dict, py_file_path='/apps/file_transformer_app/ft_excel_to_word.py'):
    return (
        f'python3 {shlex.quote(py_file_path)} '
        f'--source_path {shlex.quote(str(args["source_path"]))} '
        f'--target_path {shlex.quote(str(args["target_path"]))}'
    )


def excel_to_word(source_path: str, target_path: str) -> bool:
    try:
        import openpyxl
        from docx import Document

        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        wb = openpyxl.load_workbook(source_path, data_only=True)
        doc = Document()

        for sheet in wb.worksheets:
            doc.add_heading(sheet.title, level=1)
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            # Find actual number of columns
            col_count = max((len([c for c in row if c is not None]) for row in rows), default=0)
            if col_count == 0:
                continue
            table = doc.add_table(rows=len(rows), cols=col_count)
            table.style = 'Table Grid'
            for r_idx, row in enumerate(rows):
                for c_idx in range(col_count):
                    cell_val = row[c_idx] if c_idx < len(row) else ''
                    table.rows[r_idx].cells[c_idx].text = str(cell_val) if cell_val is not None else ''
            doc.add_paragraph()

        doc.save(target_path)
        return os.path.exists(target_path)
    except Exception:
        return False


def main(source_path: str, target_path: str) -> str:
    if not os.path.exists(source_path):
        return f"OBSERVATION: {source_path} does not exist. Conversion aborted."
    success = excel_to_word(source_path, target_path)
    if success:
        return f"OBSERVATION: Successfully converted {source_path} to {target_path}"
    return f"OBSERVATION: Failed to convert {source_path} to {target_path}"


if __name__ == '__main__':
    fire.Fire(main)
