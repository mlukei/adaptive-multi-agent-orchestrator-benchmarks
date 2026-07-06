# TODO: (zilong) We need to mark the col names and row names in the table. No need to be markdown format. Can be other formats.
# 1. Remove the surrounding blank cells
# 2. Mark the col names and row names in the same way as Excel (col: A, B, C; row: 1, 2, 3)

import os
import shlex

import fire
import openpyxl
from openpyxl.utils.exceptions import InvalidFileException

EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xltx", ".xltm", ".xls")


# DEMO = (
#     'You can write text to a cell in the excel file by calling `excel_read_file` with 2 arguments.\n'
#     '1. The path to the excel file: file_path: str\n'
#     '2. The name of the sheet: sheet_name: str\n'
#     "You can call it by: {'app': 'excel', 'action': 'excel_read_file', 'file_path': ..., 'sheet_name': ...}"
# )

DEMO = (
    "read the excel file to see the existing contents: "
    "{'app': 'excel', 'action': 'read_file', 'file_path': [THE_PATH_TO_THE_EXCEL_FILE]"
)

def construct_action(work_dir, args: dict, py_file_path='/apps/excel_app/excel_read_file.py'):
    # TODO: not sure if we need to specify the file path with the current workdir
    return f"python3 {shlex.quote(py_file_path)} --file_path {shlex.quote(str(args['file_path']))}"


def find_excel_files(directory_path):
    return [
        os.path.join(directory_path, name)
        for name in sorted(os.listdir(directory_path))
        if name.lower().endswith(EXCEL_EXTENSIONS)
    ]


def excel_read_file(file_path, sheet=None):
    try:
        if sheet is None:
            wb = openpyxl.load_workbook(file_path, data_only=True)
            ws = wb.active
        else:
            wb = openpyxl.load_workbook(file_path, data_only=True)
            ws = wb[sheet]
    except InvalidFileException:
        # Fallback for legacy .xls or misnamed files: read via pandas
        import pandas as pd
        try:
            df = pd.read_excel(file_path, sheet_name=(sheet or 0), header=None, engine="xlrd")
        except Exception:
            df = pd.read_excel(file_path, sheet_name=(sheet or 0), header=None)
        content_string = ''
        for row_idx, row in enumerate(df.itertuples(index=False), start=1):
            for col_idx, value in enumerate(row, start=1):
                if value is None or (isinstance(value, float) and __import__('math').isnan(value)):
                    value = '[Empty Cell]'
                content_string += f'({row_idx}, {col_idx}): {value}\t'
            content_string += '\n'
        return content_string

    content_string = ''
    for row in ws.iter_rows():
        for cell in row:
            row_idx = cell.row
            col_idx = cell.column
            value = cell.value
            if value is None:
                value = '[Empty Cell]'
            content_string += f'({row_idx}, {col_idx}): {value}\t'
        content_string += '\n'

    return content_string


def wrap(sheet):
    ob = ("OBSERVATION: The following is the table from the excel file:\n"
         f"{sheet}")
    return ob

def main(file_path, sheet=None, debug=False):
    if not os.path.exists(file_path):
        return f"OBSERVATION: The file {file_path} does not exist. Failed to read the file."

    if os.path.isdir(file_path):
        excel_files = find_excel_files(file_path)
        if excel_files:
            return (
                f"OBSERVATION: {file_path} is a directory, not an Excel file. "
                "Choose one of these Excel files instead:\n"
                + "\n".join(excel_files)
            )
        return (
            f"OBSERVATION: {file_path} is a directory, not an Excel file, "
            "and it contains no Excel files."
        )

    contents = excel_read_file(file_path, sheet)
    if debug:
        print(contents)
    observation = wrap(contents)
    return observation


if __name__ == '__main__':
    fire.Fire(main)
