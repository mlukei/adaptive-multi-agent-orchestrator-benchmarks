"""Evaluation functions for OfficeBench.

Implementation notes:
- The public ``evaluate_*`` functions are the OfficeBench evaluator surface
  originally implemented in ``utils/evaluate.py``.
- Blocks marked ``NOTE: Modification`` are local changes added for this
  benchmark setup, mainly to make evaluation robust
"""
import difflib
import glob
import logging
import os
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

import icalendar
import openpyxl
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from apps.excel_app import excel_read_file
from apps.word_app import word_read_file
from apps.pdf_app import pdf_read_file
from apps.email_app import email_list_emails

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Original OfficeBench primitive.
# ---------------------------------------------------------------------------
def _is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# NOTE: Modification - flexible text matching.
#
# Original OfficeBench only lowercased strings, stripped commas for numeric
# keywords, and required exact substring containment. The helpers in this
# section keep that exact check but add casefold/separator normalization and a
# TF-IDF fallback for minor differences.
# ---------------------------------------------------------------------------
_SOFT_MATCH_THRESHOLD = 0.35


def _normalize_for_match(text):
    """Case-fold and collapse whitespace/punctuation for flexible matching."""
    text = str(text).casefold()
    # Replace common separators with spaces
    text = re.sub(r'[-_\t]+', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _soft_contains(content: str, keyword: str) -> bool:
    """TF-IDF cosine similarity fallback for keyword matching.

    Splits content into sentence-like chunks and checks whether any chunk
    is semantically close enough to keyword. 
    --> Works well for plural forms, minor rewordings, and formatting differences
    """
    # Split content into overlapping windows
    chunks = re.split(r'[.\n;:!?]+', content)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 1]
    if not chunks:
        return False

    try:
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',   # character n-grams
            ngram_range=(2, 4),
            max_features=5000,
        )
        corpus = chunks + [keyword]
        tfidf = vectorizer.fit_transform(corpus)
        # Compare keyword vector (last) against all chunks
        sims = cosine_similarity(tfidf[-1:], tfidf[:-1]).flatten()
        best = float(sims.max())
        if best >= _SOFT_MATCH_THRESHOLD:
            logger.debug(
                "Soft match: keyword=%r  best_score=%.3f  chunk=%r",
                keyword, best, chunks[int(sims.argmax())],
            )
            return True
    except ValueError:
        pass
    return False


# ---------------------------------------------------------------------------
# NOTE: Modification - flexible Excel value matching.
#
# Original OfficeBench evaluated cell values by searching the serialized Excel
# text for an exact ``(row, col): value`` pattern. These helpers read workbook
# cells directly and normalize common numeric/time/string variants.
# ---------------------------------------------------------------------------
def _normalize_cell_value(value):
    """Normalize a cell value for flexible comparison (int/float/str)."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value.strip()
    return value


def _parse_numeric_like(value):
    """Parse loosely formatted numeric values such as '$1,200,000'."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None

    text = str(value).strip()
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    cleaned = (
        text.replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("¥", "")
        .strip()
    )
    if negative:
        cleaned = f"-{cleaned}"

    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _excel_values_match(expected, actual):
    if actual == expected:
        return True

    # None actual never matches a non-None expected (and vice versa)
    if actual is None or expected is None:
        return False

    expected_text = str(expected).strip()
    actual_text = str(actual).strip()
    if actual_text.casefold() == expected_text.casefold():
        return True

    expected_number = _parse_numeric_like(expected)
    actual_number = _parse_numeric_like(actual)
    if expected_number is not None and actual_number is not None:
        return expected_number == actual_number

    # Partial match: one value starts with the other (e.g. "8:00 AM" matches "8:00")
    # Guard against empty strings — they are a prefix of everything
    if actual_text and expected_text:
        if actual_text.casefold().startswith(expected_text.casefold()):
            return True
        if expected_text.casefold().startswith(actual_text.casefold()):
            return True

    # Normalize leading zeros in time strings (e.g. "08:00" == "8:00")
    def _strip_leading_zeros(s):
        return re.sub(r'\b0+(\d)', r'\1', s)

    norm_actual = _strip_leading_zeros(actual_text).casefold()
    norm_expected = _strip_leading_zeros(expected_text).casefold()
    if norm_actual == norm_expected:
        return True
    if norm_actual and norm_expected:
        if norm_actual.startswith(norm_expected):
            return True
        if norm_expected.startswith(norm_actual):
            return True

    return False


# ---------------------------------------------------------------------------
# Original OfficeBench containment helper, extended with the text matching
# modification above.
# ---------------------------------------------------------------------------
def _evaluate_contain_text(content, args):
    raw_content = _normalize_for_match(content)
    for keyword in args['keywords']:
        kw = _normalize_for_match(keyword)

        # --- Numeric normalization (strip commas) ---
        check_content = raw_content
        check_kw = kw
        if _is_number(kw):
            check_content = check_content.replace(',', '')
            check_kw = check_kw.replace(',', '')

        # 1) Exact substring (original behaviour)
        if check_kw in check_content:
            continue

        # 2) TF-IDF soft match as fallback
        if _soft_contains(check_content, check_kw):
            continue

        logger.debug("Keyword not found (exact+soft): %r", keyword)
        return False
    return True


def evaluate_contain(testbed_dir, args):
    """Original OfficeBench evaluator: check whether a document contains keywords.

    NOTE: Thesis modification:
    - email account fallback now resolves accounts case-insensitively;
    - file lookup uses fuzzy path resolution instead of exact path only;
    - text matching delegates to the extended containment helper above.
    """
    type = args['doc_type']
    if type == 'email':
        username = args['username']
        email_contents = ''
        if os.path.exists(os.path.join(testbed_dir, 'emails', username)):
            email_contents = email_list_emails.list_emails(username, testbed_dir, -1)
        elif os.path.exists(os.path.join(testbed_dir, 'emails', username.lower())):
            email_contents = email_list_emails.list_emails(username.lower(), testbed_dir, -1)
        else:
            email_accounts = glob.glob(os.path.join(testbed_dir, 'emails', '*'))
            for email_account in email_accounts:
                account_name = os.path.basename(email_account)
                if username.lower() in account_name.lower():
                    email_contents = email_list_emails.list_emails(account_name, testbed_dir, -1)
                    break
        return _evaluate_contain_text(email_contents, args)
    
    file_path = _resolve_fuzzy_path(testbed_dir, args['file'])
    if file_path is None or not os.path.isfile(file_path):
        logger.warning('File not found (exact+fuzzy): %s', args['file'])
        return False
    if type == 'xlsx':
        helper = excel_read_file.excel_read_file
    elif type == 'txt' or type == 'ics':
        helper = lambda x: open(x).read()
    elif type == 'doc' or type == 'docx':
        helper = word_read_file.word_read_file_into_string
    elif type == 'pdf':
        helper = pdf_read_file.read_pdf_to_string
    else:
        assert False, f'Not implemented doc type: {type}'
    content = helper(file_path)
    return _evaluate_contain_text(content, args)


def evaluate_not_contain(testbed_dir, args):
    """Original OfficeBench evaluator: negated containment check."""
    return not evaluate_contain(testbed_dir, args)


# ---------------------------------------------------------------------------
# NOTE: Thesis modification - fuzzy path resolution.
#
# Original OfficeBench used exact ``os.path.exists(os.path.join(...))`` checks.
# The helpers in this section keep exact paths as the first tier, then accept
# case/separator-insensitive names and unique stem-prefix variants such as
# ``report_updated.xlsx`` for ``report.xlsx``.
# ---------------------------------------------------------------------------
def _split_rel_path(rel_path: str) -> tuple[list[str], str]:
    """Split a relative path into (dir_parts, filename), stripping a leading ./ or .\\."""
    if rel_path.startswith('./') or rel_path.startswith('.\\'):
        rel_path = rel_path[2:]
    parts = rel_path.replace('\\', '/').split('/')
    return parts[:-1], parts[-1]


def _resolve_parent_dir(testbed_dir: str, dir_parts: list[str]) -> str | None:
    """Resolve parent directory components case/separator-insensitively.

    Returns the directory path, or None if any component cannot be matched.
    """
    current = testbed_dir
    for part in dir_parts:
        target_norm = _normalize_filename(part)
        try:
            entries = os.listdir(current)
        except OSError:
            return None
        match = next((e for e in entries if _normalize_filename(e) == target_norm), None)
        if match is None:
            return None
        current = os.path.join(current, match)
    return current


def _stem_prefix_candidates(directory: str, filename: str) -> list[str]:
    """Return files in *directory* with the same extension whose stem begins with
    (and is strictly longer than) the expected stem — e.g. ``report_updated.xlsx``
    when ``report.xlsx`` was expected.
    """
    expected_stem, expected_ext = os.path.splitext(filename)
    expected_stem_norm = _normalize_filename(expected_stem)
    expected_ext_lower = expected_ext.lower()
    if not expected_stem_norm:
        return []
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    candidates = []
    for entry in entries:
        entry_stem, entry_ext = os.path.splitext(entry)
        if entry_ext.lower() != expected_ext_lower:
            continue
        entry_stem_norm = _normalize_filename(entry_stem)
        if (
            entry_stem_norm.startswith(expected_stem_norm)
            and len(entry_stem_norm) > len(expected_stem_norm)
        ):
            candidates.append(os.path.join(directory, entry))
    return candidates


def _resolve_fuzzy_path(testbed_dir: str, rel_path: str) -> str | None:
    """Resolve a relative path against testbed_dir, with two fallback tiers.

    Tier 1 - exact path.
    Tier 2 - case/separator-insensitive match for every path component.
    Tier 3 - stem-prefix fallback: if the resolved parent directory exists but
              the expected filename is missing, accept the *unique* file that
              shares the same extension and whose stem begins with the expected
              stem (e.g. ``budget_updated.xlsx`` when ``budget.xlsx`` was
              expected). Requires exactly one such candidate to avoid false
              positives.

    Returns the absolute path if found, or None.
    """
    exact = os.path.join(testbed_dir, rel_path)
    if os.path.exists(exact):
        return exact

    dir_parts, filename = _split_rel_path(rel_path)
    current = _resolve_parent_dir(testbed_dir, dir_parts)
    if current is None:
        return None

    # --- Tier 2: exact normalized filename match ---
    target_norm = _normalize_filename(filename)
    try:
        entries = os.listdir(current)
    except OSError:
        return None

    exact_match = next((e for e in entries if _normalize_filename(e) == target_norm), None)
    if exact_match is not None:
        result = os.path.join(current, exact_match)
        if os.path.exists(result):
            logger.debug(
                "Fuzzy path resolved (tier-2): expected=%r  found=%r",
                rel_path, os.path.relpath(result, testbed_dir),
            )
            return result

    # --- Tier 3: unique stem-prefix sibling ---
    candidates = _stem_prefix_candidates(current, filename)
    if len(candidates) == 1:
        found = candidates[0]
        logger.warning(
            "Stem-prefix fallback (tier-3): expected=%r  found=%r — "
            "agent wrote to a renamed file; accepting for evaluation",
            rel_path, os.path.relpath(found, testbed_dir),
        )
        return found

    return None


def _normalize_filename(name: str) -> str:
    """Normalize a filename for fuzzy comparison: lowercase, strip separators."""
    return re.sub(r'[-_ ]+', '', name.casefold())


def _fuzzy_file_exists(testbed_dir: str, rel_path: str) -> bool:
    """Check if a file exists, falling back to case/separator-insensitive match."""
    return _resolve_fuzzy_path(testbed_dir, rel_path) is not None


def evaluate_file_exist(testbed_dir, args):
    """Original OfficeBench evaluator, with fuzzy path lookup added."""
    file_path = args['file']
    return _fuzzy_file_exists(testbed_dir, file_path)

def evaluate_file_not_exist(testbed_dir, args):
    """Original OfficeBench evaluator, with fuzzy path lookup added."""
    file_path = args['file']
    return not _fuzzy_file_exists(testbed_dir, file_path)

def _helper_diff_contain_text(input_content, output_content, args):
    """Original OfficeBench diff helper, with normalized keyword matching added."""
    if input_content == output_content:
        return False
    else:
        diff = difflib.unified_diff(input_content.split('\n'), output_content.split('\n'), n = 0)
        diff = '\n'.join(list(diff))
        diff = _normalize_for_match(diff)
        for matches in args['keywords']:
            matches = _normalize_for_match(matches)
            if matches not in diff:
                return False
    return True

def evaluate_diff_contain_text(testbed_dir, args):
    """Original OfficeBench evaluator: check whether a file diff contains keywords.

    NOTE: Thesis modification: keyword matching in the diff is normalized by
    ``_helper_diff_contain_text``.
    """
    type = args['doc_type']
    input_file = os.path.join(testbed_dir, args['input_file'])
    output_file = os.path.join(testbed_dir, args['output_file'])
    if type == 'xlsx':
        helper = excel_read_file.excel_read_file
    elif type == 'doc':
        helper = word_read_file.word_read_file
    else:
        assert False, f'Not implemented doc type: {type}'
    input_content = helper(input_file)
    output_content = helper(output_file)
    return _helper_diff_contain_text(input_content, output_content, args)

def _check_excel_cell_matches(file_path: str, matches: list) -> bool:
    """Return True if all cell matches pass for the given workbook file."""
    try:
        workbook = openpyxl.load_workbook(file_path, data_only=True)
    except Exception as exc:
        logger.warning("Could not open workbook %s: %s", file_path, exc)
        return False
    sheet = workbook.active
    for match in matches:
        actual_value = sheet.cell(
            row=int(match["row"]),
            column=int(match["col"]),
        ).value
        if not _excel_values_match(match["value"], actual_value):
            return False
    return True


def _find_stem_prefix_variants(testbed_dir: str, rel_path: str) -> list[str]:
    """Return all stem-prefix siblings of *rel_path* in its directory.

    Unlike Tier 3 in ``_resolve_fuzzy_path``, this always runs — even when
    the expected file itself exists — so it can be used as a content fallback.
    """
    dir_parts, filename = _split_rel_path(rel_path)
    current = _resolve_parent_dir(testbed_dir, dir_parts)
    if current is None:
        return []
    return _stem_prefix_candidates(current, filename)


def evaluate_excel_cell_value(testbed_dir, args):
    """Original OfficeBench evaluator, rewritten to inspect workbook cells.

    NOTE: Thesis modification:
    - fuzzy file lookup is used;
    - cell values are compared with Excel-aware normalization;
    - if the expected file exists but fails, a unique renamed stem-prefix copy
      may be accepted when its cell contents match.
    """
    file_path = _resolve_fuzzy_path(testbed_dir, args['file'])
    if file_path is None or not os.path.isfile(file_path):
        logger.warning('File not found (exact+fuzzy): %s', args['file'])
        return False

    if _check_excel_cell_matches(file_path, args['matches']):
        return True

    # Content fallback: expected file exists but cell values don't match.
    # The agent may have written to a renamed copy (e.g. budget_updated.xlsx).
    # Accept the unique stem-prefix variant if it passes.
    variants = _find_stem_prefix_variants(testbed_dir, args['file'])
    if len(variants) == 1:
        alt = variants[0]
        if _check_excel_cell_matches(alt, args['matches']):
            logger.warning(
                "Excel cell content fallback: expected=%r failed, "
                "accepted stem-prefix variant=%r",
                args['file'], os.path.relpath(alt, testbed_dir),
            )
            return True

    return False

def evaluate_excel_cell_comparator(testbed_dir, args):
    """Original OfficeBench evaluator, with fuzzy file lookup added."""
    file_path = _resolve_fuzzy_path(testbed_dir, args['file'])
    if file_path is None or not os.path.isfile(file_path):
        logger.warning('File not found (exact+fuzzy): %s', args['file'])
        return False
    content = excel_read_file.excel_read_file(file_path)
    # Match the '(row, col): value' format
    for match in args['matches']:
        # regex match pattern (row, col): ***\n
        pattern = r'\({}, {}\): (\w+)\t'.format(match["row"], match["col"])
        x = re.search(pattern, content)
        if x:
            value = x.group(1)
            if eval(match['comparator'])(value):
                continue
            else:
                return False
        else:
            return False
    return True

def evaluate_calendar_no_overlap(testbed_dir, args):
    """Original OfficeBench evaluator: check whether a calendar has overlaps."""
    username = args['username']
    calendar_file = f'{testbed_dir}/calendar/{username}.ics'
    calendar = icalendar.Calendar.from_ical(open(calendar_file, 'rb').read())
    # sort events by start time

    utc=pytz.UTC
    def is_naive(dt):
        return dt.tzinfo is None
    def proc_dt(dt):
        if is_naive(dt):
            return utc.localize(dt)
        else:
            return dt
    calendar.subcomponents.sort(key=lambda x: proc_dt(x.get('dtstart').dt))
    events = []
    for component in calendar.walk():
        if component.name == "VEVENT":
            events.append(component)
    for i in range(len(events)-1):
        if proc_dt(events[i].get('dtend').dt) > proc_dt(events[i+1].get('dtstart').dt):
            return False
    return True

def evaluate_exact_match(testbed_dir, args):
    """Original OfficeBench evaluator: exact document/workbook comparison.

    NOTE: Thesis modification:
    - print diagnostics were replaced with logger calls;
    - xlsx checks support optional ``match_mode='unordered_columns'`` for
      column-wise multiset equality instead of strict cell-by-cell order.
    """
    result_path = os.path.join(testbed_dir, args['result_file'])
    if not os.path.exists(result_path):
        logger.warning('File does not exist: %s', result_path)
        return False

    expected_path = os.path.join(testbed_dir, args['expected_file'])
    type = args['doc_type']
    if type != 'xlsx':
        if type == 'txt' or type == 'ics':
            helper = lambda x: open(x).read()
        elif type == 'doc':
            helper = word_read_file.word_read_file
        elif type == 'pdf':
            helper = pdf_read_file.read_pdf
        else:
            assert False, f'Not implemented doc type: {type}'
        result_content = helper(result_path)
        expected_content = helper(expected_path)
        if result_content != expected_content:
            logger.info('Exact-match content mismatch')
            logger.debug('result=%r', result_content)
            logger.debug('expected=%r', expected_content)
            return False
        return True
    else:
        result_sheet = openpyxl.load_workbook(result_path).active
        expected_sheet = openpyxl.load_workbook(expected_path).active

        match_mode = args.get('match_mode', 'strict')
        if match_mode == 'unordered_columns':
            # Generous mode: check that each column has the same multiset of values,
            # regardless of row order. Accepts both whole-row swaps and value-only swaps.
            max_col = max(result_sheet.max_column, expected_sheet.max_column)
            max_row = max(result_sheet.max_row, expected_sheet.max_row)
            for col in range(1, max_col + 1):
                result_vals = Counter(_normalize_cell_value(result_sheet.cell(row=r, column=col).value) for r in range(1, max_row + 1))
                expected_vals = Counter(_normalize_cell_value(expected_sheet.cell(row=r, column=col).value) for r in range(1, max_row + 1))
                if result_vals != expected_vals:
                    return False
            return True

        for row in result_sheet.iter_rows():
            for cell in row:
                row_idx = cell.row
                col_idx = cell.column
                result_value = cell.value
                expected_value = expected_sheet.cell(row=row_idx, column=col_idx).value
                if result_value != expected_value:
                    return False
                
        for row in expected_sheet.iter_rows():
            for cell in row:
                row_idx = cell.row
                col_idx = cell.column
                result_value = result_sheet.cell(row=row_idx, column=col_idx).value
                expected_value = cell.value
                if result_value != expected_value:
                    return False
        
        return True
