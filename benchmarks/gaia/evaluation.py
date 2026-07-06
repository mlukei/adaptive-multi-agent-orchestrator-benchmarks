"""GAIA answer-match evaluation.

Scores the agent's ``answer.txt`` against the expected GAIA answer using the
official GAIA scorer plus extraction/containment fallbacks. (Extracted from the
former ``utils/evaluate.py`` GAIA section.)
"""

from __future__ import annotations

import logging
import os
import re
import string as _string

logger = logging.getLogger(__name__)


def evaluate_answer_match(testbed_dir: str, expected: str) -> bool:
    """Check whether the agent's answer matches the expected GAIA answer.
    """
    answer_path = os.path.join(testbed_dir, "answer.txt")
    if not os.path.isfile(answer_path):
        logger.warning("Answer file not found: %s", answer_path)
        return False

    with open(answer_path, encoding="utf-8") as f:
        raw_answer = f.read().strip()

    if not raw_answer:
        return False

    # Layer 1: official GAIA scorer (exact match)
    if _gaia_question_scorer(raw_answer, expected):
        return True

    # Layer 2: extraction heuristic + re-score
    extracted = _gaia_extract_answer(raw_answer, expected)
    if extracted and extracted != raw_answer:
        if _gaia_question_scorer(extracted, expected):
            logger.info("Matched after answer extraction: '%s'", extracted)
            return True

    # Layer 3: containment check
    if _gaia_contains_expected(raw_answer, expected):
        logger.info("Matched via containment check for expected='%s'", expected)
        return True

    return False


def _gaia_question_scorer(model_answer: str, ground_truth: str) -> bool:
    """Score a model answer against ground truth using the official GAIA logic."""
    if model_answer is None:
        model_answer = "None"

    # Numeric ground truth
    if _gaia_is_float(ground_truth):
        normalized = _gaia_normalize_number_str(model_answer)
        return normalized == float(ground_truth)

    # List ground truth
    if any(c in ground_truth for c in [",", ";"]):
        gt_elems = _gaia_split_string(ground_truth)
        ma_elems = _gaia_split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _gaia_is_float(gt_elem):
                comparisons.append(
                    _gaia_normalize_number_str(ma_elem) == float(gt_elem)
                )
            else:
                comparisons.append(
                    _gaia_normalize_str(ma_elem, remove_punct=False)
                    == _gaia_normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # String ground truth
    return _gaia_normalize_str(model_answer) == _gaia_normalize_str(ground_truth)


def _gaia_is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def _gaia_normalize_number_str(number_str: str) -> float:
    """Strip currency/percent/comma symbols and parse as float."""
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def _gaia_split_string(s: str, char_list: list | None = None) -> list:
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return [x.strip() for x in re.split(pattern, s)]


def _gaia_normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Official GAIA string normalization."""
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", _string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def _gaia_extract_answer(raw: str, expected: str) -> str | None:
    """Try to extract a concise answer from a verbose model response."""
    # Strategy 1: numeric expected → find matching number
    if _gaia_is_float(expected):
        target = float(expected)
        numbers = re.findall(r"[\$]?[\d,]+\.?\d*[%]?", raw)
        for num_str in reversed(numbers):
            if _gaia_normalize_number_str(num_str) == target:
                return num_str
        if numbers:
            return numbers[-1]

    # Strategy 2: explicit answer patterns
    patterns = [
        r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:=])\s*(.+)",
        r"(?:result\s*[:=]|therefore)\s*(.+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")

    # Strategy 3: last non-empty line
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if len(lines) > 1:
        return lines[-1]

    return None


def _gaia_contains_expected(raw: str, expected: str) -> bool:
    """Check whether the expected answer is contained within the model output."""
    cleaned = re.sub(r"\*\*|__|\*|_|`", "", raw)

    if _gaia_is_float(expected):
        return False

    if any(c in expected for c in [",", ";"]):
        gt_elems = _gaia_split_string(expected)
        if any(len(e.strip()) < 3 for e in gt_elems):
            return False
        lower_cleaned = cleaned.lower()
        last_pos = -1
        for elem in gt_elems:
            norm_elem = elem.strip().lower()
            pos = lower_cleaned.find(norm_elem, last_pos + 1)
            if pos == -1:
                return False
            last_pos = pos
        return True

    norm_expected = expected.strip().lower()
    if len(norm_expected) < 3:
        return False
    norm_raw = cleaned.lower()
    return norm_expected in norm_raw
