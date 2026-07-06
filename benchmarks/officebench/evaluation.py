"""Task evaluation logic for OfficeBench benchmark tasks."""

import json
import logging
import os
import shutil

from benchmarks.officebench.evaluators import (
    evaluate_contain,
    evaluate_not_contain,
    evaluate_file_exist,
    evaluate_file_not_exist,
    evaluate_diff_contain_text,
    evaluate_excel_cell_value,
    evaluate_excel_cell_comparator,
    evaluate_exact_match,
    evaluate_calendar_no_overlap,
)

logger = logging.getLogger(__name__)

EVAL_FUNCTIONS = {
    'evaluate_contain': evaluate_contain,
    'evaluate_not_contain': evaluate_not_contain,
    'evaluate_file_exist': evaluate_file_exist,
    'evaluate_file_not_exist': evaluate_file_not_exist,
    'evaluate_diff_contain_text': evaluate_diff_contain_text,
    'evaluate_excel_cell_value': evaluate_excel_cell_value,
    'evaluate_excel_cell_comparator': evaluate_excel_cell_comparator,
    'evaluate_exact_match': evaluate_exact_match,
    'evaluate_calendar_no_overlap': evaluate_calendar_no_overlap,
}


def run_task_evaluation(task_dir: str, subtask_id: str, testbed_dir: str | None = None) -> bool | None:
    """Run evaluation checks for a task. Returns True/False, or None if no eval config found.
    
    Args:
        task_dir: Path to the task directory, used to find the eval config.
        subtask_id: Subtask ID.
        testbed_dir: Path to the testbed containing agent output files.
                     Defaults to task_dir/testbed if not provided.
    """
    task_dir = os.path.abspath(task_dir)
    eval_config_file = os.path.join(task_dir, 'subtasks', f'{subtask_id}.json')
    if not os.path.exists(eval_config_file):
        logger.info(f"No evaluation config found: {eval_config_file}")
        return None

    with open(eval_config_file) as f:
        task_config = json.load(f)

    eval_spec = task_config.get('evaluation')
    if not eval_spec:
        logger.info(f"No 'evaluation' spec in task config: {eval_config_file}")
        return None

    logger.info(f"Running evaluation with {len(eval_spec)} check(s)")

    effective_testbed = os.path.abspath(testbed_dir or os.path.join(task_dir, 'testbed'))

    # NOTE: Thesis modification - testbed output recovery.
    # Original OfficeBench assumes outputs are already under testbed/data/.
    # This benchmark runner first moves accidental root-level outputs there so
    # the original evaluator entry points can still find the produced files.
    _recover_stray_testbed_files(effective_testbed)

    for eval_item in eval_spec:
        func_name = eval_item['function']
        args = eval_item['args']
        for key, value in eval_item.items():
            if key not in ('function', 'args') and key not in args:
                args[key] = value
        func = EVAL_FUNCTIONS.get(func_name)
        if func is None:
            logger.warning(f"Unknown eval function: {func_name}")
            return False
        try:
            if not func(effective_testbed, args):
                logger.info(f"Eval check failed: {func_name}")
                return False
        except Exception as exc:
            logger.exception("Eval check crashed: %s (%s)", func_name, exc)
            return False

    return True


# NOTE: Thesis modification support - directories that belong to the testbed
# layout and must not be moved during stray output recovery.
_TESTBED_SYSTEM_DIRS = {"data", "calendar", "emails", "output", "logs"}


def _recover_stray_testbed_files(testbed_dir: str) -> None:
    """Move files/dirs written directly to the testbed root into /testbed/data/.

    Agents are instructed to write only inside ``/testbed/data/`` but sometimes
    write one level too high.  This function moves any unexpected entries
    at the testbed root into ``data/`` so that the evaluator can find them.
    Known system directories (``data``, ``calendar``, ``emails``, etc.) are
    left untouched.
    """
    if not os.path.isdir(testbed_dir):
        return
    data_dir = os.path.join(testbed_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    for entry in os.listdir(testbed_dir):
        if entry in _TESTBED_SYSTEM_DIRS:
            continue
        src = os.path.join(testbed_dir, entry)
        dst = os.path.join(data_dir, entry)
        if os.path.exists(dst):
            logger.debug("Stray recovery: destination already exists, skipping %s", entry)
            continue
        logger.info("Stray file recovery: moving %s → data/%s", entry, entry)
        shutil.move(src, dst)
