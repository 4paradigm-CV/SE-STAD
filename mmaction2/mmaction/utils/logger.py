# Copyright (c) OpenMMLab. All rights reserved.
import logging
import sys
import os
from collections import Counter

from mmcv.utils import get_logger

_log_counter = Counter()

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "mmaction".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    logger.propogate = False
    return logger
    # return get_logger(__name__.split('.')[0], log_file, log_level)

def _find_caller():
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = r"ssod"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back

def log_every_n(msg: str, n: int = 50, level: int = logging.DEBUG, backend="auto"):
    """
    Args:
        msg (Any):
        n (int):
        level (int):
        name (str):
    """
    caller_module, key = _find_caller()
    _log_counter[key] += 1
    if n == 1 or _log_counter[key] % n == 1:
        get_root_logger().log(level, msg)
