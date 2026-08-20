import math
from typing import Any

from lm_eval.tasks.ifeval import utils as official_ifeval_utils


def _coerce_int_like_floats(value: Any) -> Any:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_coerce_int_like_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _coerce_int_like_floats(item) for key, item in value.items()}
    return value


def process_results(doc, results):
    return official_ifeval_utils.process_results(_coerce_int_like_floats(doc), results)


def agg_inst_level_acc(items):
    return official_ifeval_utils.agg_inst_level_acc(items)
