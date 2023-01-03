import warnings
from collections import Counter, Mapping, Sequence
from numbers import Number
from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F

def dict_split(dict1, key):
    group_names = list(set(dict1[key]))
    dict_groups = {k: dict_select(dict1, key, k) for k in group_names}

    return dict_groups


def dict_select(dict1: Dict[str, list], key: str, value: str):
    flag = [v == value for v in dict1[key]]
    return {
        k: dict_fuse([vv for vv, ff in zip(v, flag) if ff], v) for k, v in dict1.items()
    }


def dict_fuse(obj_list, reference_obj):
    if isinstance(reference_obj, torch.Tensor):
        return torch.stack(obj_list)
    return obj_list

_step_counter = Counter()

def sequence_mul(obj, multiplier):
    if isinstance(obj, Sequence):
        return [o * multiplier for o in obj]
    else:
        return obj * multiplier

def is_match(word, word_list):
    for keyword in word_list:
        if keyword in word:
            return True
    return False

def weighted_loss(loss: dict, weight, ignore_keys=[], warmup=0):
    _step_counter["weight"] += 1
    lambda_weight = (
        lambda x: x * (_step_counter["weight"] - 1) / warmup
        if _step_counter["weight"] <= warmup
        else x
    )
    if isinstance(weight, Mapping):
        for k, v in weight.items():
            for name, loss_item in loss.items():
                if (k in name) and ("loss" in name):
                    loss[name] = sequence_mul(loss[name], lambda_weight(v))
    elif isinstance(weight, Number):
        for name, loss_item in loss.items():
            if "loss" in name:
                if not is_match(name, ignore_keys):
                    loss[name] = sequence_mul(loss[name], lambda_weight(weight))
                else:
                    loss[name] = sequence_mul(loss[name], 0.0)
    else:
        raise NotImplementedError()
    return loss