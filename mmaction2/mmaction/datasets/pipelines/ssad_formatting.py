import numpy as np
from ..builder import PIPELINES
from .formatting import Collect
from .compose import Compose as BaseCompose
import torch
import copy

@PIPELINES.register_module()
class ExtraAttrs(object):
    def __init__(self, **attrs):
        self.attrs = attrs

    def __call__(self, results):
        for k, v in self.attrs.items():
            assert k not in results
            results[k] = v
        return results


@PIPELINES.register_module()
class ExtraCollect(Collect):
    def __init__(self, *args, extra_meta_keys=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_keys = self.meta_keys + tuple(extra_meta_keys)


@PIPELINES.register_module()
class MultiBranch(object):
    def __init__(self, **transform_group):
        self.transform_group = {k: BaseCompose(v) for k, v in transform_group.items()}

    def __call__(self, results):
        multi_results = []
        for k, v in self.transform_group.items():
            res = v(copy.deepcopy(results))
            if res is None:
                return None
            # res["img_metas"]["tag"] = k
            multi_results.append(res)
        return multi_results


@PIPELINES.register_module()
class PseudoSamples(object):
    def __init__(
        self, with_bbox=False
    ):
        self.with_bbox = True
    
    def __call__(self, results):
        if self.with_bbox:
            results["gt_bboxes"] = np.zeros((0, 4))
            results["gt_labels"] = np.zeros((0, 81))
        return results