from .builder import DATASETS, build_dataset
# from .dataset_wrappers import ConcatDataset
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

@DATASETS.register_module()
class SemiDataset(_ConcatDataset):
    """Wrapper for ssad."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]