# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import platform
import random
from functools import partial

import numpy as np
import torch
# from mmcv.parallel import collate # TODO: 看官方代码应该是已经支持了 DataContainer 了, 就不按照 softteacher 再重新写一个 collate 了
from .collate import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

from .samplers import ClassSpecificDistributedSampler, DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
BLENDINGS = Registry('blending')
SAMPLERS = Registry("sampler")
SAMPLERS.register_module(module=DistributedSampler)

from torch.utils.data import Sampler
import math

@SAMPLERS.register_module()
class DistributedSemiBalanceSampler(Sampler):
    def __init__(
        self,
        dataset,
        sample_ratio=None,
        samples_per_gpu=1,
        num_replicas=None,
        rank=None,
        drop_last=True,
        seed=0,
        **kwargs
    ):
        self.seed = seed if seed is not None else 0
        # check to avoid some problem
        assert samples_per_gpu > 1, "samples_per_gpu should be greater than 1."
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        self.num_samples = 0
        self.cumulative_sizes = dataset.cumulative_sizes

        if not isinstance(sample_ratio, list):
            sample_ratio = [sample_ratio] * len(self.cumulative_sizes)
        self.sample_ratio = sample_ratio

        assert self.samples_per_gpu % sum(self.sample_ratio) == 0

        self.sample_ratio = [
            int(self.samples_per_gpu / sum(self.sample_ratio) * sr) for sr in self.sample_ratio
        ]

        self.sample_ratio[-1] = self.samples_per_gpu - sum(self.sample_ratio[:-1])

        cumulative_sizes = [0] + self.cumulative_sizes
        size_of_dataset = 0

        for j in range(len(self.cumulative_sizes)):
            size_per_dataset = cumulative_sizes[j+1] - cumulative_sizes[j]
            if self.drop_last and size_of_dataset % (self.num_replicas * self.sample_ratio[j]) != 0:
                size_of_dataset = max(
                    size_of_dataset, math.ceil(
                        (size_per_dataset - self.sample_ratio[j] * self.num_replicas) / self.sample_ratio[j] / self.num_replicas
                    )
                )
            else:
                size_of_dataset = max(
                    size_of_dataset, math.ceil(
                        size_per_dataset / self.sample_ratio[j] / self.num_replicas
                    )
                )
            
        for j in range(len(self.cumulative_sizes)):
            self.num_samples += size_of_dataset * self.sample_ratio[j]
        
        self.total_size  = self.num_samples * self.num_replicas
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        indices = []
        cumulative_sizes = [0] + self.cumulative_sizes

        indice_per_dataset = []
        for j in range(len(self.cumulative_sizes)):
            indice_per_dataset.append(
                np.arange(cumulative_sizes[j], cumulative_sizes[j+1])
            )
        
        shuffled_indice_per_dataset = [
            s[list(torch.randperm(int(s.shape[0]), generator=g).numpy())]
            for s in indice_per_dataset
        ]

        assert self.drop_last
        indexes = [0, 0]
        for i in range(self.total_size // self.samples_per_gpu):
            for j in range(len(self.sample_ratio)):
                for _ in range(self.sample_ratio[j]):
                    indices.append(shuffled_indice_per_dataset[j][indexes[j]])
                    indexes[j] += 1
                    if indexes[j] >= len(shuffled_indice_per_dataset[j]):
                        indexes[j] = 0
        
        assert len(indices) % (self.num_replicas) == 0
        assert len(indices) % (self.num_replicas * sum(self.sample_ratio)) == 0
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        return_indices = []
        for i in range(self.total_size//self.samples_per_gpu//self.num_replicas):
            return_indices += indices[
                (self.rank+i*self.num_replicas)*self.samples_per_gpu:(self.rank+i*self.num_replicas+1)*self.samples_per_gpu
            ]
        
        # print(len(return_indices), self.num_samples)
        assert len(return_indices) == self.num_samples
        return iter(return_indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset

# SAMPLERS.register_module(module=DistributedGroupSampler)
# SAMPLERS.register_module(module=GroupSampler)

def build_sampler(cfg, dist=False, default_args=None):
    if cfg and ("type" in cfg):
        sampler_type = cfg.get("type")
    else:
        sampler_type = default_args.get("type")
    if dist:
        sampler_type = "Distributed" + sampler_type
    
    if cfg:
        cfg.update(type=sampler_type)
    else:
        cfg = dict(type=sampler_type)
    return build_from_cfg(cfg, SAMPLERS, default_args)

def build_dataloader(dataset,
                     videos_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=True, # 这里改成了 True
                     pin_memory=True,
                     persistent_workers=False,
                     sampler_cfg=None, # added by ssad
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (:obj:`Dataset`): A PyTorch dataset.
        videos_per_gpu (int): Number of videos on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data
            loading for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed
            training. Default: 1.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.8.0.
            Default: False
        kwargs (dict, optional): Any keyword argument to be used to initialize
            DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    sample_by_class = getattr(dataset, 'sample_by_class', False)

    # added by ssad
    if sampler_cfg is not None:
        default_sampler_cfg = dict(type="Sampler", dataset=dataset)
        if shuffle:
            default_sampler_cfg.update(samples_per_gpu=videos_per_gpu, shuffle=True)
        else:
            default_sampler_cfg.update(shuffle=False)
        
        if dist:
            if seed is not None:
                default_sampler_cfg.update(num_replicas=world_size, rank=rank, seed=seed)
            else:
                default_sampler_cfg.update(num_replicas=world_size, rank=rank)
            # 没有 group 选项
            sampler = build_sampler(sampler_cfg, dist, default_sampler_cfg)

            batch_size = videos_per_gpu
            num_workers = workers_per_gpu
        else:
            sampler = (
                build_sampler(sampler_cfg, default_args=default_sampler_cfg)
                if shuffle
                else None
            )
            batch_size = num_gpus * videos_per_gpu
            num_workers = num_gpus * workers_per_gpu
        shuffle = False
    else:
        if dist:
            if sample_by_class:
                dynamic_length = getattr(dataset, 'dynamic_length', True)
                sampler = ClassSpecificDistributedSampler(
                    dataset,
                    world_size,
                    rank,
                    dynamic_length=dynamic_length,
                    shuffle=shuffle,
                    seed=seed)
            else:
                sampler = DistributedSampler(
                    dataset, world_size, rank, shuffle=shuffle, seed=seed)
            shuffle = False
            batch_size = videos_per_gpu
            num_workers = workers_per_gpu
        else:
            sampler = None
            batch_size = num_gpus * videos_per_gpu
            num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=videos_per_gpu, flatten=True),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
