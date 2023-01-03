import glob
import os
import os.path as osp
import shutil
import types

from mmcv.runner import BaseRunner, EpochBasedRunner, IterBasedRunner
from mmcv.utils import Config

# from .signature import parse_method_info
from .vars import resolve

def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.work_dir

def patch_config(cfg):
    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict["cfg_name"] = osp.splitext(osp.basename(cfg.filename))[0]
    cfg_dict = resolve(cfg_dict)
    cfg = Config(cfg_dict, filename=cfg.filename)
    # wrap for semi
    if cfg.get("semi_wrapper", None) is not None:
        cfg.model = cfg.semi_wrapper
        cfg.pop("semi_wrapper")
    # enable environment variables
    setup_env(cfg)
    return cfg