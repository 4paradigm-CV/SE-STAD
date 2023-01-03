import os.path as osp
import torch.distributed as dist

from mmcv.runner.hooks import LoggerHook

from mmaction.core.evaluation import DistEvalHook as BaseDistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm

# 需要适配, 因为要使用 iterbasedrunner
class DistEvalHook(BaseDistEvalHook):
    greater_keys = [
            'acc', 'top', 'AR@', 'auc', 'precision', 'mAP@', 'Recall@'
    ]
    less_keys = ['loss']
    def __init__(self, *args, by_epoch=False, pre_eval=False, save_best="auto", **kwargs):
        super().__init__(*args, by_epoch=by_epoch, save_best=save_best, **kwargs)
        self.pre_eval = pre_eval
    
    # 参考 mmaction 直接按照 mmcv 继承的 _do_evaluate 即可

    # def _do_evaluate(self, runner):
    #     pass



# 暂时没有必要使用这个, 这个只是添加了 wandb 的版本
# class DistEvalHook(BaseDistEvalHook):
#     def after_train_iter(self, runner):
#         """Called after every training iter to evaluate the results."""
#         if not self.by_epoch and self._should_evaluate(runner):
#             for hook in runner._hooks:
#                 if isinstance(hook, WandbLoggerHook):
#                     _commit_state = hook.commit
#                     hook.commit = False
#                 if isinstance(hook, LoggerHook):
#                     hook.after_train_iter(runner)
#                 if isinstance(hook, WandbLoggerHook):
#                     hook.commit = _commit_state
#             runner.log_buffer.clear()

#             self._do_evaluate(runner)

#     def _do_evaluate(self, runner):
#         """perform evaluation and save ckpt."""
#         # Synchronization of BatchNorm's buffer (running_mean
#         # and running_var) is not supported in the DDP of pytorch,
#         # which may cause the inconsistent performance of models in
#         # different ranks, so we broadcast BatchNorm's buffers
#         # of rank 0 to other ranks to avoid this.
#         if self.broadcast_bn_buffer:
#             model = runner.model
#             for name, module in model.named_modules():
#                 if isinstance(module, _BatchNorm) and module.track_running_stats:
#                     dist.broadcast(module.running_var, 0)
#                     dist.broadcast(module.running_mean, 0)

#         if not self._should_evaluate(runner):
#             return

#         tmpdir = self.tmpdir
#         if tmpdir is None:
#             tmpdir = osp.join(runner.work_dir, ".eval_hook")

#         from mmdet.apis import multi_gpu_test

#         results = multi_gpu_test(
#             runner.model, self.dataloader, tmpdir=tmpdir, gpu_collect=self.gpu_collect
#         )
#         if runner.rank == 0:
#             print("\n")
#             # runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
#             key_score = self.evaluate(runner, results)

#             if self.save_best:
#                 self._save_ckpt(runner, key_score)
