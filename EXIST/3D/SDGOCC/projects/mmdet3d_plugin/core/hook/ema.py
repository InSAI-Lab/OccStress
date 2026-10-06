# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
from copy import deepcopy
import copy
import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import EvalHook as BaseEvalHook
import time
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
from .utils import is_parallel

__all__ = ['ModelEMA']


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/
    pytorch-image-models Keep a moving average of everything in the model
    state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/
    ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training
    schemes to perform well.
    This class is sensitive where it is initialized in the sequence
    of model init, GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema_model = deepcopy(model).eval()
        self.ema = self.ema_model.module.module if is_parallel(
            self.ema_model.module) else self.ema_model.module
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)
    #训练一次iter完转到这里  EMA权重通过考虑过去所有权重的加权平均来提供一个更平滑的权重版本，非瞬时的权重
    def update(self, trainer, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()
            # print('EMA updated')


@HOOKS.register_module()
class MEGVIIEMAHook(Hook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, init_updates=0, decay=0.9990, resume=None):
        super().__init__()
        self.init_updates = init_updates
        self.resume = resume
        self.decay = decay


    def before_run(self, runner):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        runner.ema_model = ModelEMA(runner.model, self.decay)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            load_state_dict(runner.ema_model.ema, cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']



    def after_train_iter(self, runner): #训练完一个iter转到这里然后调到上面
        runner.ema_model.update(runner, runner.model.module)
    #保存完一个epoch的权重后调用
    def after_train_epoch(self, runner):
        # from projects.mmdet3d_plugin.models.apis.test import single_gpu_test
        # results = single_gpu_test(runner.model, self.dataloader, show=False)#推断结果
        # # rank, _ = get_dist_info()
        # # if rank == 0:  # true
        # #     if args.eval:
        # #         eval_kwargs = cfg.get('evaluation', {}).copy()
        # #         # hard-code way to remove EvalHook args
        # #         for key in [
        # #             'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        # #             'rule'
        # #         ]:
        # #             eval_kwargs.pop(key, None)
        # #         eval_kwargs.update(dict(metric=args.eval, **kwargs))  # map
        # #         print(dataset.evaluate(outputs, **eval_kwargs))  # 结果与pipeline和metric
        # runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        # key_score = self.evaluate(runner, results) #得到miou
        # if self.save_best:
        #     self._save_ckpt(runner, key_score)
        if self.is_last_epoch(runner):   # 只保存最后一个epoch的ema权重.
            self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch+1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')
