
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import os.path as osp
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import get_dist_info
import time
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
import torch
import copy

class OccEvalHook(BaseEvalHook):
    def __init__(self, *args,  **kwargs):
        super(OccEvalHook, self).__init__(*args, **kwargs)
        # 运行完一个epoch跳转到这
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from projects.mmdet3d_plugin.models.apis.test import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)#推断结果
        # rank, _ = get_dist_info()
        # if rank == 0:  # true
        #     if args.eval:
        #         eval_kwargs = cfg.get('evaluation', {}).copy()
        #         # hard-code way to remove EvalHook args
        #         for key in [
        #             'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        #             'rule'
        #         ]:
        #             eval_kwargs.pop(key, None)
        #         eval_kwargs.update(dict(metric=args.eval, **kwargs))  # map
        #         print(dataset.evaluate(outputs, **eval_kwargs))  # 结果与pipeline和metric
        # 手动计算损失
        # runner.model.eval()
        # total_loss = 0
        # count = 0
        # with torch.no_grad():
        #     for data in self.dataloader:
        #         loss = runner.model(**data, return_loss=True)
        #         total_loss += sum(loss.values()).item()
        #         count += 1
        #
        # # 计算平均损失
        # val_loss = total_loss / count if count != 0 else 0
        # runner.logger.info(f'Validation Loss: {val_loss}')
        # runner.log_buffer.output['val_loss'] = val_loss
        # runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results) #得到miou
        if self.save_best:
            self._save_ckpt(runner, key_score)
    #
    # def construct_input(self, DUMMY_SHAPE=None, m_info=None):
    #     if m_info is None:
    #         m_info = next(iter(self.dataloader))
    #     img_metas = m_info['img_metas'][0].data
    #     input = dict(
    #         img_metas=img_metas,
    #     )
    #     if 'img_inputs' in m_info.keys():
    #         img_inputs = m_info['img_inputs'][0]
    #         for i in range(len(img_inputs)):
    #             if isinstance(img_inputs[i], list):
    #                 for j in range(len(img_inputs[i])):
    #                     img_inputs[i][j] = img_inputs[i][j].cuda()
    #             else:
    #                 img_inputs[i] = img_inputs[i].cuda()
    #         input['img_inputs'] = img_inputs
    #
    #     if 'points' in m_info.keys():
    #         points = m_info['points'][0].data[0]
    #         points[0] = points[0].cuda()
    #         input['points'] = points
    #     return input
    # # #
    # def before_run(self, runner):
    #     torch.cuda.reset_peak_memory_stats()
    #     model = copy.deepcopy(runner.model)
    #     if hasattr(model, 'module'):
    #         model = model.module
    #     if hasattr(model, 'forward_dummy'):
    #         model.forward_train = model.forward_dummy
    #         model.forward_test = model.forward_dummy
    #         model.eval()
    #     else:
    #         raise NotImplementedError(
    #             'FLOPs counter is currently not supported for {}'.format(
    #                 model.__class__.__name__))
    #
    #     # flops and params
    #     if runner.rank == 0:
    #         flops, params = get_model_complexity_info(
    #             model, (None, None), input_constructor=self.construct_input)
    #
    #         split_line = '=' * 30
    #         gpu_measure = torch.cuda.max_memory_allocated() / 1024. / 1024. / 1024.
    #         runner.logger.info(
    #             f'{split_line}\n' f'Flops: {flops}\nParams: {params}\nGPU memory: {gpu_measure:.2f}GB\n{split_line}')
    #     print('end')




        # class OccDistEvalHook(BaseDistEvalHook):
#     def __init__(self, *args,  **kwargs):
#         super(OccDistEvalHook, self).__init__(*args, **kwargs)
#
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
#                 if isinstance(module,
#                               _BatchNorm) and module.track_running_stats:
#                     dist.broadcast(module.running_var, 0)
#                     dist.broadcast(module.running_mean, 0)
#
#         if not self._should_evaluate(runner):
#             return
#
#         tmpdir = self.tmpdir
#         if tmpdir is None:
#             tmpdir = osp.join(runner.work_dir, '.eval_hook')
#
#         from projects.mmdet3d_plugin.models.apis.test import single_gpu_test
#
#         results = custom_multi_gpu_test(
#             runner.model,
#             self.dataloader,
#             tmpdir=tmpdir,
#             gpu_collect=self.gpu_collect)
#
#         if runner.rank == 0:
#             print('\n')
#             runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
#
#             key_score = self.evaluate(runner, results)
#
#             if self.save_best:
#                 self._save_ckpt(runner, key_score)
  
