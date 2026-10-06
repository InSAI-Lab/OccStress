from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class HistoryWarmupStepHook(Hook):

    def _set_model_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if hasattr(model, 'set_history_warmup_iter'):
            model.set_history_warmup_iter(runner.iter)

    def before_run(self, runner):
        self._set_model_iter(runner)

    def before_train_iter(self, runner):
        self._set_model_iter(runner)
