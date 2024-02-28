from mmcv.runner import (HOOKS, Hook)
from torch.nn.utils import clip_grad
import pdb
@HOOKS.register_module()
class OptimizerHook_(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        pdb.set_trace()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        model = runner.model
        for key, value in model.module.roi_head.mask_predictor.named_parameters():
            value.grad = 0.05*value.grad
        runner.optimizer.step()
