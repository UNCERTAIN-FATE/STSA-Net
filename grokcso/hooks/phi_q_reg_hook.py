from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from grokcso.models.blocks.phi_q import phi_q_regularization_loss

@HOOKS.register_module()
class PhiQRegHook(Hook):
    def __init__(self, loss_weight=1.0):
        self.loss_weight = loss_weight

    def after_train_iter(self, runner):
        if hasattr(runner.model, 'Phi') and hasattr(runner.model, 'Q'):
            reg_loss = phi_q_regularization_loss(runner.model.Phi, runner.model.Q)
            if 'loss' not in runner.outputs:
                runner.outputs['loss'] = 0.0
            runner.outputs['loss'] = runner.outputs['loss'] + self.loss_weight * reg_loss
