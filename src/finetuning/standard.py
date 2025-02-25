from __future__ import annotations

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class StandardStrategy(BaseTrainingStrategy):
    """ """

    def __init__(self, **kwargs):
        """

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.lora = False

    def __call__(self, model, inputs, outputs, step, kd_loss, **kwargs):
        total_loss = 0
        # lora=False
        # update super-network
        # for n,p in model.named_parameters():
        #    if "lora" in n:
        #        lora = True
        #        break
        model.reset_super_network()
        if self.lora:
            y_supernet = model(inputs, lm_head_chunk_size=128)
            y_supernet[-1] = y_supernet[-1][..., :-1, :]
        else:
            y_supernet = model(inputs)
            y_supernet = y_supernet[..., :-1, :]
            # y_supernet[-1] = y_supernet[-1][..., :-1, :]
        loss = self.loss_function(y_supernet, outputs[..., 1:])
        # loss = self.loss_function(y_supernet, outputs)
        self.fabric.backward(loss)
        total_loss += loss.item()

        return total_loss
