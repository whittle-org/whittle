from __future__ import annotations

from typing import Any

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class StandardStrategy(BaseTrainingStrategy):
    """
    Standard strategy.

    Implements the standard update rule and updates all weights of the super-network.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialises a `StandardStrategy`

        Args:
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        y_hat = model(inputs)
        loss = self.loss_function(y_hat, outputs)
        loss *= scale_loss
        loss.backward() if self.fabric is None else self.fabric.backward(loss)
        return loss.item()
