from __future__ import annotations

from typing import Any

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class RandomStrategy(BaseTrainingStrategy):
    """
    Random strategy.

    Randomly samples and updates `random_samples` sub-networks in each step.
    """

    def __init__(self, random_samples: int = 1, **kwargs: Any):
        """
        Initialises a `RandomStrategy`

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        """Updates randomly sampled sub-networks in each step."""
        total_loss = 0
        for i in range(self.random_samples):
            config = self.sampler.sample()
            model.set_sub_network(**config)
            loss = self.compute_loss(model, inputs, outputs)
            loss *= scale_loss
            loss.backward() if self.fabric is None else self.fabric.backward(loss)
            model.reset_super_network()

            total_loss += loss
        return total_loss.item()
