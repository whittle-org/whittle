from __future__ import annotations

from typing import Any

from torch.profiler import ProfilerActivity, profile, record_function

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class RandomStrategy(BaseTrainingStrategy):
    """
    Random strategy.

    Randomly samples and updates `random_samples` sub-networks in each step.
    """

    def __init__(self, random_samples: int = 5, **kwargs: Any):
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
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            for i in range(self.random_samples):
                with record_function(f"RS::model.sample_{i}"):
                    config = self.sampler.sample()

                with record_function(f"RS::model.set_sub_network_{i}"):
                    model.set_sub_network(**config)

                with record_function(f"RS:model.compute_loss_{i}"):
                    loss = self.compute_loss(model, inputs, outputs)

                with record_function(f"RS::model.scale_loss_{i}"):
                    loss *= scale_loss

                with record_function(f"RS::model.backward_{i}"):
                    loss.backward() if self.fabric is None else self.fabric.backward(loss)

                with record_function(f"RS::model.reset_network{i}"):
                    model.reset_super_network()

                total_loss += loss.item()

        prof.export_chrome_trace("trace.json")
        return total_loss
