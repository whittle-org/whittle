from __future__ import annotations

from typing import Any

from torch.profiler import ProfilerActivity, profile, record_function

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class SandwichStrategy(BaseTrainingStrategy):
    """
    Sandwich strategy.

    In each step, the sandwich strategy updates the super-network, the smallest, and a set of randomly sampled
    sub-networks.

    refs:
        Universally Slimmable Networks and Improved Training Techniques
        Jiahui Yu, Thomas Huang
        International Conference on Computer Vision 2019
        https://arxiv.org/abs/1903.05134
    """

    def __init__(self, random_samples: int = 5, **kwargs: Any):
        """
        Initialises a `SandwichStrategy`

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        total_loss = 0
        # update super-network

        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            with record_function("Sandwich::largest_network"):
                model.reset_super_network()
                loss = self.compute_loss(model, inputs, outputs)
                loss *= scale_loss
                loss.backward() if self.fabric is None else self.fabric.backward(loss)
                total_loss += loss.item()

            # update random sub-networks
            for i in range(self.random_samples):
                with record_function(f"Sandwich::random_sample_{i}"):
                    config = self.sampler.sample()
                    model.set_sub_network(**config)
                    loss = self.compute_loss(model, inputs, outputs)
                    loss *= scale_loss
                    loss.backward() if self.fabric is None else self.fabric.backward(loss)
                    model.reset_super_network()
                    total_loss += loss.item()

            with record_function("Sandwich::smallest_network"):
                # smallest network
                config = self.sampler.get_smallest_sub_network()
                model.set_sub_network(**config)
                loss = self.compute_loss(model, inputs, outputs)
                loss *= scale_loss
                loss.backward() if self.fabric is None else self.fabric.backward(loss)
                model.reset_super_network()
                total_loss += loss.item()

        prof.export_chrome_trace("trace.json")

        return total_loss
