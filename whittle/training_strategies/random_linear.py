from __future__ import annotations

from typing import Any

import numpy as np

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class RandomLinearStrategy(BaseTrainingStrategy):
    """
    Random linear strategy.

    Updates `random_samples` randomly sampled sub-network with probability `p` or the super-network with `1 - p`. `p`
    linearly increases with the step count.

    refs:
        Structural Pruning of Pre-trained Language Models via Neural Architecture Search
        Aaron Klein, Jacek Golebiowski, Xingchen Ma, Valerio Perrone, Cedric Archambeau
        https://arxiv.org/abs/2405.02267

        Understanding and Simplifying One-Shot Architecture Search
        Gabriel Bender, Pieter-Jan Kindermans, Barret Zoph, Vijay Vasudevan, Quoc Le
        International Conference on Machine Learning (ICML) 2018
        https://proceedings.mlr.press/v80/bender18a/bender18a.pdf
    """

    def __init__(
        self, total_number_of_steps: int, random_samples: int = 1, **kwargs: Any
    ):
        """
        Initialises a `RandomLinearStrategy`

        Args:
            total_number_of_steps: the number of steps the optimization runs for
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples
        self.total_number_of_steps = total_number_of_steps
        self.current_step = 0
        self.rate = np.linspace(0.0, 1, total_number_of_steps)

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        total_loss = 0
        if np.random.rand() <= self.rate[self.current_step]:
            # update random sub-networks
            for i in range(self.random_samples):
                config = self.sampler.sample()
                model.set_sub_network(**config)
                loss = self.compute_loss(model, inputs, outputs)
                loss *= scale_loss
                loss.backward() if self.fabric is None else self.fabric.backward(loss)
                model.reset_super_network()

                total_loss += loss
        else:
            loss = self.compute_loss(model, inputs, outputs)
            loss *= scale_loss
            loss.backward() if self.fabric is None else self.fabric.backward(loss)
            total_loss = loss
        self.current_step += 1
        return total_loss.item()
