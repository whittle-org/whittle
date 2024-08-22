from __future__ import annotations

from typing import Callable

import torch

from whittle.loss import DistillLoss
from whittle.sampling.random_sampler import RandomSampler


class BaseTrainingStrategy:
    """
    Base Training Strategy.

    Base class that all training strategies inherit from.
    """

    def __init__(
        self,
        sampler: RandomSampler,
        loss_function: Callable,
        kd_loss: Callable | None = None,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Initialises a `BaseTrainingStrategy`
        Args:
            sampler: sampler that returns a sub-network when called
            loss_function: loss function to compute the loss of a sub-network
            device: device to run the model on
            **kwargs:
        """
        self.sampler = sampler
        self.loss_function = loss_function
        self.device = device
        self.kd_loss = kd_loss
        if isinstance(self.kd_loss, DistillLoss):
            if not isinstance(loss_function, torch.nn.CrossEntropyLoss):
                raise TypeError(
                    "KD Loss not yet supported: Expected torch.nn.CrossEntropyLoss"
                )

    def update(self, **kwargs):
        raise NotImplementedError
