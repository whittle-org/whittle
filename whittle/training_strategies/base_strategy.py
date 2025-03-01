from __future__ import annotations

from collections.abc import Callable

import torch
from lightning.fabric import Fabric

from whittle.loss import DistillLoss


class BaseTrainingStrategy:
    """
    Base Training Strategy.

    Base class that all training strategies inherit from.
    """

    def __init__(
        self,
        sampler,
        loss_function: Callable,
        kd_loss: Callable | None = None,
        device: str = "cuda",
        lora: bool = False,
        fabric: Fabric | None = None,
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
        self.lora = lora
        self.fabric = fabric
        if isinstance(self.kd_loss, DistillLoss):
            if not isinstance(loss_function, torch.nn.CrossEntropyLoss):
                raise TypeError(
                    "KD Loss not yet supported: Expected torch.nn.CrossEntropyLoss"
                )

    def chunked_loss(self, model, inputs, y):
        y_hat = model(inputs, lm_head_chunk_size=128)
        y_hat[-1] = y_hat[-1][..., :-1, :]
        return self.loss_function(y_hat, y[..., 1:])

    def compute_loss(self, model, inputs, outputs):
        if self.lora:
            loss = self.chunked_loss(model, inputs, outputs)
        else:
            y_hat = model(inputs)
            loss = self.loss_function(y_hat, outputs)
        return loss

    def __call__(self, model, inputs, outputs, scale_loss=1, **kwargs):
        raise NotImplementedError
