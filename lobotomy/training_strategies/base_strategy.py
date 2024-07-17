from typing import Callable, Optional

import torch

from lobotomy.loss import DistillLoss
from lobotomy.sampling.random_sampler import RandomSampler


class BaseTrainingStrategy(object):
    def __init__(
        self,
        sampler: RandomSampler,
        loss_function: Callable,
        kd_loss: Optional[DistillLoss] = None,
        device: str = "cuda",
        **kwargs,
    ):
        """ """
        self.sampler = sampler
        self.loss_function = loss_function
        self.device = device
        self.kd_loss = kd_loss
        if self.kd_loss is not None:
            if not isinstance(loss_function, torch.nn.CrossEntropyLoss):
                raise TypeError(
                    "KD Loss not yet supported: Expected torch.nn.CrossEntropyLoss"
                )

    def update(self, **kwargs):
        raise NotImplementedError
