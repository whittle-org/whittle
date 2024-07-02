from lobotomy.sampling.random_sampler import RandomSampler
from typing import Callable
from lobotomy.loss import DistillLoss


class BaseTrainingStrategy(object):
    def __init__(
        self,
        sampler: RandomSampler,
        loss_function: Callable,
        use_kd_loss: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        """ """
        self.sampler = sampler
        self.loss_function = loss_function
        self.kd_loss = DistillLoss(0.5, 0.5)
        self.device = device
        self.use_kd_loss = use_kd_loss

    def update(self, **kwargs):
        raise NotImplementedError
