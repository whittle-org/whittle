from lobotomy.sampling.random_sampler import RandomSampler
from typing import Callable


class BaseTrainingStrategy(object):
    def __init__(
        self, sampler: RandomSampler, loss_function: Callable, device: str = "cuda", **kwargs
    ):
        """

        """
        self.sampler = sampler
        self.loss_function = loss_function
        self.device = device

    def update(self, **kwargs):
        raise NotImplementedError
