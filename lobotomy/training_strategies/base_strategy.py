from lobotomy.sampling.random_sampler import RandomSampler
from typing import Callable


class BaseTrainingStrategy(object):
    """Base Training Strategy

    Base class that all training strategies inherit from.
    """

    def __init__(
        self,
        sampler: RandomSampler,
        loss_function: Callable,
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

    def update(self, **kwargs):
        raise NotImplementedError
