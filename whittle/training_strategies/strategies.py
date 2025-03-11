from .ats import ATS
from .random import RandomStrategy
from .random_linear import RandomLinearStrategy
from .sandwich import SandwichStrategy
from .standard import StandardStrategy


class TrainingStrategies:
    RANDOM = "random"
    STANDARD = "standard"
    SANDWICH = "sandwich"
    RANDOM_LINEAR = "random_linear"
    ATS = "ats"

def get_training_strategy(strategy_type, **kwargs):
    if strategy_type == TrainingStrategies.RANDOM:
        return RandomStrategy(**kwargs)
    elif strategy_type == TrainingStrategies.STANDARD:
        return StandardStrategy(**kwargs)
    elif strategy_type == TrainingStrategies.SANDWICH:
        return SandwichStrategy(**kwargs)
    elif strategy_type == TrainingStrategies.RANDOM_LINEAR:
        return RandomLinearStrategy(**kwargs)
    elif strategy_type == TrainingStrategies.ATS:
        return ATS(**kwargs)
    else:
        raise ValueError(f"Training strategy type {strategy_type} not recognised")