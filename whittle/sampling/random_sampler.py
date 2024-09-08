from __future__ import annotations
import warnings

import numpy as np
from syne_tune.config_space import Categorical, Domain


class RandomSampler:
    def __init__(self, config_space: dict, seed: int | None = None):
        self.config_space = config_space
        self.rng = np.random.RandomState(seed)

    def sample(self):
        config = {}
        for hp_name, hparam in self.config_space.items():
            if isinstance(hparam, Domain):
                config[hp_name] = hparam.sample(random_state=self.rng)
        return config

    def get_smallest_sub_network(self):
        config = {}
        for k, v in self.config_space.items():
            if isinstance(v, Domain):
                if isinstance(v, Categorical):
                    if all(isinstance(e, (int, float)) for e in v.categories):
                        config[k] = min(v.categories)
                    else:
                        warnings.warn(
                            "Warning: Categoricals are non-integers, check if smallest network is as intended"
                        )
                        config[k] = v.categories[0]
                else:
                    config[k] = v.lower
        return config

    def get_largest_sub_network(self):
        config = {}
        for k, v in self.config_space.items():
            if isinstance(v, Domain):
                if isinstance(v, Categorical):
                    if all(isinstance(e, (int, float)) for e in v.categories):
                        config[k] = max(v.categories)
                    else:
                        warnings.warn(
                            "Warning: Categoricals are non-integers, check if largest network is as intended"
                        )
                        config[k] = v.categories[-1]
                else:
                    config[k] = v.upper
        return config
