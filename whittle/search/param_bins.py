import numpy as np
from whittle.metrics.parameters import (
    compute_parameters,
)


class ParamsEstimator:
    def __init__(self, model):
        self.model = model

    def get_params(self, config):
        self.model.select_sub_network(config)
        params = compute_parameters(self.model)
        self.model.reset_super_network()
        return params
    
    def __call__(self, config):
        return self.get_params(config)


class ParamBins:
    def __init__(
        self,
        min_config: dict,
        max_config: dict,
        params_func: callable,
        num_bins: int = 20,
        log_bins: bool = False,
        start_bin_size: int = 1,
        empty_bin_tolerance: int = 4
    ):
        self.params_func = params_func
        self.min_params = self.get_params(min_config)
        self.max_params = self.get_params(max_config)

        # get evenly spaced / log spaced bins between min_params and max_params
        if log_bins:
            self.values = np.logspace(
                np.log10(self.min_params), np.log10(self.max_params), num=num_bins
            )
        else:
            self.values = np.linspace(
                self.min_params, self.max_params, num=num_bins
            )

        self.bins = [0 for _ in self.values[1:]]  # one bin for every lower bound
        self.current_bin_length = start_bin_size
        self.empty_bin_tolerance = empty_bin_tolerance

    def get_params(self, config):
        return self.params_func(config)
    
    def put_in_bin(self, config):
        params = self.get_params(config)
        
        found = False
        placed = False
        at_max_length = 0
        for i, value in enumerate(self.values):
            # get the first bin
            if not found and params < value:
                found = True
                # place into a bin with space left
                if self.bins[i - 1] < self.current_bin_length:
                    self.bins[i - 1] += 1
                    placed = True
            
            # found a bin with space left, don't increase bin length
            if self.bins[i - 1] == self.current_bin_length:
                at_max_length += 1            

        # increase bin length if almost all bins are full
        if (at_max_length + self.empty_bin_tolerance) >= len(self.bins):
            self.current_bin_length += 1    
        
        return placed
