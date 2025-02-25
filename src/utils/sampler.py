import numpy as np
import pickle
from syne_tune.config_space import Categorical, Domain
from whittle.sampling.random_sampler import RandomSampler as WhittleRandomSampler
from whittle.metrics.parameters import (
    compute_all_parameters,
    compute_parameters,
)


class ImportanceSampler:
    def __init__(self, sorted_ids_path, search_space, seed: int | None = None):
        self.search_space = search_space
        self.sampler = WhittleRandomSampler(self.search_space.config_space, seed=seed)
        with open(sorted_ids_path, "rb") as f:
            self.sorted_ids = pickle.load(f)

    def sort_layers(self, config):
        layer_order = self.sorted_ids["layer_order"]
        layer_ids_selected = sorted(layer_order[: int(config["sub_network_n_layers"])])
        config["sampled_layer_indices"] = layer_ids_selected
        return config

    def sample(self):
        config = self.search_space.cast(self.sampler.sample())
        config = self.sort_layers(config)
        return config

    def get_smallest_sub_network(self):
        config = self.search_space.cast(self.sampler.get_smallest_sub_network())
        config = self.sort_layers(config)
        return config

    def get_medium_sub_network(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = hp.categories[len(hp.categories) // 2]
            else:
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(0.5 * (u - l) + l)
        config = self.search_space.cast(config)
        config = self.sort_layers(config)
        return config

    def get_largest_sub_network(self):
        return self.search_space.cast(self.sampler.get_largest_sub_network())


class RandomSampler:
    def __init__(self, search_space, seed: int | None = None):
        self.search_space = search_space
        self.sampler = WhittleRandomSampler(self.search_space.config_space, seed=seed)

    def sample(self):
        return self.search_space.cast(self.sampler.sample())

    def get_smallest_sub_network(self):
        return self.search_space.cast(self.sampler.get_smallest_sub_network())

    def get_medium_sub_network(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = hp.categories[len(hp.categories) // 2]
            else:
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(0.5 * (u - l) + l)
        return self.search_space.cast(config)

    def get_largest_sub_network(self):
        return self.search_space.cast(self.sampler.get_largest_sub_network())


class FixGridSampler:
    def __init__(self, search_space, num_configs: int = 21, seed: int | None = None):
        self.search_space = search_space
        self.rng = np.random.RandomState(seed)
        values = [(i) / num_configs for i in range(num_configs + 1)]
        print(values)
        self.grid = []
        for value in values:
            config = {}
            for hp_name, hp in self.search_space.config_space.items():
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(value * (u - l) + l)
            self.grid.append(config)

    def sample(self):
        return self.search_space.cast(self.rng.choice(self.grid))

    def get_smallest_sub_network(self):
        return self.search_space.cast(self.grid[0])

    def get_medium_sub_network(self):
        return self.search_space.cast(self.grid[len(self.grid) // 2])

    def get_largest_sub_network(self):
        return self.search_space.cast(self.grid[-1])


class CalibFixGridSampler(FixGridSampler):
    def __init__(
        self,
        objective,
        checkpoint_dir,
        search_space_type,
        search_space,
        num_configs: int = 21,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.rng = np.random.RandomState(seed)
        # Note: need to run calibration calibrate/run_calibration.py for search space type before hand
        grid_path = f"{checkpoint_dir}/grid_{search_space_type}_{objective}.pkl"
        with open(grid_path, "rb") as f:
            self.grid = pickle.load(f)

    def sample(self):
        return self.rng.choice(self.grid)

    def get_smallest_sub_network(self):
        return self.grid[0]

    def get_medium_sub_network(self):
        return self.grid[len(self.grid) // 2]

    def get_largest_sub_network(self):
        return self.grid[-1]


class FixParamGridSampler(FixGridSampler):
    def __init__(
        self,
        search_space,
        num_configs: int = 21,
        n_trials=5000,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.rng = np.random.RandomState(seed)
        self.sampler = WhittleRandomSampler(self.search_space.config_space, seed=seed)
        self.values = [(i) / num_configs for i in range(num_configs + 1)]
        print(self.values)
        self.grid = []

    def add_max_config(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = max(hp.categories)
            else:
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(self.values[-1] * (u - l) + l)
        self.grid.append(config)

    def add_min_config(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = min(hp.categories)
            else:
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(self.values[0] * (u - l) + l)
        self.grid.append(config)

    def initialize_grid(self, model):
        model.reset_super_network()
        u = compute_all_parameters(model)
        l = self.get_smallest_params(model)
        params_min = l
        self.add_min_config()
        for value in self.values[1:]:
            params_max = int(value * (u - l) + l)
            config = self.constrained_search(params_min, params_max, model)
            if config is not None:
                self.grid.append(config)
            params_min = params_max
        self.add_max_config()

    def constrained_search(self, params_min, params_max, model):
        for _ in range(self.n_trials):
            config = self.sampler.sample()
            model.set_sub_network(**self.search_space.cast(config))
            params = compute_parameters(model)
            model.reset_super_network()
            if params >= params_min and params < params_max:
                # print(params)
                return config

    def get_smallest_params(self, model):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = min(hp.categories)
            else:
                config[hp_name] = hp.lower
        model.set_sub_network(**self.search_space.cast(config))
        params = compute_parameters(model)
        model.reset_super_network()
        return params


class ImportanceFixGridSampler:
    def __init__(
        self,
        sorted_ids_path,
        search_space,
        num_configs: int = 21,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.rng = np.random.RandomState(seed)
        values = [(i) / num_configs for i in range(num_configs + 1)]
        print(values)
        self.grid = []
        for value in values:
            config = {}
            for hp_name, hp in self.search_space.config_space.items():
                if isinstance(hp, Categorical):
                    u = max(hp.categories)
                    l = min(hp.categories)
                else:
                    u = hp.upper
                    l = hp.lower
                config[hp_name] = int(value * (u - l) + l)
            self.grid.append(config)
        with open(sorted_ids_path, "rb") as f:
            self.sorted_ids = pickle.load(f)

    def sort_layers(self, config):
        layer_order = self.sorted_ids["layer_order"]
        layer_ids_selected = sorted(layer_order[: int(config["sub_network_n_layers"])])
        config["sampled_layer_indices"] = layer_ids_selected
        return config

    def sample(self):
        config = self.search_space.cast(self.rng.choice(self.grid))
        config = self.sort_layers(config)
        return config

    def get_smallest_sub_network(self):
        config = self.search_space.cast(self.grid[0])
        config = self.sort_layers(config)
        return config

    def get_medium_sub_network(self):
        config = self.search_space.cast(self.grid[len(self.grid) // 2])
        config = self.sort_layers(config)
        return config

    def get_largest_sub_network(self):
        config = self.search_space.cast(self.grid[-1])
        config = self.sort_layers(config)
        return config


class LlamaGridSampler(ImportanceFixGridSampler):
    def __init__(
        self,
        sorted_ids_path,
        seed: int | None = None,
    ):
        self.rng = np.random.RandomState(seed)
        # Note: need to run calibration calibrate/run_calibration.py for search space type before hand
        self.grid = []
        self.grid.append(self.get_llama3_2_1b())
        self.grid.append(self.get_llama3_2_3b())
        self.grid.append(self.get_llama3_1_8b())
        with open(sorted_ids_path, "rb") as f:
            self.sorted_ids = pickle.load(f)

    def get_llama3_2_1b(self):
        config = {}
        config["sub_network_n_embd"] = 2048
        config["sub_network_intermediate_size"] = [8192 for _ in range(16)]
        config["sub_network_num_heads"] = [32 for _ in range(16)]
        config["sub_network_n_layers"] = 16
        config["sub_network_head_size"] = 64
        return config

    def get_llama3_2_3b(self):
        config = {}
        config["sub_network_n_embd"] = 3072
        config["sub_network_intermediate_size"] = [8192 for _ in range(28)]
        config["sub_network_num_heads"] = [24 for _ in range(28)]
        config["sub_network_n_layers"] = 28
        config["sub_network_head_size"] = 128
        return config

    def get_llama3_1_8b(self):
        config = {}
        config["sub_network_n_embd"] = 4096
        config["sub_network_intermediate_size"] = [14336 for _ in range(32)]
        config["sub_network_num_heads"] = [32 for _ in range(32)]
        config["sub_network_n_layers"] = 32
        config["sub_network_head_size"] = 128
        return config

    def sort_layers(self, config):
        layer_order = self.sorted_ids["layer_order"]
        layer_ids_selected = sorted(layer_order[: int(config["sub_network_n_layers"])])
        config["random_layers"] = layer_ids_selected
        return config

    def sample(self):
        config = self.rng.choice(self.grid)
        config = self.sort_layers(config)
        return config

    def get_smallest_sub_network(self):
        config = self.grid[0]
        config = self.sort_layers(config)
        return config

    def get_medium_sub_network(self):
        config = self.grid[1]
        config = self.sort_layers(config)
        return config

    def get_largest_sub_network(self):
        config = self.grid[-1]
        config = self.sort_layers(config)
        return config


class ImportanceCalibFixGridSampler(ImportanceFixGridSampler):
    def __init__(
        self,
        objective,
        importance_objective,
        sorted_ids_path,
        checkpoint_dir,
        search_space_type,
        search_space,
        num_configs: int = 21,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.rng = np.random.RandomState(seed)
        # Note: need to run calibration calibrate/run_calibration.py for search space type before hand
        grid_path = f"{checkpoint_dir}/grid_{search_space_type}_{objective}_{importance_objective}.pkl"
        with open(grid_path, "rb") as f:
            self.grid = pickle.load(f)
        with open(sorted_ids_path, "rb") as f:
            self.sorted_ids = pickle.load(f)

    def sort_layers(self, config):
        layer_order = self.sorted_ids["layer_order"]
        layer_ids_selected = sorted(layer_order[: int(config["sub_network_n_layers"])])
        config["sampled_layer_indices"] = layer_ids_selected
        return config

    def sample(self):
        config = self.rng.choice(self.grid)
        config = self.sort_layers(config)
        return config

    def get_smallest_sub_network(self):
        config = self.grid[0]
        config = self.sort_layers(config)
        return config

    def get_medium_sub_network(self):
        config = self.grid[len(self.grid) // 2]
        config = self.sort_layers(config)
        return config

    def get_largest_sub_network(self):
        config = self.grid[-1]
        config = self.sort_layers(config)
        return config


class ImportanceParamGridSampler(ImportanceFixGridSampler):
    def __init__(
        self,
        sorted_ids_path,
        search_space,
        num_configs: int = 21,
        n_trials=10,
        seed: int | None = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.rng = np.random.RandomState(seed)
        self.sampler = WhittleRandomSampler(self.search_space.config_space, seed=seed)
        with open(sorted_ids_path, "rb") as f:
            self.sorted_ids = pickle.load(f)
        self.values = [(i) / num_configs for i in range(num_configs + 1)]
        print(self.values)
        self.grid = []

    def add_max_config(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = max(hp.categories)
            else:
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(self.values[-1] * (u - l) + l)
        self.grid.append(config)

    def sort_layers(self, config):
        layer_order = self.sorted_ids["layer_order"]
        layer_ids_selected = sorted(layer_order[: int(config["sub_network_n_layers"])])
        config["sampled_layer_indices"] = layer_ids_selected
        return config

    def add_min_config(self):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = min(hp.categories)
            else:
                u = hp.upper
                l = hp.lower
                config[hp_name] = int(self.values[0] * (u - l) + l)
        print(config)
        layer_order = self.sorted_ids["layer_order"]
        layer_ids_selected = sorted(layer_order[: int(config["depth"])])
        config["sampled_layer_indices"] = layer_ids_selected
        self.grid.append(config)

    def initialize_grid(self, model):
        model.reset_super_network()
        u = compute_all_parameters(model)
        l = self.get_smallest_params(model)
        params_min = l
        self.add_min_config()
        for value in self.values[1:]:
            params_max = int(value * (u - l) + l)
            config = self.constrained_search(params_min, params_max, model)
            if config is not None:
                self.grid.append(config)
            params_min = params_max
        self.add_max_config()

    def constrained_search(self, params_min, params_max, model):
        for _ in range(self.n_trials):
            config = self.sample()
            model.set_sub_network(**config)
            params = compute_parameters(model)
            model.reset_super_network()
            if params >= params_min and params < params_max:
                # print(params)
                return config

    def get_smallest_params(self, model):
        config = {}
        for hp_name, hp in self.search_space.config_space.items():
            if isinstance(hp, Categorical):
                config[hp_name] = min(hp.categories)
            else:
                config[hp_name] = hp.lower
        model.set_sub_network(**self.search_space.cast(config))
        params = compute_parameters(model)
        model.reset_super_network()
        return params


if __name__ == "__main__":
    from litgpt.config import Config

    config = Config(name="EleutherAI/pythia-410m")
    print(config)
    from search_spaces import HWGPTBench, MEDIUM
    from whittle.models.gpt.model import GPT

    search_space = HWGPTBench(gpt_model_specification=config)
    # increase n_trials to sample architectures in all parameter bins
    sampler = FixParamGridSampler(search_space=search_space, n_trials=1000)
    config.fix_head_size = True
    model = GPT(config)
    sampler.initialize_grid(model)
    print(len(sampler.grid))
    for config in sampler.grid:
        print(config)
    search_space = MEDIUM(gpt_model_specification=config)
    config = Config(name="EleutherAI/pythia-410m")
    sampler = CalibFixGridSampler(
        "checkpoints/EleutherAI/pythia-410m", "medium", search_space=search_space
    )
    config.fix_head_size = True
    for config in sampler.grid:
        print(config)
