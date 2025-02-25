from syne_tune.config_space import randint, lograndint, choice
import math


class SMALL:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "n_embd": lograndint(64, gpt_model_specification.n_embd),
            "n_layers": randint(1, gpt_model_specification.n_layer),
            "heads": randint(1, gpt_model_specification.n_head),
            "intermediate_size": randint(1, gpt_model_specification.intermediate_size),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["n_embd"],
            "sub_network_intermediate_size": [config[f"intermediate_size"]]
            * config["n_layers"],
            "sub_network_num_heads": [config[f"heads"]] * config["n_layers"],
            "sub_network_n_layers": config["n_layers"],
        }


class MEDIUM:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "n_embd": lograndint(64, gpt_model_specification.n_embd),
            "n_layers": randint(1, gpt_model_specification.n_layer),
        }

        for li in range(gpt_model_specification.n_layer):
            self.config_space[f"heads_{li}"] = randint(
                1, gpt_model_specification.n_head
            )
            self.config_space[f"intermediate_size_{li}"] = randint(
                1, gpt_model_specification.intermediate_size
            )

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["n_embd"],
            "sub_network_intermediate_size": [
                config[f"intermediate_size_{li}"] for li in range(config["n_layers"])
            ],
            "sub_network_num_heads": [
                config[f"heads_{li}"] for li in range(config["n_layers"])
            ],
            "sub_network_n_layers": config["n_layers"],
        }


class HWGPTBench:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": lograndint(1, gpt_model_specification.n_embd),
            "num_heads": randint(1, gpt_model_specification.n_head),
            "mlp_ratio": randint(1, 4),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": [
                config[f"mlp_ratio"] * config["embed_dim"]
                for _ in range(config["depth"])
            ],
            "sub_network_num_heads": [
                config[f"num_heads"] for _ in range(config["depth"])
            ],
            "sub_network_n_layers": config["depth"],
        }


class HWGPTBenchLlama:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": lograndint(1, gpt_model_specification.n_embd),
            "num_heads": choice([gpt_model_specification.n_head]),
            "mlp_ratio": randint(
                1,
                gpt_model_specification.intermediate_size
                // gpt_model_specification.n_embd,
            ),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": [
                config[f"mlp_ratio"] * config["embed_dim"]
                for _ in range(config["depth"])
            ],
            "sub_network_num_heads": [
                config[f"num_heads"] for _ in range(config["depth"])
            ],
            "sub_network_n_layers": config["depth"],
        }


class HWGPTBenchLlama2:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": lograndint(64, gpt_model_specification.n_embd),
            "num_heads": choice([gpt_model_specification.n_head]),
            "mlp_ratio": choice(
                [1.0, 2.0, 3.0, 3.5]
            ),  # gpt_model_specification.intermediate_size//gpt_model_specification.n_embd),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": [
                int(config[f"mlp_ratio"] * config["embed_dim"])
                for _ in range(config["depth"])
            ],
            "sub_network_num_heads": [
                config[f"num_heads"] for _ in range(config["depth"])
            ],
            "sub_network_n_layers": config["depth"],
        }


class LlamaSmall:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": choice(
                [
                    2**i
                    for i in range(
                        5, int(math.log(gpt_model_specification.n_embd, 2)) + 1
                    )
                ]
            ),
            "num_heads": choice([8, 16, 32]),
            "mlp_ratio": choice(
                [1.0, 2.0, 3.0, 3.5]
            ),  # gpt_model_specification.intermediate_size//gpt_model_specification.n_embd),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": [
                int(config[f"mlp_ratio"] * config["embed_dim"])
                for _ in range(config["depth"])
            ],
            "sub_network_num_heads": [
                config[f"num_heads"] for _ in range(config["depth"])
            ],
            "sub_network_n_layers": config["depth"],
        }


class LlamaHeads:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": choice(
                [
                    2**i
                    for i in range(
                        5, int(math.log(gpt_model_specification.n_embd, 2)) + 1
                    )
                ]
            ),
            "num_heads": choice([8, 16, 32]),
            "mlp_ratio": choice(
                [1.0, 2.0, 3.0, 3.5]
            ),  # gpt_model_specification.intermediate_size//gpt_model_specification.n_embd),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": int(config[f"mlp_ratio"] * config["embed_dim"]),
            "sub_network_num_heads": config[f"num_heads"],
            "sub_network_n_layers": config["depth"],
        }


class LlamaJoint:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": choice(
                [
                    2**i
                    for i in range(
                        5, int(math.log(gpt_model_specification.n_embd, 2)) + 1
                    )
                ]
            ),
            "num_heads": choice([8, 16, 32]),
            "mlp_ratio": choice(
                [1.0, 2.0, 3.0, 3.5]
            ),  # gpt_model_specification.intermediate_size//gpt_model_specification.n_embd),
            "head_size": choice([8, 16, 32, 64, 128]),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        if "embed_dim" not in config:
            return config
        config_return = {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": int(config[f"mlp_ratio"] * config["embed_dim"]),
            "sub_network_num_heads":config[f"num_heads"],
            "sub_network_n_layers": config["depth"],
            "sub_network_head_size": config["head_size"],
        }
        if "sampled_layer_indices" in config:
            config_return["sampled_layer_indices"] = config["sampled_layer_indices"]
        return config_return


class LlamaHeadSize:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": choice(
                [
                    2**i
                    for i in range(
                        5, int(math.log(gpt_model_specification.n_embd, 2)) + 1
                    )
                ]
            ),
            "num_heads": choice([32]),
            "mlp_ratio": choice(
                [1.0, 2.0, 3.0, 3.5]
            ),  # gpt_model_specification.intermediate_size//gpt_model_specification.n_embd),
            "head_size": choice([8, 16, 32, 64, 128]),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size":int(config[f"mlp_ratio"] * config["embed_dim"]),
            "sub_network_num_heads": config[f"num_heads"],
            "sub_network_n_layers": config["depth"],
            "sub_network_head_size": config["head_size"],
        }


class HWGPTBenchLlama27b:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": lograndint(1, gpt_model_specification.n_embd),
            "num_heads": choice([gpt_model_specification.n_head]),
            "mlp_ratio": choice(
                [1.0, 2.0, 2.5, 2.6875]
            ),  # gpt_model_specification.intermediate_size//gpt_model_specification.n_embd),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": int(config[f"mlp_ratio"] * config["embed_dim"]),
            "sub_network_num_heads":config[f"num_heads"],
            "sub_network_n_layers": config["depth"],
        }


class HWGPTBenchLlama213b:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": lograndint(1, gpt_model_specification.n_embd),
            "num_heads": choice([gpt_model_specification.n_head]),
            "mlp_ratio": choice(
                [1.0, 2.0, 2.5, 2.7]
            ),  # gpt_model_specification.intermediate_size//gpt_model_specification.n_embd),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": [
                int(config[f"mlp_ratio"] * config["embed_dim"])
                for _ in range(config["depth"])
            ],
            "sub_network_num_heads": [
                config[f"num_heads"] for _ in range(config["depth"])
            ],
            "sub_network_n_layers": config["depth"],
        }


class HWGPTBenchPhi3:
    def __init__(self, gpt_model_specification):
        self.config_space = {
            "embed_dim": lograndint(1, gpt_model_specification.n_embd),
            "num_heads": choice([gpt_model_specification.n_head]),
            "mlp_ratio": choice(
                [1.0, 2.0, 2.5, 2.6666666666666665]
            ),  # gpt_model_specification.intermediate_size//gpt_model_specification.n_embd),
            "depth": randint(1, gpt_model_specification.n_layer),
        }

    @staticmethod
    def cast(config):
        return {
            "sub_network_n_embd": config["embed_dim"],
            "sub_network_intermediate_size": [
                int(config[f"mlp_ratio"] * config["embed_dim"])
                for _ in range(config["depth"])
            ],
            "sub_network_num_heads": [
                config[f"num_heads"] for _ in range(config["depth"])
            ],
            "sub_network_n_layers": config["depth"],
        }


search_spaces = {
    "small": SMALL,
    "medium": MEDIUM,
    "hw_gpt_bench": HWGPTBench,
    "llama": HWGPTBenchLlama,
    "llama2": HWGPTBenchLlama2,
    "llama_head_size": LlamaHeadSize,
    "llama_joint": LlamaJoint,
    "llama_heads": LlamaHeads,
    "llama2-7b": HWGPTBenchLlama27b,
    "llama2-13b": HWGPTBenchLlama213b,
    "phi-3": HWGPTBenchPhi3,
}
