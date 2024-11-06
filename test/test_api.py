from __future__ import annotations

import json
import pathlib
import sys

import litgpt.eval.evaluate as module
import numpy as np
import pytest
import torch
from litgpt import Config
from litgpt.scripts.download import download_from_hub
from lm_eval import tasks
from lm_eval.api.instance import Instance

from whittle.eval.utils import convert_and_evaluate
from whittle.eval.whittle_llms import WhittleLM
from whittle.models.gpt import GPT


def copy_subnetwork_weights(sub_network, super_network):
    """
    Copies weights from the sub_network to the super_network.

    Parameters:
    - sub_network: the smaller neural network with the weights to copy.
    - super_network: the larger neural network to which weights will be copied.
    """

    # Ensure both networks are in evaluation mode to avoid any changes in weights during copying.
    sub_network.eval()
    super_network.eval()

    # Get the state dictionaries of both networks
    sub_state_dict = sub_network.state_dict()
    super_state_dict = super_network.state_dict()

    # Iterate over the subnetwork's state dictionary and copy weights to the supernetwork
    for layer_name, sub_weight in sub_state_dict.items():
        if layer_name in super_state_dict:
            super_weight = super_state_dict[layer_name]

            # Ensure the subnetwork's weight can fit into the supernetwork's weight tensor
            if sub_weight.size() == super_weight.size():
                super_state_dict[layer_name] = sub_weight
            else:
                # Copy the sub_weight values into the corresponding part of super_weight
                if len(sub_weight.shape) == 1:
                    super_weight[0 : sub_weight.shape[0]] = sub_weight
                elif len(sub_weight.shape) == 2:
                    super_weight[0 : sub_weight.shape[0], 0 : sub_weight.shape[1]] = (
                        sub_weight
                    )
                super_state_dict[layer_name] = super_weight
        else:
            raise KeyError(f"Layer {layer_name} not found in super-network.")

    # Load the modified state dictionary back into the supernetwork
    super_network.load_state_dict(super_state_dict)
    return super_network


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    # img = compute_expensive_image()
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-70m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-70m"


@pytest.fixture(scope="session")
def checkpoint_dir_14m(tmp_path_factory):
    # img = compute_expensive_image()
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-14m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-14m"


@pytest.fixture(scope="session")
def out_dir(tmp_path_factory):
    # img = compute_expensive_image()
    out_dir = tmp_path_factory.getbasetemp()

    return pathlib.Path(out_dir) / "out_dir"


class Test_WhittleLM:
    # torch.use_deterministic_algorithms(True)
    task_manager = tasks.TaskManager()
    task_list = task_manager.load_task_or_group(["arc_easy", "gsm8k", "wikitext"])
    version_minor = sys.version_info.minor
    multiple_choice_task = task_list["arc_easy"]  # type: ignore
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: list[Instance] = multiple_choice_task.instances
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.set_fewshot_seed(1234)  # fewshot random generator seed
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until: list[Instance] = generate_until_task.instances
    rolling_task = task_list["wikitext"]  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: list[Instance] = rolling_task.instances
    TEST_STRING = "foo bar"
    MULTIPLE_CH_RES = [
        -41.902435302734375,
        -42.939308166503906,
        -33.914180755615234,
        -37.07139205932617,
        -22.95258331298828,
        -20.342208862304688,
        -14.818366050720215,
        -27.942853927612305,
        -15.80704116821289,
        -15.936427116394043,
        -13.052018165588379,
        -18.04828453063965,
        -13.345029830932617,
        -13.366025924682617,
        -12.127134323120117,
        -11.872495651245117,
        -47.10598373413086,
        -47.76410675048828,
        -36.4406852722168,
        -50.0289421081543,
        -16.72093963623047,
        -18.535587310791016,
        -26.46993637084961,
        -20.355995178222656,
        -17.757919311523438,
        -21.80595588684082,
        -33.1990852355957,
        -39.28636932373047,
        -14.759679794311523,
        -16.753942489624023,
        -11.486852645874023,
        -15.42177677154541,
        -13.15798282623291,
        -15.887393951416016,
        -15.28614616394043,
        -12.339089393615723,
        -44.59441375732422,
        -55.40888214111328,
        -52.70050811767578,
        -56.25089645385742,
    ]
    generate_until_RES = [
        " The average of $2.50 each is $",
        " A robe takes 2 bolts of blue fiber and half",
        " $50,000 in repairs.\n\nQuestion",
        " He runs 1 sprint 3 times a week.",
        " They feed each of her chickens three cups of mixed",
        " The price of the glasses is $5, but",
        " The total percentage of students who said they like to",
        " Carla is downloading a 200 GB file. Normally",
        " John drives for 3 hours at a speed of 60",
        " Eliza sells 4 tickets to 5 friends so she",
    ]
    ROLLING_RES = [
        -3603.6328125,
        -19779.23974609375,
        -8834.16455078125,
        -27967.591796875,
        -7636.794982910156,
        -9491.93505859375,
        -41043.4248046875,
        -8397.689819335938,
        -45969.47155761719,
        -7158.90625,
    ]

    def test_logliklihood(self, checkpoint_dir, out_dir) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"

        # model = LitGPT(config)
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        LM = WhittleLM(pretrained=gpt, dtype="float32", device=device)
        res = LM.loglikelihood(self.MULTIPLE_CH)
        _RES, _res = self.MULTIPLE_CH_RES, [r[0] for r in res]
        # log samples to CI
        dir_path = out_dir
        dir_path.mkdir(parents=True, exist_ok=True)
        assert np.allclose(_res, _RES, atol=1e-2)
        # check indices for Multiple Choice
        argmax_RES, argmax_res = (
            np.argmax(np.array(_RES).reshape(-1, 4), axis=1),
            np.argmax(np.array(_res).reshape(-1, 4), axis=1),
        )
        assert (argmax_RES == argmax_res).all()

    def test_generate_until(self, checkpoint_dir) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"

        # model = LitGPT(config)
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        LM = WhittleLM(pretrained=gpt, dtype="float32", device=device)
        res = LM.generate_until(self.generate_until)
        assert res == self.generate_until_RES

    def test_logliklihood_rolling(self, checkpoint_dir) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"

        # model = LitGPT(config)
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        LM = WhittleLM(pretrained=gpt, dtype="float32", device=device)
        res = LM.loglikelihood_rolling(self.ROLLING)
        assert np.allclose(res, self.ROLLING_RES, atol=1e-1)

    def test_toc_encode(self, checkpoint_dir) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"

        # model = LitGPT(config)
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        LM = WhittleLM(pretrained=gpt, dtype="float32", device=device)
        res = LM.tok_encode(self.TEST_STRING)
        assert res == [12110, 2534]

    def test_toc_decode(self, checkpoint_dir) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"

        # model = LitGPT(config)
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        LM = WhittleLM(pretrained=gpt, dtype="float32", device=device)
        res = LM.tok_decode([12110, 2534])
        assert res == self.TEST_STRING

    def test_batch_encode(self, checkpoint_dir) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"

        # model = LitGPT(config)
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        LM = WhittleLM(pretrained=gpt, dtype="float32", device=device)
        res = LM.tok_batch_encode([self.TEST_STRING, "bar foo"])[0].tolist()
        assert res == [[12110, 2534], [2009, 17374]]

    def test_model_generate(self, checkpoint_dir) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"

        # model = LitGPT(config)
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        LM = WhittleLM(pretrained=gpt, dtype="float32", device=device)
        context = LM.tok_batch_encode([self.TEST_STRING])[0].to(device)
        res = LM._model_generate(context, max_length=10, stop=["\n\n"])
        res = LM.tok_decode(res[0])
        assert res == "foo bar\n<bazhang> !info bar"

    def test_evaluate(self, checkpoint_dir, out_dir):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"

        # model = LitGPT(config)
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        convert_and_evaluate(
            gpt,
            out_dir=out_dir,
            device=str(device),
            dtype=torch.float32,
            limit=10,
            tasks="logiqa",
            batch_size=1,  # Test for non-positive integer
        )
        with open(str(out_dir / "results.json")) as f:
            results = json.load(f)
        acc_api = results["results"]["logiqa"]["acc,none"]
        stderr_api = results["results"]["logiqa"]["acc_stderr,none"]

        module.convert_and_evaluate(
            checkpoint_dir,
            out_dir=out_dir,
            device=str(device),
            dtype=torch.float32,
            limit=10,
            tasks="logiqa",
            force_conversion=True,
            batch_size=1,  # Test for non-positive integer
        )
        with open(str(out_dir / "results.json")) as f:
            results = json.load(f)
        acc_lit = results["results"]["logiqa"]["acc,none"]
        stderr_lit = results["results"]["logiqa"]["acc_stderr,none"]
        assert acc_api == acc_lit
        assert stderr_api == stderr_lit

    def test_compare_litgpt(self, checkpoint_dir, checkpoint_dir_14m, out_dir):
        torch.manual_seed(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
        config.fix_head_size = True
        config.model_type = "gpt"
        config.tie_embeddings = False
        gpt = GPT(config).to(device)
        gpt.name_or_path = "EleutherAI/pythia-70m"
        gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
        config_14m = Config.from_file(str(checkpoint_dir_14m / "model_config.yaml"))
        config_14m.fix_head_size = True
        config_14m.model_type = "gpt"
        config_14m.tie_embeddings = False
        gpt_14m = GPT(config_14m).to(device)
        gpt_14m.name_or_path = "EleutherAI/pythia-14m"
        gpt_14m.load_state_dict(torch.load(str(checkpoint_dir_14m / "lit_model.pth")))
        gpt = copy_subnetwork_weights(gpt_14m, gpt)
        gpt.max_seq_length = config_14m.block_size
        gpt.set_sub_network(
            sub_network_n_embd=config_14m.n_embd,
            sub_network_intermediate_size=config_14m.intermediate_size,
            sub_network_num_heads=config_14m.n_head,
            sub_network_n_layers=config_14m.n_layer,
            sub_network_query_groups=config_14m.n_query_groups,
            sub_network_head_size=config_14m.head_size,
        )
        convert_and_evaluate(
            gpt,
            out_dir=out_dir,
            device=str(device),
            dtype=torch.float32,
            limit=10,
            tasks="logiqa",
            batch_size=1,  # Test for non-positive integer
        )
        with open(str(out_dir / "results.json")) as f:
            results = json.load(f)
        acc_api = results["results"]["logiqa"]["acc,none"]
        stderr_api = results["results"]["logiqa"]["acc_stderr,none"]

        module.convert_and_evaluate(
            checkpoint_dir_14m,
            out_dir=out_dir,
            device=str(device),
            dtype=torch.float32,
            limit=10,
            tasks="logiqa",
            force_conversion=True,
            batch_size=1,  # Test for non-positive integer
        )
        with open(str(out_dir / "results.json")) as f:
            results = json.load(f)
        acc_lit = results["results"]["logiqa"]["acc,none"]
        stderr_lit = results["results"]["logiqa"]["acc_stderr,none"]
        assert acc_api == acc_lit
        assert stderr_api == stderr_lit
