import torch
from litgpt import Config
from whittle.models.gpt import GPT
import sys
import os
import numpy as np
import pathlib
from whittle.eval.whittle_llms import WhittleLM
from whittle.eval.utils import convert_and_evaluate
from litgpt.scripts.download import download_from_hub
import pytest
from lm_eval import tasks
from lm_eval.api.instance import Instance
import json


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    # img = compute_expensive_image()
    checkpoint_dir = tmp_path_factory.getbasetemp()
    download_from_hub(repo_id="EleutherAI/pythia-70m", checkpoint_dir=checkpoint_dir)
    return pathlib.Path(checkpoint_dir) / "EleutherAI" / "pythia-70m"


@pytest.fixture(scope="session")
def out_dir(tmp_path_factory):
    # img = compute_expensive_image()
    out_dir = tmp_path_factory.getbasetemp()

    return pathlib.Path(out_dir) / "out_dir"


def test_api(checkpoint_dir, out_dir):
    torch.use_deterministic_algorithms(True)
    config = Config.from_file(str(checkpoint_dir / "model_config.yaml"))
    config.fix_head_size = True
    config.model_type = "gpt"
    config.tie_embeddings = False
    gpt = GPT(config)
    gpt.device = "cpu"
    gpt.name_or_path = "EleutherAI/pythia-70m"

    # model = LitGPT(config)
    gpt.load_state_dict(torch.load(str(checkpoint_dir / "lit_model.pth")))
    eval_gpt = WhittleLM(pretrained=gpt, dtype="float32")

    task_manager = tasks.TaskManager()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    task_list = task_manager.load_task_or_group(["arc_easy", "gsm8k", "wikitext"])
    multiple_choice_task = task_list["arc_easy"]
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: list[Instance] = multiple_choice_task.instances
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.set_fewshot_seed(1234)  # fewshot random generator seed
    rng = torch.Generator(device="cpu")
    rng.manual_seed(15485863)
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until: list[Instance] = generate_until_task.instances
    rolling_task = task_list["wikitext"]  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: list[Instance] = rolling_task.instances
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
    version_minor = sys.version_info.minor
    res = eval_gpt.loglikelihood(MULTIPLE_CH)
    _RES, _res = MULTIPLE_CH_RES, [r[0] for r in res]
    # log samples to CI
    dir_path = out_dir
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"outputs_log_{version_minor}.txt"
    file_path = file_path.resolve()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in _res))
    assert np.allclose(_res, _RES, atol=1e-2)
    # check indices for Multiple Choice
    argmax_RES, argmax_res = (
        np.argmax(np.array(_RES).reshape(-1, 4), axis=1),
        np.argmax(np.array(_res).reshape(-1, 4), axis=1),
    )
    res = eval_gpt.loglikelihood_rolling(ROLLING)
    assert np.allclose(res, ROLLING_RES, atol=1e-1)
    TEST_STRING = "foo bar"
    res = eval_gpt.tok_encode(TEST_STRING)
    assert res == [12110, 2534]
    res = eval_gpt.tok_decode([12110, 2534])
    assert res == TEST_STRING
    res = eval_gpt.tok_batch_encode([TEST_STRING, "bar foo"])[0].tolist()
    assert res == [[12110, 2534], [2009, 17374]]
    context = eval_gpt.tok_batch_encode([TEST_STRING])[0]
    res = eval_gpt._model_generate(context, max_length=10, stop=["\n\n"])[0]
    res = eval_gpt.tok_decode(res)
    assert res == "foo bar\n<bazhang>!info bar"
    res = eval_gpt.generate_until(generate_until)
    assert res == generate_until_RES
    assert (argmax_RES == argmax_res).all()
    convert_and_evaluate(
        gpt,
        out_dir=out_dir,
        device=None,
        dtype=torch.float32,
        limit=10,
        tasks="logiqa",
        batch_size=1,  # Test for non-positive integer
    )
    with open(str(out_dir / "results.json"), "r") as f:
        results = json.load(f)
    acc_api = results["results"]["hellaswag"]["acc,none"]
    stderr_api = results["results"]["hellaswag"]["acc_stderr,none"]
    import litgpt.eval.evaluate as module

    module.convert_and_evaluate(
        checkpoint_dir,
        out_dir=out_dir,
        device=None,
        dtype=torch.float32,
        limit=10,
        tasks="logiqa",
        force_conversion=True,
        batch_size=1,  # Test for non-positive integer
    )
    with open(str(out_dir / "results.json"), "r") as f:
        results = json.load(f)
    acc_lit = results["results"]["hellaswag"]["acc,none"]
    stderr_lit = results["results"]["hellaswag"]["acc_stderr,none"]
    assert acc_api == acc_lit
    assert stderr_api == stderr_lit
