import torch
from whittle.lora.lora_gpt import GPT as LoRAGPT
from whittle.models.gpt import GPT
from litgpt.lora import mark_only_lora_as_trainable, LoRALayer, lora_filter
from litgpt.utils import save_config
from whittle.lora.lora_qkv_linear import LoRAQKVLinear
from whittle.lora.lora_embedding import LoRAEmbedding
from whittle.lora.config import LoRAConfig as Config
import lightning as L
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from litgpt.utils import check_valid_checkpoint_dir, extend_checkpoint_dir
from whittle.lora.utils import merge_lora
from whittle.lora.lora_qkv_linear import LoRAQKVLinear
from whittle.lora.lora_embedding import LoRAEmbedding
from whittle.lora.lora_linear import LoRALinearProj, LoRALinearQKV, LoRALinear


def load_lora_metadata(
    checkpoint_dir: Path,
) -> Tuple[Dict[str, Any], Path, Optional[str]]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"The path {str(hparams_file)!r} is not a valid checkpoint directory. It is missing a"
            f" `hyperparameters.yaml` file. Please point to the checkpoint directory that was produced by"
            f" the `litgpt/finetune/lora.py` script."
        )

    with open(hparams_file, "r", encoding="utf-8") as file:
        hparams = yaml.safe_load(file)

    lora_params = {k: v for k, v in hparams.items() if k.startswith("lora_")}
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")
    return lora_params, pretrained_checkpoint_dir, precision


def test_merge_extract():
    checkpoint_dir = Path('lora_temp/')

    lora_params, meta_pretrained_checkpoint_dir, lora_precision = load_lora_metadata(
        checkpoint_dir
    )
    precision = lora_precision

    pretrained_checkpoint_dir = meta_pretrained_checkpoint_dir
    pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)
    pretrained_checkpoint_dir = '../do-not-touch/compressing_llms' / pretrained_checkpoint_dir

    print(lora_params)
    fabric = L.Fabric(devices=1, precision=precision, accelerator="cpu")
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **lora_params)
    config.fix_head_size = True

    def get_lora_model():
        with fabric.init_module(), torch.device("meta"):
            model = LoRAGPT(config)

        lora_path = checkpoint_dir / "lit_model.pth.lora"
        pretrained_checkpoint = torch.load(
            str(pretrained_checkpoint_dir / "lit_model.pth"), mmap=True
        )
        lora_checkpoint = torch.load(str(lora_path), mmap=True)
        lora_checkpoint = lora_checkpoint.get("model", lora_checkpoint)
        pretrained_checkpoint = {
            k.replace('attn.weight', 'attn.linear.linear.weight'): v
            for k, v in pretrained_checkpoint.items()
        }

        # Merge LoRA weights into the base model
        pretrained_checkpoint.update(lora_checkpoint)
        model.load_state_dict(pretrained_checkpoint, assign=True)

        model.reset_super_network()
        return model

    pre_model = get_lora_model()
    pre_model.eval()
    pre_model.to('cuda:1')

    # seed torch
    #torch.manual_seed(40)
    #random_input = torch.randint(0, 64, size=(2, 4), dtype=torch.int64)
    #pre_merge_y = pre_model(random_input)

    merge_lora(checkpoint_dir, pretrained_checkpoint_dir, overwrite=True)

    # load merged weights
    with fabric.init_module(), torch.device("meta"):
        post_merge_model = GPT(config)

    ckp = torch.load(str(checkpoint_dir / "lit_model.pth"), mmap=True)
    post_merge_model.load_state_dict(ckp, assign=True)
    post_merge_model.to('cuda:1')
    post_merge_model.eval()

    random_input = torch.randint(0, config.vocab_size, size=(2, 4), dtype=torch.int64).to('cuda:1')
    post_merge_y = post_merge_model(random_input)
    pre_merge_y = pre_model(random_input)

    pre_modules = []
    pre_modules.append(pre_model.lm_head)
    pre_modules.append(pre_model.transformer.wte)
    for i, block in enumerate(pre_model.transformer.h):
        print(i)
        pre_modules.append(block.attn.attn)
        pre_modules.append(block.attn.proj)
        pre_modules.append(block.mlp.fc_1)
        pre_modules.append(block.mlp.fc_2)
        pre_modules.append(block.mlp.proj)
        # TODO also mlp and lm_head

    post_modules = []
    post_modules.append(post_merge_model.lm_head)
    post_modules.append(post_merge_model.transformer.wte)
    for i, block in enumerate(post_merge_model.transformer.h):
        print(i)
        post_modules.append(block.attn.attn)
        post_modules.append(block.attn.proj)
        post_modules.append(block.mlp.fc_1)
        post_modules.append(block.mlp.fc_2)
        post_modules.append(block.mlp.proj)
        # TODO also mlp and lm_head

    print(config.head_size, config.n_head, config.n_query_groups)
    # TODO bf16

    for pre_module, post_module in zip(pre_modules, post_modules):
        if isinstance(pre_module, (LoRALinear, LoRALinearProj)):
            input_data = torch.randn(8, pre_module.in_features).to(torch.bfloat16)
        elif isinstance(pre_module, LoRAQKVLinear):
            input_data = torch.randn((2, 4, config.n_embd)).to(torch.bfloat16)
        elif isinstance(pre_module, LoRAEmbedding):
            input_data = torch.randint(0, config.vocab_size, size=(2, 4), dtype=torch.int64)
        else:
            continue
        input_data = input_data.to('cuda:1')
        pre_y = pre_module(input_data)
        if pre_module.r > 0:
            pre_module.get_lora_AB()
        post_y = post_module(input_data)
        if not torch.allclose(pre_y, post_y):
            #print(pre_module.__class__.__name__, 'failed')
            diff = (pre_y - post_y).flatten()
            print(torch.sort(diff[diff != 0])[0])
            print(len(diff[diff != 0]), len(diff))
        else:
            pass
            #print(pre_module.__class__.__name__, 'passed')

    print((pre_merge_y - post_merge_y))
    print("pre: ", pre_merge_y)
    print("post: ", post_merge_y)
    print(pre_merge_y.shape)
    assert torch.allclose(pre_merge_y, post_merge_y)


if __name__ == "__main__":
    test_merge_extract()
