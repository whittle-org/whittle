from __future__ import annotations

import yaml
import lightning as  L
from whittle.lora.config import LoRAConfig as Config
import torch
from litgpt.lora import LoRALayer, lora_filter
from whittle.lora.lora_gpt import GPT
from pathlib import Path
from litgpt.utils import check_valid_checkpoint_dir, extend_checkpoint_dir
from typing import Any, Dict, Tuple
from tqdm import tqdm


def merge_lora_weights(model: GPT, verbose: bool = False) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in tqdm(model.modules()):
        if isinstance(module, LoRALayer):
            if verbose:
                print(module, module.merged)
            module.merge()


def merge_lora(
    checkpoint_dir: Path,
    pretrained_checkpoint_dir: Path,
    precision: str | None = None,
    accelerator: str = "cpu",
    overwrite: bool = False,
) -> None:
    """Merges the LoRA weights with the base model.

    See ``whittle finetune lora``.

    Creates a new ``lit_model.pth`` file by merging the LoRA weights (``lit_model.pth.lora``)
    with the original checkpoint weights.

    Arguments:
        checkpoint_dir: Path to the checkpoint directory with trained LoRA weights, which is the output of
            ``whittle finetune lora``.
        pretrained_checkpoint_dir: Optional path to the checkpoint directory with the weights of the base model
            corresponding to the LoRA checkpoint. By default, this will automatically be inferred from the metadata
            in the given `checkpoint_dir` directory. Only set this if the base model's checkpoint directory
            has moved or was renamed.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
        accelerator: Optional accelerator setting to instantiate the model weights on (passed to L.Fabric). By default, this will
            be set to "cpu".
        overwrite: Whether to overwrite the existing ``lit_model.pth`` file in the checkpoint directory. By default,
            this will be set to False.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    if pretrained_checkpoint_dir is not None:
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)

    check_valid_checkpoint_dir(checkpoint_dir, model_filename="lit_model.pth.lora")
    if pretrained_checkpoint_dir is not None:
        check_valid_checkpoint_dir(pretrained_checkpoint_dir)
    if (checkpoint_dir / "lit_model.pth").is_file():
        if not overwrite:
            print("LoRA weights have already been merged in this checkpoint.")
            return
        else:
            print("Overwriting the existing merged weights.")

    if pretrained_checkpoint_dir is None:
        pretrained_checkpoint_dir = meta_pretrained_checkpoint_dir
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)

    lora_params, meta_pretrained_checkpoint_dir, lora_precision = load_lora_metadata(
        checkpoint_dir
    )
    precision = lora_precision
    fabric = L.Fabric(devices=1, accelerator=accelerator, precision=precision)
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **lora_params)
    config.fix_head_size = True
    with fabric.init_module(), torch.device("meta"):
        model = GPT(config)
        # we don't care about these to perform merging
        model.cos = None
        model.sin = None

    lora_path = checkpoint_dir / "lit_model.pth.lora"
    pretrained_checkpoint = torch.load(
        str(pretrained_checkpoint_dir / "lit_model.pth"), mmap=True
    )
    pretrained_checkpoint = {
            k.replace('attn.weight', 'attn.linear.linear.weight'): v
            for k, v in pretrained_checkpoint.items()
    }

    lora_checkpoint = torch.load(str(lora_path), mmap=True)
    lora_checkpoint = lora_checkpoint.get("model", lora_checkpoint)

    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint)
    model.load_state_dict(pretrained_checkpoint, assign=True)
    # since LoRA finetuning only saves the LoRA weights, we treat the lora weights dtype as the expected dtype
    lora_dtype = next(iter(lora_checkpoint.values())).dtype
    model.to(dtype=lora_dtype, device="cpu")
    model.eval()
    merge_lora_weights(model)

    # Remove LoRA parameters and the LoRA linear substring
    state_dict = {
        # linear.linear is in lora_qkv_linear.py, linear is in lora_linear.py, embedding is in lora_embedding.py
        k.replace("linear.linear.", "")
        .replace("linear.", "")
        .replace("embedding.", ""): v
        for k, v in model.state_dict().items()
        if not lora_filter(k, v)
    }
    save_path = checkpoint_dir / "lit_model.pth"
    torch.save(state_dict, save_path)

    fabric.print(f"Saved merged weights to {str(checkpoint_dir / 'lit_model.pth')!r}")


def load_lora_metadata(
    checkpoint_dir: Path,
) -> Tuple[Dict[str, Any], Path, str | None]:
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
