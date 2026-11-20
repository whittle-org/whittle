from __future__ import annotations

from pathlib import Path
from typing import Literal

from jsonargparse import CLI
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import DataModule, TinyStories

from whittle import distill_model
from whittle.args import DistillArgs


def setup(
    out_dir: Path = Path("out/distill"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    teacher_checkpoint_dir: Path | None = None,
    student_dir: Path | None = None,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(5e8),
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    distill: DistillArgs = DistillArgs(
        method="logits",
        temperature=10,
        alpha=0.3,
        beta=0.7,
        loss="forward_kld",
        weight_scheme="other",
    ),
    eval: EvalArgs = EvalArgs(interval=50, max_iters=100, initial_validation=True),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 42,
    min_ratio: float = 0.6,
    max_ratio: float = 0.61,
    random_init_student: bool = False,
    strategy: Literal["ddp", "fsdp"] = "fsdp",
):
    """Train a (random) subnet of the teacher model using knowledge distillation.

    A wrapper around ``whittle.distill_model`` with ``subnetwork_student=True``, kept for
    existing commands. Prefer ``whittle.distill_model``. Pre-computed teacher logits
    (``use_saved_logits``, ``teacher_logits_dir``) are no longer supported.

    Arguments:
        out_dir: Directory in which to save checkpoints and logs.
        precision: The precision to use for finetuning.
        teacher_checkpoint_dir: Path to a checkpoint directory to initialize the teacher model from.
        student_dir: Optional path to a directory to initialize the student model from.
            If given, no sub-network is sampled.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyStories``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        distill: Distillation-related arguments. See ``whittle.args.DistillArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        min_ratio: Minimum allowed ratio (student_params / teacher_params).
        max_ratio: Maximum allowed ratio (student_params / teacher_params).
        random_init_student: If True, the student sub-network will be randomly initialized instead of inheriting weights from the teacher.
        strategy: ``"fsdp"`` or ``"ddp"`` to train on several devices. See ``whittle.distill_model.setup``.
    """
    distill_model.setup(
        out_dir=out_dir,
        precision=precision,
        teacher_checkpoint_dir=teacher_checkpoint_dir,
        initial_checkpoint_dir=student_dir,
        config_path=(
            str(student_dir / "model_config.yaml") if student_dir is not None else None
        ),
        subnetwork_student=student_dir is None,
        data=TinyStories() if data is None else data,
        train=train,
        distill=distill,
        eval=eval,
        optimizer=optimizer,
        devices=devices,
        num_nodes=num_nodes,
        tokenizer_dir=tokenizer_dir,
        logger_name=logger_name,
        seed=seed,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        random_init_student=random_init_student,
        strategy=strategy,
    )


if __name__ == "__main__":
    CLI(setup)
