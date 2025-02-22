import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Tokenizer

from litgpt import Config
from whittle.models.gpt.model import GPT
from whittle.args import DistillArgs
from whittle.distillation.utils import (
    create_dataloader,
    TeacherLogitsLoader
)
from whittle.distillation.knowledge_distill import KD
from jsonargparse import CLI
from pathlib import Path


def evaluate(model: torch.nn.Module, dataloader, device: str) -> float:
    model.to(device)
    model.eval()
    total_loss = 0.0
    num_batches = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Average loss: {avg_loss:.4f}")
    return avg_loss

def main(
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    seq_len: int = 64,
    batch_size: int = 5,
    verbose: bool = False,
    save_dir: str = 'save_dir',
    dataset: str = 'tiny_stories',
    dataset_path: str = 'roneneldan/TinyStories',
    teacher_ckpt_path: Path = Path('checkpoints/teacher_checkpoint.pth'),
    teacher_config_path: Path = Path('checkpoints/teacher_config.yaml'),
    student_ckpt_path: Path = Path('checkpoints/student_checkpoint.pth'),
    student_config_path: Path = Path('checkpoints/student_config.yaml'),

    distill: DistillArgs = DistillArgs(
        method='logits',
        on_cluster=False,
        use_topk_logits=False,
        use_precomputed_logits=False,
        temperature=0.5,
        alpha=0.5
    )    
):
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = load_dataset(dataset_path, dataset, split="train")
    test_dataset  = load_dataset(dataset_path, dataset, split="test")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")    

    train_dataloader = create_dataloader(train_dataset, tokenizer, seq_len, batch_size, device, verbose, seed)
    test_dataloader  = create_dataloader(test_dataset, tokenizer, seq_len, batch_size, device, verbose, seed)

    teacher_config = Config.from_file(teacher_config_path)
    teacher = GPT(teacher_config)
    
    if distill.use_precomputed_logits:
        if not distill.precomputed_logits_path:
            raise ValueError("Provide `precomputed_logits_path` when using precomputed logits.")
        print("Using separate teacher logits loader...")
        teacher_logits_loader = TeacherLogitsLoader(distill.precomputed_logits_path)
    else:
        teacher_logits_loader = None
        teacher = GPT(teacher_config)
        teacher.load_state_dict(torch.load(teacher_ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded teacher model from {teacher_ckpt_path}")

    if student_config_path.exists():
        student_config = Config.from_file(student_config_path)
        student = GPT(student_config)
        if student_ckpt_path.exists():
            student.load_state_dict(torch.load(student_ckpt_path, map_location=device))
            print(f"Loaded student model from {student_ckpt_path}")
        else:
            print("Student checkpoint not found. Initializing student model with random weights.")
    else:
        print("Student configuration not found. Initializing student model with random weights.")
        student = GPT(teacher_config)
        
    print("\nEvaluating student model BEFORE distillation:")
    evaluate(student, test_dataloader, device)

    kd = KD(
        teacher=teacher,
        student=student,
        seq_len=seq_len,
        method=distill.method,
        dataloader=train_dataloader,
        device=device,
        seed=seed,
        verbose=verbose,
        kd_epochs=distill.kd_epochs,
        teacher_logits_loader=teacher_logits_loader,
        distillation_weight=distill.alpha,
        temperature=distill.temperature
    )
    
    student = kd.distill()

    print("\nEvaluating student model AFTER distillation:")
    evaluate(student, test_dataloader, device)

    student_ckpt = os.path.join(save_dir, "distilled_student_model_checkpoint.pth")
    torch.save(student.state_dict(), student_ckpt)
    print(f"Student model checkpoint saved to {student_ckpt}")

if __name__ == '__main__':
    CLI(main)