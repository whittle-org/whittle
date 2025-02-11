import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

from litgpt import Config
from whittle.models.gpt.model import GPT
from whittle.args import DistillArgs
from whittle.distillation.utils import (
    create_dataloader,
    load_teacher_predictions,
    create_tiny_gpt,
    compute_parameters,
    merge_saved_logits,
    TeacherLogitsLoader
)
from whittle.distillation.knowledge_distill import KD
from jsonargparse import CLI

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
    dataset: str = 'wikitext-2-raw-v1',
    dataset_path: str = 'wikitext',
    teacher_ckpt: str = './checkpoints/teacher_checkpoint.pth',
    student_ckpt: str = '',
    student_config: str = '',

    distill: DistillArgs = DistillArgs(
        method='logits',
        on_cluster=False,
        smoke_test=True,
        use_topk_logits=False,
        use_precomputed_logits=False
    )    
):
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = load_dataset(dataset_path, dataset, split="train")
    test_dataset  = load_dataset(dataset_path, dataset, split="test")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    train_dataloader = create_dataloader(train_dataset, tokenizer, seq_len, batch_size, device, verbose, seed)
    test_dataloader  = create_dataloader(test_dataset, tokenizer, seq_len, batch_size, device, verbose, seed)

    if distill.use_precomputed_logits:
        if not distill.precomputed_logits_path:
            raise ValueError("Provide `precomputed_logits_path` when using precomputed logits.")
        print("Using separate teacher logits loader...")
        teacher_logits_loader = TeacherLogitsLoader(distill.precomputed_logits_path)
    else:
        teacher_logits_loader = None
        teacher = create_tiny_gpt()
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
        print(f"Loaded teacher model from {teacher_ckpt}")

    if student_config:
        print("Loading student model from provided configuration.")
        student_config = Config.from_file(distill.student_config)
        student = GPT(student_config)
        # Optionally load weights from the teacher (non-strict, in case shapes differ).
        teacher = create_tiny_gpt()
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
        print(f"Loaded teacher model from {teacher_ckpt}")

        # prev_params = compute_parameters(teacher)

        student.load_state_dict(teacher.state_dict(), strict=False)
        print("Loaded student model from provided configuration.\n")
    
    elif student_ckpt:
        print(f"Loading student model from checkpoint: {student_ckpt}")
        config_path = os.path.join(os.path.dirname(student_ckpt), "model_config.yaml")
        config = Config.from_file(config_path)
        student = GPT(config)
        student.load_state_dict(torch.load(student_ckpt, map_location=device))
        print(f"Loaded student model from checkpoint: {student_ckpt}")
    
    else:
        print("\nNo student configuration or checkpoint provided. Using teacher's configuration for the student model.\n")

        teacher = create_tiny_gpt()
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
        print(f"Loaded teacher model from {teacher_ckpt}")

        student = GPT(teacher.config)
        print("Student model configuration:", student.config)
        student.load_state_dict(teacher.state_dict())
        small_network = {
            "sub_network_n_embd": teacher.config.n_embd // 2,
            "sub_network_intermediate_size": teacher.config.intermediate_size // 2,
            "sub_network_num_heads": teacher.config.n_head // 2,
            "sub_network_n_layers": teacher.config.n_layer // 2,
            "sub_network_head_size": teacher.config.head_size
        }
        print("Sub-network configuration:", small_network)
        print(f"\nNumber of parameters before pruning the student model: {compute_parameters(student)}")
        student.set_sub_network(**small_network)
        print(f"Number of parameters after pruning the student model: {compute_parameters(student)}")

    # assert compute_parameters(teacher) == prev_params

    # print("\nTeacher Model Evaluation:")
    # evaluate(teacher, test_dataloader, device)

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
        teacher_logits_loader=teacher_logits_loader
    )
    
    student = kd.distill()

    print("\nEvaluating student model AFTER distillation:")
    evaluate(student, test_dataloader, device)

    student_ckpt = os.path.join(save_dir, "distilled_student_model_checkpoint.pth")
    torch.save(student.state_dict(), student_ckpt)
    print(f"Student model checkpoint saved to {student_ckpt}")

if __name__ == '__main__':
    CLI(main)