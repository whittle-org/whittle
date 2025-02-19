import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from jsonargparse import CLI
from tqdm import tqdm

from whittle.args import DistillArgs
from whittle.models.gpt.model import GPT
from litgpt import Config
from litgpt.scripts.download import download_from_hub
from whittle.distillation.utils import (
    create_dataloader,
    create_student_training_data,
    create_tiny_gpt,
    save_config_to_file
)

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    epochs: int,
) -> torch.nn.Module:
    """
    Train the model using cross-entropy loss and Adam optimizer.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = torch.nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss average: {avg_loss:.4f}")

    return model

def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str
) -> float:
    """
    Evaluate the model on the provided dataset.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            logits = model(input_ids)
            loss = torch.nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) 
    print(f"Average loss: {avg_loss:.4f}")
    return avg_loss

def load_teacher( 
    teacher_source: str, 
    teacher_ckpt: str, 
    model_id: str, 
    checkpoint_dir: str, 
    access_token,
    device: str,
    teacher_config_path: Path 
):
    """
    Load the teacher model based on teacher_source option:
    - "hub": Download from hub.
    - "checkpoint": Load from a checkpoint file if available, otherwise fallback to tiny model.
    - "tiny": Create a tiny GPT teacher model.
    """
    if teacher_source == "checkpoint":
        if os.path.exists(teacher_ckpt):
            print("\nLoading teacher model from checkpoint.")
            if model_id != '':
                model_dir = os.path.join(checkpoint_dir, model_id)
                config_path = os.path.join(model_dir, "model_config.yaml")
                config = Config.from_file(config_path)
                config.fix_head_size = True
                config.model_type = "gpt"
                teacher = GPT(config)
                teacher_config = config
            else:
                tiny_config = Config.from_file(teacher_config_path)
                teacher = GPT(tiny_config)
                teacher_config = tiny_config
                print("Loaded teacher configuration from the provided checkpoint.") 
            
            teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
            teacher.reset_super_network()
            smoke_test = False
    
    elif teacher_source == "hub":
        if model_id == '':
            raise ValueError("Model ID must be provided when teacher_source is hub.")
        print(f"\nDownloading teacher model from hub. Number of CUDA devices: {torch.cuda.device_count()}")
        download_from_hub(
            repo_id=model_id,
            checkpoint_dir=Path(checkpoint_dir),
            access_token=access_token,
        )
        model_dir = os.path.join(checkpoint_dir, model_id)
        config_path = os.path.join(model_dir, "model_config.yaml")
        config_path_hf = os.path.join(model_dir, "config.json")
        config = Config.from_file(config_path)
        config.fix_head_size = True
        config.model_type = "gpt"
        with open(config_path_hf) as f:
            hf_config = json.load(f)
        config.tie_embeddings = hf_config["tie_word_embeddings"]

        teacher = GPT(config)
        teacher.reset_super_network()
        smoke_test = False
        teacher_config = config

    elif teacher_source == "tiny":
        teacher, teacher_config = create_tiny_gpt()
        smoke_test=True
    
    else:
        raise ValueError("Invalid teacher_source option. Choose from 'hub', 'checkpoint', or 'tiny'.")

    return teacher, smoke_test, teacher_config

def main(
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    seq_len: int = 64,
    batch_size: int = 5,
    verbose: bool = False,
    checkpoint_dir: Path = Path('checkpoints'),
    save_dir: Path = Path('save_dir'),
    dataset: str = 'wikitext-2-raw-v1',
    dataset_path: str = 'wikitext',
    access_token: str = '',
    epochs: int = 3,
    teacher_ckpt: Path = Path('checkpoints/teacher_checkpoint.pth'),
    model_id: str = '',
    teacher_source: str = 'checkpoint',
    teacher_config_path: Path = Path('checkpoints/teacher_config.yaml'),

    distill: DistillArgs = DistillArgs(
        method='logits',
        on_cluster=False,
        smoke_test=True,
        top_k = 100,
    )
):
    """
    Generate training data for the student model by passing input data through 
    the teacher model, extracting top-k logits per token, and saving the results 
    in a torch file. If smoke_test is enabled, the tiny GPT teacher model is 
    additionally trained using the train() function.
    The teacher model checkpoint is saved in the 'checkpoints' folder,
    and the student training data is stored in the './save_dir/' folder.
    """
    if verbose:
        print(f"Generating student training data using model with configuration:\n\n{distill}\n\n")
    
    teacher, distill.smoke_test, teacher_config = load_teacher(teacher_source, teacher_ckpt, model_id, checkpoint_dir, access_token, device, teacher_config_path)
    teacher.to(device)
    
    split = "train"
    ds = load_dataset(dataset_path, dataset, split=split)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataloader = create_dataloader(ds, tokenizer, seq_len, batch_size, device, verbose, seed)

    if distill.smoke_test:
        print(f"\nSmoke test enabled: training the tinyGPT teacher for {epochs} epochs...")
        teacher = train(teacher, dataloader, device, epochs)
        print("\nFinished training the teacher model.")

    teacher.eval()    
    print("\nEvaluating the teacher model on the test dataset...")
    test_ds = load_dataset(dataset_path, dataset, split="test")
    test_dataloader = create_dataloader(test_ds, tokenizer, seq_len, batch_size, device, verbose, seed)
    eval_loss = evaluate(teacher, test_dataloader, device)
    print(f"\nTeacher model evaluation loss: {eval_loss:.4f}")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    teacher_checkpoint_path = os.path.join(checkpoint_dir, "teacher_checkpoint.pth")
    torch.save(teacher.state_dict(), teacher_checkpoint_path)
    teacher_config_path = os.path.join(checkpoint_dir, "teacher_config.yaml")
    save_config_to_file(teacher_config, teacher_config_path)

    print(f"\nTeacher model checkpoint saved to {teacher_checkpoint_path} and configuration saved to {teacher_config_path}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"\nCreated directory {save_dir} for student training data.")
        
    training_data_output = os.path.join(save_dir, os.path.basename(distill.precomputed_logits_path))
    create_student_training_data(teacher, dataloader, device, training_data_output, distill.top_k, distill.subset_size)
    
    print(f"\nStudent training data saved to {training_data_output}")

if __name__ == "__main__":
    CLI(main)