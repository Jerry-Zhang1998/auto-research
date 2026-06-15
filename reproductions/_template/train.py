import torch
import torch.nn as nn
import os, json, time, random
import numpy as np
from config import Config
from model import ExampleModel
from loss import PaperLoss
from dataset import get_dataloader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: Config) -> None:
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    os.makedirs(config.train.output_dir, exist_ok=True)

    model = ExampleModel(config.model).to(device)
    criterion = PaperLoss(config.train).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    train_loader = get_dataloader(config.train, "train")
    val_loader = get_dataloader(config.train, "val")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.train.max_epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["total"]
            loss.backward()

            if config.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

            optimizer.step()
            global_step += 1

            if global_step % config.train.log_every == 0:
                log = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
                print(f"Epoch {epoch} Step {global_step} | {log}")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss_dict = criterion(outputs, targets)
                val_losses.append(loss_dict["total"].item())

        val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch} | Val loss: {val_loss:.4f}")

        # Checkpoint
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(ckpt, os.path.join(config.train.output_dir, "latest.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(config.train.output_dir, "best.pt"))
            print(f"  Saved best checkpoint (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = Config()
    if args.lr:
        config.train.lr = args.lr
    if args.epochs:
        config.train.max_epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size

    train(config)
