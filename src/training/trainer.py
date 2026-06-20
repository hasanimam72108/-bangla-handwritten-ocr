import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.training.metrics import evaluate_model


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        config: dict,
        device: torch.device,
        output_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        total_steps = len(train_loader) * config["training"]["epochs"]
        warmup_steps = config["training"]["warmup_steps"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )

        self.scaler = GradScaler("cuda", enabled=config["training"]["mixed_precision"])
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config["model"]["label_smoothing"],
            ignore_index=tokenizer.pad_token_id,
        )
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

        self.best_cer = float("inf")
        self.patience_counter = 0
        self.patience = config["training"]["early_stop_patience"]
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in progress:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["gradient_clip"],
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

            if self.global_step % self.config["training"]["log_every"] == 0:
                self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("LR", lr, self.global_step)

            self.global_step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def validate(self, epoch: int) -> dict:
        metrics = evaluate_model(
            self.model, self.val_loader, self.tokenizer, self.device
        )
        self.writer.add_scalar("CER/val", metrics["cer"], epoch)
        self.writer.add_scalar("WER/val", metrics["wer"], epoch)
        self.writer.add_scalar("Accuracy/val", metrics["exact_match_accuracy"], epoch)
        return metrics

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )

    def run(self):
        for epoch in range(1, self.config["training"]["epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"CER={val_metrics['cer']:.4f}, WER={val_metrics['wer']:.4f}, "
                f"Acc={val_metrics['exact_match_accuracy']:.4f}"
            )

            if epoch % self.config["training"]["save_every"] == 0:
                ckpt_path = os.path.join(self.output_dir, "last_model.pt")
                self.save_checkpoint(ckpt_path, epoch, val_metrics)

            if val_metrics["cer"] < self.best_cer:
                self.best_cer = val_metrics["cer"]
                self.patience_counter = 0
                best_path = os.path.join(self.output_dir, "best_model.pt")
                self.save_checkpoint(best_path, epoch, val_metrics)
                print(f"  → New best model saved (CER={val_metrics['cer']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        self.writer.close()
        return self.best_cer
