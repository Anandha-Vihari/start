import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, dataset, learning_rate, batch_size, num_epochs, peft_method):
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.peft_method = peft_method

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = GradScaler() if self.device == "cuda" else None

        if self.device == "cuda":
            # Enable memory efficient attention if available
            if hasattr(self.model.config, 'use_memory_efficient_attention'):
                self.model.config.use_memory_efficient_attention = True
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

    def train(self):
        """Training loop with generator for progress updates"""
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=self.device=="cuda"
        )

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                self.optimizer.zero_grad()

                inputs = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Use automatic mixed precision for faster training on GPU
                if self.device == "cuda":
                    with autocast():
                        outputs = self.model(input_ids=inputs, labels=labels)
                        loss = outputs.loss

                    # Scale loss and backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(input_ids=inputs, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            metrics = {
                "loss": avg_loss,
                "epoch": epoch + 1,
                "learning_rate": self.learning_rate
            }

            # Add GPU memory info if available
            if self.device == "cuda":
                metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024**3)  # GB

            yield epoch, metrics

    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.device=="cuda"
        )

        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=inputs, labels=labels)
                total_loss += outputs.loss.item()

        return {
            "loss": total_loss / len(dataloader),
            "device": self.device
        }