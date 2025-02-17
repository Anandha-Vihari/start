import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        self.model.to(self.device)
    
    def train(self):
        """Training loop with generator for progress updates"""
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(dataloader):
                self.optimizer.zero_grad()
                
                inputs = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
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
            
            yield epoch, metrics
