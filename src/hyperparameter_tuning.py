import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

class HyperparameterTuner:
    def __init__(self, model_handler, dataset, n_trials=20):
        self.model_handler = model_handler
        self.dataset = dataset
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        # Hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_int('batch_size', 4, 32),
            'lora_r': trial.suggest_int('lora_r', 4, 32),
            'lora_alpha': trial.suggest_int('lora_alpha', 8, 64),
            'lora_dropout': trial.suggest_float('lora_dropout', 0.0, 0.5)
        }
        
        # Create validation split
        train_size = int(0.8 * len(self.dataset))
        train_dataset, val_dataset = random_split(
            self.dataset, 
            [train_size, len(self.dataset) - train_size]
        )
        
        # Configure model with trial parameters
        model = self.model_handler.apply_peft(
            method="LoRA",
            r=params['lora_r'],
            lora_alpha=params['lora_alpha'],
            lora_dropout=params['lora_dropout']
        )
        
        # Training configuration
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Quick training loop for parameter evaluation
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(model.device),
                labels=batch['labels'].to(model.device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'].to(model.device),
                    labels=batch['labels'].to(model.device)
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss
    
    def optimize(self):
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': study.trials_dataframe()
        }

class ModelEvaluator:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.metrics = {}
    
    def evaluate(self, batch_size=8):
        """Evaluate model performance with various metrics"""
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=batch_size)
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.model.device),
                    labels=batch['labels'].to(self.model.device)
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                predictions = torch.argmax(logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        self.metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        return self.metrics
    
    def get_confusion_matrix(self):
        """Generate confusion matrix for visualization"""
        if not self.metrics:
            self.evaluate()
        
        from sklearn.metrics import confusion_matrix
        import plotly.figure_factory as ff
        
        cm = confusion_matrix(all_labels, all_preds)
        
        # Create heatmap
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=list(range(len(cm))),
            y=list(range(len(cm))),
            colorscale='Viridis'
        )
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        return fig
