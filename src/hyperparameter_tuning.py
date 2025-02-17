import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go

class HyperparameterTuner:
    def __init__(self, model_handler, dataset, n_trials=20, optimization_metrics=None):
        self.model_handler = model_handler
        self.dataset = dataset
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        self.optimization_metrics = optimization_metrics or ['loss']
        self.study = None

    def get_parameter_space(self) -> Dict[str, Any]:
        """Define the hyperparameter search space"""
        return {
            'learning_rate': {
                'type': 'float',
                'range': [1e-5, 1e-3],
                'log': True
            },
            'batch_size': {
                'type': 'int',
                'range': [4, 32]
            },
            'warmup_steps': {
                'type': 'int',
                'range': [50, 500]
            },
            'weight_decay': {
                'type': 'float',
                'range': [0.01, 0.1]
            },
            'dropout': {
                'type': 'float',
                'range': [0.1, 0.5]
            }
        }

    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        # Get hyperparameters for this trial
        params = self._suggest_parameters(trial)

        # Create validation split
        train_size = int(0.8 * len(self.dataset))
        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, len(self.dataset) - train_size]
        )

        # Configure model with trial parameters
        model = self.model_handler.configure_model(params)

        # Training configuration
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size']
        )

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
        metrics = self._evaluate_model(model, val_loader)

        # Report intermediate values for visualization
        for metric_name, value in metrics.items():
            trial.report(value, step=0)

        return metrics['loss']  # Primary optimization metric

    def _suggest_parameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial"""
        params = {}
        param_space = self.get_parameter_space()

        for param_name, config in param_space.items():
            if config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    *config['range'],
                    log=config.get('log', False)
                )
            elif config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    *config['range']
                )

        return params

    def _evaluate_model(self, model, dataloader) -> Dict[str, float]:
        """Evaluate model performance with various metrics"""
        metrics = {}
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                outputs = model(
                    input_ids=batch['input_ids'].to(model.device),
                    labels=batch['labels'].to(model.device)
                )

                loss = outputs.loss
                logits = outputs.logits

                predictions = torch.argmax(logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item()

        # Calculate metrics
        metrics['loss'] = total_loss / len(dataloader)
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['precision'] = precision_score(
            all_labels, all_preds, average='weighted'
        )
        metrics['recall'] = recall_score(
            all_labels, all_preds, average='weighted'
        )
        metrics['f1'] = f1_score(
            all_labels, all_preds, average='weighted'
        )

        return metrics

    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler()
        )

        self.study.optimize(self.objective, n_trials=self.n_trials)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.get_optimization_history(),
            'parameter_importance': self.get_parameter_importance()
        }

    def get_optimization_history(self) -> go.Figure:
        """Generate optimization history visualization"""
        if not self.study:
            raise ValueError("No optimization history available. Run optimize() first.")

        fig = go.Figure()

        # Add trace for each metric
        for metric in self.optimization_metrics:
            values = [
                t.value for t in self.study.trials if t.value is not None
            ]

            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines+markers',
                name=metric.capitalize()
            ))

        fig.update_layout(
            title='Hyperparameter Optimization History',
            xaxis_title='Trial',
            yaxis_title='Metric Value',
            showlegend=True
        )

        return fig

    def get_parameter_importance(self) -> go.Figure:
        """Generate parameter importance visualization"""
        if not self.study:
            raise ValueError("No optimization history available. Run optimize() first.")

        importance = optuna.importance.get_param_importances(self.study)

        fig = go.Figure([go.Bar(
            x=list(importance.keys()),
            y=list(importance.values()),
        )])

        fig.update_layout(
            title='Hyperparameter Importance',
            xaxis_title='Parameter',
            yaxis_title='Importance Score',
            showlegend=False
        )

        return fig