from .models import Experiment, TrainingMetric, init_db
from datetime import datetime

class DatabaseHandler:
    def __init__(self):
        self.session = init_db()
    
    def create_experiment(self, model_name, peft_method, learning_rate, batch_size, num_epochs):
        experiment = Experiment(
            model_name=model_name,
            peft_method=peft_method,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        self.session.add(experiment)
        self.session.commit()
        return experiment
    
    def add_training_metric(self, experiment_id, epoch, loss):
        metric = TrainingMetric(
            experiment_id=experiment_id,
            epoch=epoch,
            loss=loss
        )
        self.session.add(metric)
        self.session.commit()
        return metric
    
    def get_experiments(self):
        return self.session.query(Experiment).all()
    
    def get_experiment_metrics(self, experiment_id):
        return self.session.query(TrainingMetric).filter_by(experiment_id=experiment_id).all()
