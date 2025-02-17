import gradio as gr
import torch
from model_handler import ModelHandler
from training_utils import Trainer
from visualization import plot_training_metrics
from dataset_handler import DatasetHandler
from database.db_handler import DatabaseHandler
import pandas as pd
import plotly.graph_objects as go

class FineTuningApp:
    def __init__(self):
        self.db = DatabaseHandler()
        self.model = None
        self.dataset = None
        self.model_handler = None

    def load_model(self, model_name):
        try:
            self.model_handler = ModelHandler(model_name)
            self.model = self.model_handler.load_model()
            return f"Model {model_name} loaded successfully!"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def load_dataset(self, file):
        try:
            if file is None:
                return None, "Please upload a dataset file"

            dataset_handler = DatasetHandler()
            df = dataset_handler.load_dataset(file.name)
            self.dataset = df
            preview = df.head().to_html()
            return preview, "Dataset loaded successfully!"
        except Exception as e:
            return None, f"Error loading dataset: {str(e)}"

    def train_model(self, model_name, peft_method, learning_rate, batch_size, num_epochs, progress=gr.Progress()):
        try:
            if self.model is None:
                return "Please load a model first", None
            if self.dataset is None:
                return "Please upload a dataset first", None

            # Create experiment record
            experiment = self.db.create_experiment(
                model_name=model_name,
                peft_method=peft_method,
                learning_rate=float(learning_rate),
                batch_size=int(batch_size),
                num_epochs=int(num_epochs)
            )

            trainer = Trainer(
                model=self.model,
                dataset=self.dataset,
                learning_rate=float(learning_rate),
                batch_size=int(batch_size),
                num_epochs=int(num_epochs),
                peft_method=peft_method
            )

            # Training loop with progress updates
            metrics_history = []
            for epoch, metrics in progress.tqdm(trainer.train(), desc="Training"):
                # Save metrics to database
                self.db.add_training_metric(
                    experiment_id=experiment.id,
                    epoch=epoch,
                    loss=metrics['loss']
                )
                metrics_history.append(metrics)

            # Create final plot
            fig = plot_training_metrics(metrics_history[-1])
            return "Training completed successfully!", fig

        except Exception as e:
            return f"Error during training: {str(e)}", None

    def get_previous_experiments(self):
        experiments = self.db.get_experiments()
        if not experiments:
            return "No previous experiments found"

        html = "<div style='max-height: 400px; overflow-y: auto;'>"
        for exp in experiments:
            html += f"<div style='margin: 10px; padding: 10px; border: 1px solid #ddd;'>"
            html += f"<h4>Experiment {exp.id} - {exp.model_name}</h4>"
            html += f"<p>PEFT Method: {exp.peft_method}</p>"
            html += f"<p>Learning Rate: {exp.learning_rate}</p>"
            html += f"<p>Batch Size: {exp.batch_size}</p>"
            html += f"<p>Number of Epochs: {exp.num_epochs}</p>"
            html += f"<p>Created At: {exp.created_at}</p>"

            metrics = self.db.get_experiment_metrics(exp.id)
            if metrics:
                metrics_data = {
                    "loss": metrics[-1].loss,
                    "epoch": metrics[-1].epoch,
                    "learning_rate": exp.learning_rate
                }
                fig = plot_training_metrics(metrics_data)
                html += f"<img src='{fig.to_image(format='png')}' />"
            html += "</div>"
        html += "</div>"
        return html

def create_interface():
    app = FineTuningApp()

    with gr.Blocks(title="LLM Fine-tuning Laboratory") as interface:
        gr.Markdown("# 🔬 LLM Fine-tuning Laboratory")

        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    choices=["facebook/opt-125m", "EleutherAI/gpt-neo-125M", "google/flan-t5-small"],
                    label="Select Base Model",
                    value="facebook/opt-125m"
                )
                load_model_btn = gr.Button("Load Model")
                model_status = gr.Textbox(label="Model Status", interactive=False)

                dataset_upload = gr.File(label="Upload Training Data (CSV)")
                dataset_preview = gr.HTML(label="Dataset Preview")
                dataset_status = gr.Textbox(label="Dataset Status", interactive=False)

            with gr.Column():
                peft_method = gr.Dropdown(
                    choices=["LoRA", "Prefix Tuning", "P-Tuning"],
                    label="PEFT Method",
                    value="LoRA"
                )
                learning_rate = gr.Slider(
                    minimum=1e-6,
                    maximum=1e-3,
                    value=1e-4,
                    label="Learning Rate",
                    info="Select learning rate between 1e-6 and 1e-3"
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=8,
                    step=1,
                    label="Batch Size"
                )
                num_epochs = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=3,
                    step=1,
                    label="Number of Epochs"
                )
                train_btn = gr.Button("Start Fine-tuning")
                training_status = gr.Textbox(label="Training Status", interactive=False)
                training_plot = gr.Plot(label="Training Progress")

        with gr.Tab("Previous Experiments"):
            refresh_btn = gr.Button("Refresh Experiments")
            experiments_display = gr.HTML(label="Previous Experiments")

        # Event handlers
        load_model_btn.click(
            app.load_model,
            inputs=[model_name],
            outputs=[model_status]
        )

        dataset_upload.change(
            app.load_dataset,
            inputs=[dataset_upload],
            outputs=[dataset_preview, dataset_status]
        )

        train_btn.click(
            app.train_model,
            inputs=[model_name, peft_method, learning_rate, batch_size, num_epochs],
            outputs=[training_status, training_plot]
        )

        refresh_btn.click(
            app.get_previous_experiments,
            outputs=[experiments_display]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=5000)