import gradio as gr
import torch
from model_handler import ModelHandler
from training_utils import Trainer
from visualization import plot_training_metrics, plot_training_history
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
            model_info = self.model_handler.get_model_info()
            info_text = (
                f"Model loaded successfully!\n"
                f"Framework: {model_info['framework']}\n"
                f"Device: {model_info['device']}\n"
                f"Total parameters: {model_info['parameters']:,}\n"
                f"Trainable parameters: {model_info['trainable_parameters']:,}"
            )
            return info_text
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
            stats = (
                f"Dataset loaded successfully!\n"
                f"Rows: {len(df):,}\n"
                f"Columns: {', '.join(df.columns)}"
            )
            return preview, stats
        except Exception as e:
            return None, f"Error loading dataset: {str(e)}"

    def train_model(self, model_name, peft_method, peft_config, learning_rate, 
                   batch_size, num_epochs, progress=gr.Progress()):
        try:
            if self.model is None:
                return "Please load a model first", None
            if self.dataset is None:
                return "Please upload a dataset first", None

            # Parse PEFT configuration
            try:
                peft_params = eval(peft_config) if peft_config else {}
            except:
                peft_params = {}

            # Create experiment record
            experiment = self.db.create_experiment(
                model_name=model_name,
                peft_method=peft_method,
                learning_rate=float(learning_rate),
                batch_size=int(batch_size),
                num_epochs=int(num_epochs)
            )

            # Apply PEFT method with custom configuration
            self.model = self.model_handler.apply_peft(peft_method, **peft_params)

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

            # Create final plots
            history_fig = plot_training_history(metrics_history)
            return "Training completed successfully!", history_fig

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
                fig = plot_training_history([{
                    "epoch": m.epoch,
                    "loss": m.loss
                } for m in metrics])
                html += f"<div style='height: 300px;'>{fig.to_html()}</div>"
            html += "</div>"
        html += "</div>"
        return html

def create_interface():
    app = FineTuningApp()

    # Define model options with categories
    llama_models = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf"
    ]

    other_models = [
        "facebook/opt-125m",
        "EleutherAI/gpt-neo-125M",
        "google/flan-t5-small"
    ]

    all_models = llama_models + other_models

    peft_methods = [
        "LoRA",
        "AdaLoRA",
        "Prefix Tuning",
        "P-Tuning",
        "IA3"
    ]

    with gr.Blocks(title="LLM Fine-tuning Laboratory") as interface:
        gr.Markdown("# 🔬 LLM Fine-tuning Laboratory")

        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    choices=all_models,
                    label="Select Base Model",
                    value="meta-llama/Llama-2-7b-hf",
                    info="Choose from Llama 2 models or other available models"
                )
                load_model_btn = gr.Button("Load Model")
                model_status = gr.Textbox(label="Model Status", interactive=False)

                dataset_upload = gr.File(label="Upload Training Data (CSV)")
                dataset_preview = gr.HTML(label="Dataset Preview")
                dataset_status = gr.Textbox(label="Dataset Status", interactive=False)

            with gr.Column():
                peft_method = gr.Dropdown(
                    choices=peft_methods,
                    label="PEFT Method",
                    value="LoRA",
                    info="Choose Parameter-Efficient Fine-Tuning method"
                )

                peft_config = gr.Textbox(
                    label="PEFT Configuration (Optional)",
                    info="Enter configuration as Python dict, e.g., {'r': 8, 'lora_alpha': 16}",
                    placeholder="{'r': 16, 'lora_alpha': 32}"
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

                train_btn = gr.Button("Start Fine-tuning", variant="primary")
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
            inputs=[
                model_name,
                peft_method,
                peft_config,
                learning_rate,
                batch_size,
                num_epochs
            ],
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