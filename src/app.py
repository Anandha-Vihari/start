import gradio as gr
import torch
from model_handler import ModelHandler
from training_utils import Trainer
from visualization import plot_training_metrics, plot_training_history
from dataset_handler import DatasetHandler
from database.db_handler import DatabaseHandler
from hyperparameter_tuning import HyperparameterTuner, ModelEvaluator
import pandas as pd
import plotly.graph_objects as go
from framework_converter import FrameworkConverter

class FineTuningApp:
    def __init__(self):
        self.db = DatabaseHandler()
        self.model = None
        self.dataset = None
        self.model_handler = None
        self.best_hyperparameters = None
        self.framework_converter = FrameworkConverter()

    def get_available_frameworks(self):
        """Get list of available frameworks"""
        frameworks = ["PyTorch", "scikit-learn"]
        if self.framework_converter.tf_available:
            frameworks.append("TensorFlow")
        return frameworks

    def load_model(self, model_name):
        try:
            self.model_handler = ModelHandler(model_name)
            self.model = self.model_handler.load_model()
            model_info = self.model_handler.get_model_info()

            # Add framework availability info
            available_frameworks = self.get_available_frameworks()
            tensorflow_status = ("TensorFlow is installed and available." 
                               if self.framework_converter.tf_available 
                               else "TensorFlow is not installed. Some features will be limited.")

            info_text = (
                f"Model loaded successfully!\n"
                f"Framework: {model_info['framework']}\n"
                f"Device: {model_info['device']}\n"
                f"Total parameters: {model_info['parameters']:,}\n"
                f"Trainable parameters: {model_info['trainable_parameters']:,}\n\n"
                f"Framework Status:\n"
                f"Available frameworks: {', '.join(available_frameworks)}\n"
                f"{tensorflow_status}"
            )
            return info_text
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def load_dataset(self, file):
        try:
            if file is None:
                return "Please upload a dataset file", None

            dataset_handler = DatasetHandler()
            df = pd.read_csv(file.name)
            self.dataset = dataset_handler.load_dataset(file.name)
            preview_html = df.head().to_html()
            stats = (
                f"Dataset loaded successfully!\n"
                f"Rows: {len(df):,}\n"
                f"Columns: {', '.join(df.columns)}"
            )
            return preview_html, stats
        except Exception as e:
            return None, f"Error loading dataset: {str(e)}"

    def optimize_hyperparameters(self, n_trials=20):
        try:
            if self.model is None or self.dataset is None:
                return "Please load both model and dataset first", None

            tuner = HyperparameterTuner(
                model_handler=self.model_handler,
                dataset=self.dataset,
                n_trials=n_trials
            )

            results = tuner.optimize()
            self.best_hyperparameters = results['best_params']

            # Create visualization of optimization history
            history_df = results['optimization_history']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=history_df['value'],
                mode='lines+markers',
                name='Validation Loss'
            ))
            fig.update_layout(
                title='Hyperparameter Optimization History',
                xaxis_title='Trial',
                yaxis_title='Validation Loss'
            )

            params_text = "\n".join([
                f"{k}: {v}" for k, v in results['best_params'].items()
            ])
            return f"Best hyperparameters found:\n{params_text}", fig
        except Exception as e:
            return f"Error during hyperparameter optimization: {str(e)}", None

    def train_model(self, model_name, peft_method, peft_config, learning_rate, 
                   batch_size, num_epochs, use_optimal_params=False):
        try:
            if self.model is None:
                return "Please load a model first", None
            if self.dataset is None:
                return "Please upload a dataset first", None

            # Parse PEFT configuration
            if use_optimal_params and self.best_hyperparameters:
                peft_params = {
                    'r': self.best_hyperparameters['lora_r'],
                    'lora_alpha': self.best_hyperparameters['lora_alpha'],
                    'lora_dropout': self.best_hyperparameters['lora_dropout']
                }
                learning_rate = self.best_hyperparameters['learning_rate']
                batch_size = self.best_hyperparameters['batch_size']
            else:
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
            progress = ""
            for epoch, metrics in trainer.train():
                # Save metrics to database
                self.db.add_training_metric(
                    experiment_id=experiment.id,
                    epoch=epoch,
                    loss=metrics['loss']
                )
                metrics_history.append(metrics)
                progress = f"Epoch {epoch}/{num_epochs}: Loss = {metrics['loss']:.4f}"
                yield progress, plot_training_history(metrics_history)

            # Evaluate model
            evaluator = ModelEvaluator(self.model, self.dataset)
            eval_metrics = evaluator.evaluate()

            # Create final plots
            history_fig = plot_training_history(metrics_history)

            # Add evaluation metrics to the response
            metrics_text = "\n".join([
                f"{k}: {v:.4f}" for k, v in eval_metrics.items()
            ])

            final_text = f"Training completed!\n\nEvaluation Metrics:\n{metrics_text}"
            yield final_text, history_fig

        except Exception as e:
            yield f"Error during training: {str(e)}", None

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

    def export_model(self, save_path, target_framework=None):
        """Export the fine-tuned model"""
        try:
            if self.model is None:
                return "Please load a model first"

            # Check if trying to use TensorFlow when not available
            if target_framework == "TensorFlow" and not self.framework_converter.tf_available:
                return ("TensorFlow is not installed. To use TensorFlow features, "
                       "please install TensorFlow first.")

            # Validate target framework
            if target_framework and target_framework not in self.get_available_frameworks():
                return (f"Framework {target_framework} is not available.\n"
                       f"Available frameworks: {', '.join(self.get_available_frameworks())}")

            success, message = self.model_handler.export_model(save_path, target_framework)
            if success:
                return f"Model exported successfully to {save_path}"
            return f"Error exporting model: {message}"

        except Exception as e:
            return f"Error during export: {str(e)}"

    def import_model(self, load_path, target_framework=None):
        """Import a previously exported model"""
        try:
            if target_framework == "TensorFlow" and not self.framework_converter.tf_available:
                return ("TensorFlow is not installed. To use TensorFlow features, "
                       "please install TensorFlow first.")

            if self.model_handler is None:
                self.model_handler = ModelHandler("")

            success, message = self.model_handler.import_model(load_path, target_framework)
            if success:
                self.model = self.model_handler.model
                model_info = self.model_handler.get_model_info()

                tensorflow_status = ("TensorFlow is available" 
                                   if self.framework_converter.tf_available 
                                   else "TensorFlow is not installed")

                return (
                    f"Model imported successfully!\n"
                    f"Framework: {model_info['framework']}\n"
                    f"Device: {model_info['device']}\n"
                    f"Parameters: {model_info['parameters']:,}\n\n"
                    f"Framework Status: {tensorflow_status}"
                )
            return f"Error importing model: {message}"

        except Exception as e:
            return f"Error during import: {str(e)}"


def create_interface():
    app = FineTuningApp()

    # Define model options
    llama_models = [
        "facebook/opt-125m",  # Smaller model for testing
        "EleutherAI/gpt-neo-125M",
        "google/flan-t5-small"
    ]

    peft_methods = [
        "LoRA",
        "AdaLoRA",
        "Prefix Tuning",
        "P-Tuning",
        "IA3"
    ]

    # Get supported frameworks
    supported_frameworks = app.get_available_frameworks()

    with gr.Blocks(title="🔬 LLM Fine-tuning Laboratory") as interface:
        gr.Markdown("""
        # 🔬 LLM Fine-tuning Laboratory
        ## Available Frameworks
        This instance supports: """ + ", ".join(supported_frameworks))

        with gr.Tab("Model & Dataset"):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(
                        choices=llama_models,
                        label="Select Base Model",
                        value=llama_models[0],
                        info="Choose from available models"
                    )
                    load_model_btn = gr.Button("Load Model")
                    model_info = gr.Textbox(
                        label="Model Status",
                        interactive=False,
                        lines=6
                    )

            with gr.Row():
                with gr.Column():
                    dataset_file = gr.File(
                        label="Upload Training Data (CSV)",
                        file_types=[".csv"]
                    )
                    dataset_preview = gr.HTML(label="Dataset Preview")
                    dataset_info = gr.Textbox(
                        label="Dataset Info",
                        interactive=False
                    )

        with gr.Tab("Hyperparameter Tuning"):
            with gr.Row():
                with gr.Column():
                    n_trials = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Number of Optimization Trials"
                    )
                    optimize_btn = gr.Button("Start Optimization")
                    optimization_status = gr.Textbox(
                        label="Optimization Status",
                        interactive=False
                    )
                    optimization_plot = gr.Plot(label="Optimization Progress")

        with gr.Tab("Training"):
            with gr.Row():
                with gr.Column():
                    peft_method = gr.Dropdown(
                        choices=peft_methods,
                        label="PEFT Method",
                        value=peft_methods[0],
                        info="Choose fine-tuning method"
                    )
                    peft_config = gr.Textbox(
                        label="PEFT Configuration",
                        placeholder="{'r': 16, 'lora_alpha': 32}",
                        info="Optional: Enter as Python dict"
                    )
                    use_optimal = gr.Checkbox(
                        label="Use Optimal Parameters",
                        info="Use results from optimization"
                    )
                    learning_rate = gr.Slider(
                        minimum=1e-6,
                        maximum=1e-3,
                        value=1e-4,
                        label="Learning Rate"
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
                    train_btn = gr.Button("Start Training", variant="primary")
                    training_status = gr.Textbox(
                        label="Training Status",
                        interactive=False
                    )
                    training_plot = gr.Plot(label="Training Progress")

        with gr.Tab("Export/Import"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Export Model")
                    export_path = gr.Textbox(
                        label="Export Path",
                        placeholder="Path to save model",
                        info="Enter save directory path"
                    )
                    export_framework = gr.Dropdown(
                        choices=supported_frameworks,
                        label="Target Framework",
                        value=supported_frameworks[0] if supported_frameworks else None,
                        info="Select export framework"
                    )
                    export_btn = gr.Button("Export Model")
                    export_status = gr.Textbox(
                        label="Export Status",
                        interactive=False
                    )

                with gr.Column():
                    gr.Markdown("### Import Model")
                    import_path = gr.Textbox(
                        label="Import Path",
                        placeholder="Path to load model",
                        info="Enter model directory path"
                    )
                    import_framework = gr.Dropdown(
                        choices=supported_frameworks,
                        label="Target Framework",
                        value=supported_frameworks[0] if supported_frameworks else None,
                        info="Select import framework"
                    )
                    import_btn = gr.Button("Import Model")
                    import_status = gr.Textbox(
                        label="Import Status",
                        interactive=False
                    )

        # Event handlers
        load_model_btn.click(
            app.load_model,
            inputs=[model_name],
            outputs=[model_info]
        )

        dataset_file.upload(
            app.load_dataset,
            inputs=[dataset_file],
            outputs=[dataset_preview, dataset_info]
        )

        optimize_btn.click(
            app.optimize_hyperparameters,
            inputs=[n_trials],
            outputs=[optimization_status, optimization_plot]
        )

        train_btn.click(
            app.train_model,
            inputs=[
                model_name,
                peft_method,
                peft_config,
                learning_rate,
                batch_size,
                num_epochs,
                use_optimal
            ],
            outputs=[training_status, training_plot]
        )

        export_btn.click(
            app.export_model,
            inputs=[export_path, export_framework],
            outputs=[export_status]
        )

        import_btn.click(
            app.import_model,
            inputs=[import_path, import_framework],
            outputs=[import_status]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=5000)