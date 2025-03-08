import gradio as gr
import torch
import os
import requests
from model_handler import ModelHandler
from training_utils import Trainer
from visualization import plot_training_metrics, plot_training_history
from dataset_handler import DatasetHandler
from database.db_handler import DatabaseHandler
from hyperparameter_tuning import HyperparameterTuner
from data_augmentation import DataAugmentor
import pandas as pd
import plotly.graph_objects as go
from framework_converter import FrameworkConverter
from huggingface_hub import HfApi


class FineTuningApp:
    def __init__(self):
        self.db = DatabaseHandler()
        self.model = None
        self.dataset = None
        self.model_handler = None
        self.best_hyperparameters = None
        self.framework_converter = FrameworkConverter()
        self.data_augmentor = None

    def get_available_frameworks(self):
        """Get list of available frameworks"""
        frameworks = ["PyTorch", "scikit-learn"]
        if self.framework_converter.tf_available:
            frameworks.append("TensorFlow")
        return frameworks

    def has_access_to_model(self, model_name, token):
        api = HfApi()
        try:
            api.model_info(model_name, token=token)
            return True
        except Exception as e:
            print(f"Error checking model access: {str(e)}")
            return False

    def load_model(self, model_name):
        try:
            # Check if trying to load a Llama model
            if "meta-llama" in model_name:
                token = os.getenv("HUGGING_FACE_TOKEN")
                if not token:
                    return ("Error: Hugging Face token is required for Llama models. "
                            "Please set your HUGGING_FACE_TOKEN environment variable.")

                # Verify token access
                if not self.has_access_to_model(model_name, token):
                    return ("Error: Your Hugging Face token doesn't have access to Llama models. "
                            "Please request access at https://huggingface.co/meta-llama")

            self.model_handler = ModelHandler(model_name)
            self.model = self.model_handler.load_model()
            model_info = self.model_handler.get_model_info()

            return f"Model {model_name} loaded successfully!"

        except Exception as e:
            return f"Error loading model: {str(e)}"

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
                return "Please upload a dataset file", None, None

            dataset_handler = DatasetHandler()
            df = pd.read_csv(file.name)
            self.dataset = dataset_handler.prepare_dataset(df, self.model_handler.tokenizer)

            # Initialize data augmentor
            self.data_augmentor = DataAugmentor(self.model_handler.tokenizer)

            preview_html = df.head().to_html()
            stats = (
                f"Dataset loaded successfully!\n"
                f"Rows: {len(df):,}\n"
                f"Columns: {', '.join(df.columns)}"
            )

            # Data quality info
            quality_info = self._analyze_data_quality(df)

            return preview_html, stats, quality_info
        except Exception as e:
            return None, f"Error loading dataset: {str(e)}", None

    def _analyze_data_quality(self, df):
        """Analyze data quality metrics"""
        try:
            quality_info = {
                "Missing Values": df.isnull().sum().to_dict(),
                "Data Types": df.dtypes.astype(str).to_dict(),
                "Sample Values": {col: df[col].head().tolist() for col in df.columns}
            }
            return quality_info
        except Exception as e:
            return f"Error analyzing data quality: {str(e)}"

    def apply_data_augmentation(self, method, config_str):
        try:
            if self.dataset is None:
                return "Please load a dataset first", None

            try:
                config = eval(config_str) if config_str else {}
            except:
                config = {}

            augmented_texts = self.data_augmentor.apply_augmentation(
                self.dataset.texts,
                method=method,
                **config
            )

            # Preview augmented data
            preview = pd.DataFrame({
                'Original': self.dataset.texts[:5],
                'Augmented': augmented_texts[:5]
            }).to_html()

            return "Data augmentation applied successfully", preview
        except Exception as e:
            return f"Error during data augmentation: {str(e)}", None

    def optimize_hyperparameters(self, n_trials=20, optimization_metrics=None):
        try:
            if self.model is None or self.dataset is None:
                return "Please load both model and dataset first", None, None

            tuner = HyperparameterTuner(
                model_handler=self.model_handler,
                dataset=self.dataset,
                n_trials=n_trials,
                optimization_metrics=optimization_metrics
            )

            results = tuner.optimize()
            self.best_hyperparameters = results['best_params']

            # Create visualizations
            history_fig = results['optimization_history']
            importance_fig = results['parameter_importance']

            params_text = "\n".join([
                f"{k}: {v}" for k, v in results['best_params'].items()
            ])

            return (
                f"Best hyperparameters found:\n{params_text}",
                history_fig,
                importance_fig
            )
        except Exception as e:
            return f"Error during hyperparameter optimization: {str(e)}", None, None

    def train_model(self, model_name, peft_method, peft_config, learning_rate, 
                   batch_size, num_epochs, use_optimal_params=False):
        try:
            if self.model is None:
                return "Please load a model first", None
            if self.dataset is None:
                return "Please upload a dataset first", None

            # Parse configuration
            if use_optimal_params and self.best_hyperparameters:
                params = self.best_hyperparameters
            else:
                try:
                    peft_params = eval(peft_config) if peft_config else {}
                    params = {
                        'learning_rate': float(learning_rate),
                        'batch_size': int(batch_size),
                        'num_epochs': int(num_epochs),
                        **peft_params
                    }
                except:
                    return "Error parsing configuration", None

            # Create experiment record
            experiment = self.db.create_experiment(
                model_name=model_name,
                peft_method=peft_method,
                **params
            )

            # Initialize training
            trainer = Trainer(
                model=self.model,
                dataset=self.dataset,
                **params
            )

            metrics_history = []
            for epoch, metrics in trainer.train():
                # Save metrics
                self.db.add_training_metric(
                    experiment_id=experiment.id,
                    epoch=epoch,
                    **metrics
                )
                metrics_history.append(metrics)

                progress = (f"Epoch {epoch + 1}/{num_epochs}: "
                          f"Loss = {metrics['loss']:.4f}")

                yield progress, plot_training_history(metrics_history)

            # Final evaluation
            eval_metrics = trainer.evaluate()
            metrics_text = "\n".join([
                f"{k}: {v:.4f}" for k, v in eval_metrics.items()
            ])

            final_text = f"Training completed!\n\nEvaluation Metrics:\n{metrics_text}"
            yield final_text, plot_training_history(metrics_history)

        except Exception as e:
            yield f"Error during training: {str(e)}", None

    def export_model(self, save_path, target_framework=None):
        """Export the fine-tuned model"""
        try:
            if self.model is None:
                return "Please load a model first"

            if target_framework == "TensorFlow" and not self.framework_converter.tf_available:
                return ("TensorFlow is not installed. To use TensorFlow features, "
                       "please install TensorFlow first.")

            if target_framework and target_framework not in self.get_available_frameworks():
                return (f"Framework {target_framework} is not available.\n"
                       f"Available frameworks: {', '.join(self.get_available_frameworks())}")

            success, message = self.model_handler.export_model(save_path, target_framework)
            if success:
                return f"Model exported successfully to {save_path}"
            return f"Error exporting model: {message}"

        except Exception as e:
            return f"Error during export: {str(e)}"

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

    # Define available options
    models = [
        # Llama family models
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        # Other base models
        "facebook/opt-125m",
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

    augmentation_methods = [
        "mlm",
        "backtranslation",
        "synonym_replacement"
    ]

    with gr.Blocks(title="🔬 Model Fine-Tuning Laboratory") as interface:
        gr.Markdown("""
        # 🔬 Model Fine-Tuning Laboratory
        A comprehensive suite for fine-tuning and optimizing machine learning models
        """)

        with gr.Tabs():
            # Model & Dataset Tab
            with gr.TabItem("📚 Model & Dataset"):
                with gr.Row():
                    with gr.Column():
                        model_name = gr.Dropdown(
                            choices=models,
                            label="Select Base Model",
                            info="Choose from available models"
                        )
                        load_model_btn = gr.Button("Load Model")
                        model_info = gr.Textbox(
                            label="Model Information",
                            interactive=False,
                            lines=6
                        )

                with gr.Row():
                    with gr.Column():
                        dataset_file = gr.File(
                            label="Upload Dataset (CSV)",
                            file_types=[".csv"]
                        )
                        dataset_preview = gr.HTML(label="Dataset Preview")
                        dataset_info = gr.Textbox(
                            label="Dataset Information",
                            interactive=False
                        )
                        data_quality = gr.JSON(
                            label="Data Quality Metrics"
                        )

            # Data Augmentation Tab
            with gr.TabItem("🔄 Data Augmentation"):
                with gr.Row():
                    with gr.Column():
                        aug_method = gr.Dropdown(
                            choices=augmentation_methods,
                            label="Augmentation Method"
                        )
                        aug_config = gr.Textbox(
                            label="Configuration",
                            placeholder="{'num_augmentations': 2}"
                        )
                        aug_btn = gr.Button("Apply Augmentation")
                        aug_status = gr.Textbox(label="Status")
                        aug_preview = gr.HTML(label="Augmented Data Preview")

            # Hyperparameter Tuning Tab
            with gr.TabItem("⚙️ Hyperparameter Tuning"):
                with gr.Row():
                    with gr.Column():
                        n_trials = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=5,
                            label="Number of Trials"
                        )
                        optimization_metrics = gr.CheckboxGroup(
                            choices=["loss", "accuracy", "f1"],
                            label="Metrics to Optimize",
                            value=["loss"]
                        )
                        optimize_btn = gr.Button("Start Optimization")
                        optimization_status = gr.Textbox(label="Status")
                        history_plot = gr.Plot(label="Optimization History")
                        importance_plot = gr.Plot(label="Parameter Importance")

            # Training Tab
            with gr.TabItem("🚀 Training"):
                with gr.Row():
                    with gr.Column():
                        peft_method = gr.Dropdown(
                            choices=peft_methods,
                            label="PEFT Method",
                            info="Choose fine-tuning method"
                        )
                        peft_config = gr.Textbox(
                            label="PEFT Configuration",
                            placeholder="{'r': 16, 'lora_alpha': 32}"
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
                        training_status = gr.Textbox(label="Status")
                        training_plot = gr.Plot(label="Training Progress")

            # Export Tab
            with gr.TabItem("💾 Export/Import"):
                with gr.Row():
                    with gr.Column():
                        export_path = gr.Textbox(
                            label="Export Path",
                            placeholder="Path to save model"
                        )
                        export_framework = gr.Dropdown(
                            choices=app.get_available_frameworks(),
                            label="Target Framework"
                        )
                        export_btn = gr.Button("Export Model")
                        export_status = gr.Textbox(label="Status")

        # Event handlers
        load_model_btn.click(
            app.load_model,
            inputs=[model_name],
            outputs=[model_info]
        )

        dataset_file.upload(
            app.load_dataset,
            inputs=[dataset_file],
            outputs=[dataset_preview, dataset_info, data_quality]
        )

        aug_btn.click(
            app.apply_data_augmentation,
            inputs=[aug_method, aug_config],
            outputs=[aug_status, aug_preview]
        )

        optimize_btn.click(
            app.optimize_hyperparameters,
            inputs=[n_trials, optimization_metrics],
            outputs=[optimization_status, history_plot, importance_plot]
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

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=5000,share=True)
