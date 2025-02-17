import torch
import numpy as np
from sklearn.base import BaseEstimator
import json
import os

class FrameworkConverter:
    """Handles model conversion between different frameworks"""

    def __init__(self):
        self.supported_frameworks = ["pytorch", "sklearn"]
        self._check_tensorflow()

    def _check_tensorflow(self):
        """Check if TensorFlow is available"""
        try:
            import tensorflow as tf
            self.supported_frameworks.append("tensorflow")
            self.tf = tf
            self.tf_available = True
        except ImportError:
            print("TensorFlow not available. TensorFlow-related features will be disabled.")
            self.tf_available = False
            self.tf = None

    def detect_framework(self, model):
        """Detect the framework of the input model"""
        if isinstance(model, torch.nn.Module):
            return "pytorch"
        elif isinstance(model, BaseEstimator):
            return "sklearn"
        elif self.tf_available and isinstance(model, self.tf.keras.Model):
            return "tensorflow"
        else:
            raise ValueError("Unsupported model type")

    def save_model(self, model, save_path, framework=None):
        """Save model in the specified framework format"""
        if framework is None:
            framework = self.detect_framework(model)

        if framework == "tensorflow" and not self.tf_available:
            raise ImportError("TensorFlow is not installed. Please install TensorFlow to use this feature.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if framework == "pytorch":
            if isinstance(model, torch.nn.Module):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': model.config.to_dict() if hasattr(model, 'config') else {},
                    'framework': 'pytorch'
                }, save_path)
            else:
                raise ValueError("Model is not a PyTorch model")

        elif framework == "sklearn":
            if isinstance(model, BaseEstimator):
                from joblib import dump
                dump(model, save_path)
                with open(f"{save_path}.metadata", 'w') as f:
                    json.dump({'framework': 'sklearn'}, f)
            else:
                raise ValueError("Model is not a Scikit-learn model")

        elif framework == "tensorflow":
            if isinstance(model, self.tf.keras.Model):
                model.save(save_path)
                with open(f"{save_path}/metadata.json", 'w') as f:
                    json.dump({'framework': 'tensorflow'}, f)
            else:
                raise ValueError("Model is not a TensorFlow model")

    def load_model(self, load_path, target_framework=None):
        """Load model and optionally convert to target framework"""
        if os.path.isfile(load_path):
            if load_path.endswith('.pt') or load_path.endswith('.pth'):
                source_framework = 'pytorch'
            elif load_path.endswith('.joblib'):
                source_framework = 'sklearn'
            else:
                try:
                    with open(f"{load_path}/metadata.json", 'r') as f:
                        metadata = json.load(f)
                        source_framework = metadata.get('framework')
                except:
                    raise ValueError("Could not determine source framework")

        if source_framework == "tensorflow" and not self.tf_available:
            raise ImportError("TensorFlow is not installed. Please install TensorFlow to load this model.")

        if source_framework == "pytorch":
            checkpoint = torch.load(load_path)
            model_state = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
            return model_state, config

        elif source_framework == "sklearn":
            from joblib import load
            model = load(load_path)
            if target_framework:
                raise NotImplementedError(
                    f"Conversion from sklearn to {target_framework} not supported yet"
                )
            return model

        elif source_framework == "tensorflow":
            model = self.tf.keras.models.load_model(load_path)
            return model

    def get_supported_formats(self):
        """Return supported model formats and conversions"""
        formats = {
            "pytorch": {
                "extensions": [".pt", ".pth"],
                "can_convert_to": ["sklearn"],
            },
            "sklearn": {
                "extensions": [".joblib"],
                "can_convert_to": ["pytorch"],
            }
        }

        if self.tf_available:
            formats["tensorflow"] = {
                "extensions": [".h5", ".keras"],
                "can_convert_to": ["pytorch", "sklearn"],
            }
            formats["pytorch"]["can_convert_to"].append("tensorflow")
            formats["sklearn"]["can_convert_to"].append("tensorflow")

        return formats

    def _pytorch_to_tensorflow(self, pytorch_state, config):
        """Convert PyTorch model state to TensorFlow model"""
        # Basic implementation for now
        input_size = config.get('input_size', 768)  # Default for many transformer models

        # Create a simple feed-forward network as an example
        tf_model = self.tf.keras.Sequential([
            self.tf.keras.layers.Dense(input_size, activation='relu'),
            self.tf.keras.layers.Dense(config.get('hidden_size', 512), activation='relu'),
            self.tf.keras.layers.Dense(config.get('num_classes', 2), activation='softmax')
        ])

        # Convert weights (simplified)
        for tf_layer, (name, pt_weights) in zip(tf_model.layers, pytorch_state.items()):
            if 'weight' in name.lower():
                tf_layer.set_weights([
                    pt_weights.numpy(),
                    np.zeros(tf_layer.bias.shape)
                ])

        return tf_model

    def _tensorflow_to_pytorch(self, tf_model):
        """Convert TensorFlow model to PyTorch model"""
        # Basic implementation for now
        class PyTorchModel(torch.nn.Module):
            def __init__(self, tf_model):
                super().__init__()
                self.layers = torch.nn.ModuleList()

                # Convert each layer (simplified)
                for layer in tf_model.layers:
                    if isinstance(layer, self.tf.keras.layers.Dense):
                        pt_layer = torch.nn.Linear(
                            layer.input_shape[-1],
                            layer.units
                        )
                        # Copy weights
                        weights = layer.get_weights()
                        pt_layer.weight.data = torch.from_numpy(weights[0].T)
                        if len(weights) > 1:
                            pt_layer.bias.data = torch.from_numpy(weights[1])
                        self.layers.append(pt_layer)

            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x

        return PyTorchModel(tf_model)