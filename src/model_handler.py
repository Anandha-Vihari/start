import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, PromptTuningConfig
from peft import AdaLoraConfig, IA3Config
from framework_converter import FrameworkConverter

class ModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.auth_token = os.getenv("HUGGING_FACE_TOKEN")
        self.supported_frameworks = {
            "pytorch": ["meta-llama", "facebook/opt", "EleutherAI"],
            "tensorflow": ["google/flan"]
        }
        self.framework_converter = FrameworkConverter()

    def get_framework(self):
        """Determine the framework based on model name"""
        for framework, prefixes in self.supported_frameworks.items():
            if any(prefix in self.model_name for prefix in prefixes):
                return framework
        return "pytorch"  # default to PyTorch

    def load_model(self):
        """Load the base model and tokenizer with framework-specific handling"""
        try:
            # Check if model is from Llama family
            if "meta-llama" in self.model_name and not self.auth_token:
                raise Exception("Hugging Face token is required for Llama models")

            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "token": self.auth_token,
                "device_map": "auto" if self.device == "cuda" else None
            }

            # Load model based on task
            if "t5" in self.model_name.lower():
                from transformers import T5ForConditionalGeneration
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.auth_token
            )

            return self.model

        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def get_peft_config(self, method, **kwargs):
        """Get PEFT configuration based on method and parameters"""
        configs = {
            "LoRA": LoraConfig(
                r=kwargs.get('r', 16),
                lora_alpha=kwargs.get('lora_alpha', 32),
                target_modules=kwargs.get('target_modules', ["q_proj", "v_proj"]),
                lora_dropout=kwargs.get('lora_dropout', 0.05),
                bias=kwargs.get('bias', "none"),
                task_type="CAUSAL_LM"
            ),
            "AdaLoRA": AdaLoraConfig(
                r=kwargs.get('r', 16),
                target_modules=kwargs.get('target_modules', ["q_proj", "v_proj"]),
                lora_alpha=kwargs.get('lora_alpha', 32),
                task_type="CAUSAL_LM"
            ),
            "Prefix Tuning": PrefixTuningConfig(
                num_virtual_tokens=kwargs.get('num_virtual_tokens', 20),
                task_type="CAUSAL_LM"
            ),
            "P-Tuning": PromptTuningConfig(
                num_virtual_tokens=kwargs.get('num_virtual_tokens', 20),
                task_type="CAUSAL_LM"
            ),
            "IA3": IA3Config(
                target_modules=kwargs.get('target_modules', ["q_proj", "v_proj"]),
                feedforward_modules=kwargs.get('feedforward_modules', ["down_proj"]),
                task_type="CAUSAL_LM"
            )
        }

        return configs.get(method)

    def apply_peft(self, method="LoRA", **kwargs):
        """Apply PEFT method to the model with custom configurations"""
        peft_config = self.get_peft_config(method, **kwargs)
        if peft_config is None:
            raise ValueError(f"Unsupported PEFT method: {method}")

        self.model = get_peft_model(self.model, peft_config)
        return self.model

    def export_model(self, save_path, target_framework=None):
        """Export the model to the specified framework format"""
        try:
            if target_framework and target_framework not in self.framework_converter.supported_frameworks:
                raise ValueError(f"Unsupported target framework: {target_framework}")

            # Save model and tokenizer
            os.makedirs(save_path, exist_ok=True)

            # Save the model
            self.framework_converter.save_model(
                self.model,
                os.path.join(save_path, "model"),
                target_framework
            )

            # Save the tokenizer
            self.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))

            # Save configuration
            config = {
                "model_name": self.model_name,
                "framework": target_framework or self.get_framework(),
                "device": self.device,
                "model_config": self.model.config.to_dict() if hasattr(self.model, 'config') else {}
            }

            with open(os.path.join(save_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)

            return True, "Model exported successfully"

        except Exception as e:
            return False, f"Error exporting model: {str(e)}"

    def import_model(self, load_path, target_framework=None):
        """Import a model from the specified path and optionally convert to target framework"""
        try:
            if target_framework and target_framework not in self.framework_converter.supported_frameworks:
                raise ValueError(f"Unsupported target framework: {target_framework}")

            # Load configuration
            with open(os.path.join(load_path, "config.json"), 'r') as f:
                config = json.load(f)

            # Load model
            model_path = os.path.join(load_path, "model")
            self.model = self.framework_converter.load_model(
                model_path,
                target_framework
            )

            # Load tokenizer
            tokenizer_path = os.path.join(load_path, "tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            self.model_name = config["model_name"]
            return True, "Model imported successfully"

        except Exception as e:
            return False, f"Error importing model: {str(e)}"

    def get_model_info(self):
        """Get model information and capabilities"""
        return {
            "name": self.model_name,
            "framework": self.get_framework(),
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

    def get_supported_export_formats(self):
        """Get list of supported export formats"""
        return self.framework_converter.get_supported_formats()