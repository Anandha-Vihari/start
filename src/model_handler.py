import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, PromptTuningConfig
from peft import AdaLoraConfig, IA3Config

class ModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.auth_token = os.getenv("HUGGING_FACE_TOKEN")
        self.supported_frameworks = {
            "pytorch": ["meta-llama", "facebook/opt", "EleutherAI"],
            "tensorflow": ["google/flan"]
        }

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

    def get_model_info(self):
        """Get model information and capabilities"""
        return {
            "name": self.model_name,
            "framework": self.get_framework(),
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }