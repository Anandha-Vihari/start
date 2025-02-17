import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, PromptTuningConfig

class ModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.auth_token = os.getenv("HUGGING_FACE_TOKEN")

    def load_model(self):
        """Load the base model and tokenizer"""
        try:
            # Check if model is from Llama family
            if "meta-llama" in self.model_name and not self.auth_token:
                raise Exception("Hugging Face token is required for Llama models")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=self.auth_token  # Will be None for non-Llama models
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.auth_token
            )

            return self.model

        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def apply_peft(self, method="LoRA"):
        """Apply PEFT method to the model"""
        if method == "LoRA":
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
        elif method == "Prefix Tuning":
            peft_config = PrefixTuningConfig(
                num_virtual_tokens=20,
                task_type="CAUSAL_LM"
            )
        else:  # P-Tuning
            peft_config = PromptTuningConfig(
                num_virtual_tokens=20,
                task_type="CAUSAL_LM"
            )

        self.model = get_peft_model(self.model, peft_config)
        return self.model