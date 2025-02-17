import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

class DataAugmentor:
    """Handles various data augmentation techniques for text data"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mlm_data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
    
    def apply_augmentation(self, texts, method="mlm", **kwargs):
        """Apply specified augmentation method to the input texts"""
        if method == "mlm":
            return self._masked_language_modeling(texts, **kwargs)
        elif method == "backtranslation":
            return self._back_translation(texts, **kwargs)
        elif method == "synonym_replacement":
            return self._synonym_replacement(texts, **kwargs)
        else:
            raise ValueError(f"Unsupported augmentation method: {method}")
    
    def _masked_language_modeling(self, texts, num_augmentations=1):
        """Augment data using masked language modeling"""
        augmented_texts = []
        
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        for _ in range(num_augmentations):
            # Apply MLM data collator
            mlm_inputs = self.mlm_data_collator(
                [{"input_ids": ids} for ids in encodings["input_ids"]]
            )
            
            # Decode augmented texts
            augmented = self.tokenizer.batch_decode(
                mlm_inputs["input_ids"],
                skip_special_tokens=True
            )
            augmented_texts.extend(augmented)
        
        return augmented_texts
    
    def _back_translation(self, texts, target_lang="fr", source_lang="en"):
        """Augment data using back-translation"""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Load translation models
            model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
            forward_tokenizer = MarianTokenizer.from_pretrained(model_name)
            forward_model = MarianMTModel.from_pretrained(model_name)
            
            back_model_name = f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}'
            backward_tokenizer = MarianTokenizer.from_pretrained(back_model_name)
            backward_model = MarianMTModel.from_pretrained(back_model_name)
            
            # Forward translation
            inputs = forward_tokenizer(texts, return_tensors="pt", padding=True)
            translated = forward_model.generate(**inputs)
            translated_texts = forward_tokenizer.batch_decode(translated, skip_special_tokens=True)
            
            # Backward translation
            inputs = backward_tokenizer(translated_texts, return_tensors="pt", padding=True)
            back_translated = backward_model.generate(**inputs)
            augmented_texts = backward_tokenizer.batch_decode(back_translated, skip_special_tokens=True)
            
            return augmented_texts
        except ImportError:
            print("MarianMT models not available. Please install transformers with torch.")
            return texts
    
    def _synonym_replacement(self, texts, replacement_prob=0.1):
        """Augment data using synonym replacement"""
        try:
            from nltk.corpus import wordnet
            import nltk
            nltk.download('wordnet')
            nltk.download('punkt')
            
            augmented_texts = []
            for text in texts:
                words = nltk.word_tokenize(text)
                new_words = words.copy()
                
                for i, word in enumerate(words):
                    if np.random.random() < replacement_prob:
                        # Find synonyms
                        synonyms = []
                        for syn in wordnet.synsets(word):
                            for lemma in syn.lemmas():
                                if lemma.name() != word:
                                    synonyms.append(lemma.name())
                        
                        if synonyms:
                            new_words[i] = np.random.choice(synonyms)
                
                augmented_texts.append(" ".join(new_words))
            return augmented_texts
        except ImportError:
            print("NLTK not available. Please install nltk package.")
            return texts

class AugmentedDataset(Dataset):
    """Dataset class that incorporates augmented data"""
    
    def __init__(self, original_dataset, augmentor, augmentation_config):
        self.original_dataset = original_dataset
        self.augmentor = augmentor
        self.augmentation_config = augmentation_config
        self.augmented_data = self._create_augmented_data()
    
    def _create_augmented_data(self):
        """Create augmented versions of the original data"""
        texts = [item['text'] for item in self.original_dataset]
        augmented_texts = []
        
        for method, config in self.augmentation_config.items():
            if config.get('enabled', False):
                aug_texts = self.augmentor.apply_augmentation(
                    texts,
                    method=method,
                    **config.get('params', {})
                )
                augmented_texts.extend(aug_texts)
        
        # Create dataset items from augmented texts
        augmented_items = []
        for text in augmented_texts:
            item = {
                'text': text,
                'is_augmented': True
            }
            augmented_items.append(item)
        
        return augmented_items
    
    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_data)
    
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            item = self.original_dataset[idx]
            item['is_augmented'] = False
            return item
        
        aug_idx = idx - len(self.original_dataset)
        return self.augmented_data[aug_idx]
