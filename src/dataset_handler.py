import pandas as pd
from torch.utils.data import Dataset

class DatasetHandler:
    def load_dataset(self, file):
        """Load and preprocess dataset from CSV file"""
        try:
            df = pd.read_csv(file)
            return df
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def prepare_dataset(self, df, tokenizer, max_length=512):
        """Convert DataFrame to PyTorch Dataset"""
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item
            
            def __len__(self):
                return len(self.labels)
        
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        return TextDataset(texts, labels, tokenizer, max_length)
