import torch
from torch.utils.data import Dataset
import numpy as np
import os
import clip

class EEGClipDataset(Dataset):
    """
    Dataset for EEG data with CLIP embeddings as targets
    """
    def __init__(self, data_path, clip_model=None, preprocess=None, mode='train', granularity='both'):
        self.data_path = data_path
        self.mode = mode
        self.granularity = granularity

        self.clip_model = clip_model
        self.preprocess = preprocess
        
        # Load the dataset
        if data_path.endswith('.pth'):
            data = torch.load(data_path, map_location='cpu', weights_only=False)

        else:
            raise ValueError("Unsupported data format. Use .pth files")
        
        self.dataset = data['dataset']
        self.labels = data.get('labels', [])
        
        # Filter by granularity
        if granularity != 'both':
            self.dataset = [item for item in self.dataset if item['granularity'] == granularity]
        
        # Precompute CLIP text embeddings for labels
        self.label_embeddings = self._precompute_label_embeddings()
        
        print(f"Loaded {len(self.dataset)} samples with granularity: {granularity}")
    
    def _precompute_label_embeddings(self):
        """Precompute CLIP embeddings for all unique labels"""
        if self.clip_model is None:
            return {}
            
        unique_labels = list(set([item['label'] for item in self.dataset]))
        
        # Convert ImageNet labels to descriptive text
        label_descriptions = self._imagenet_label_to_text(unique_labels)
        
        with torch.no_grad():
            text_tokens = clip.tokenize(label_descriptions)
            if self.clip_model is not None:
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            else:
                # Fallback: random embeddings
                text_features = torch.randn(len(unique_labels), 512)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
        return {label: embedding for label, embedding in zip(unique_labels, text_features)}
    
    def _imagenet_label_to_text(self, labels):
        """Convert ImageNet labels to descriptive text prompts"""
        descriptions = []
        for label in labels:
            # Convert label like 'n02106662' to descriptive text
            # In practice, you might want to map these to actual class names
            descriptions.append(f"a photo of a {label}")
        return descriptions
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get EEG data
        eeg_data = item['eeg_data']
        if isinstance(eeg_data, torch.Tensor):
            eeg_tensor = eeg_data.float()
        else:
            eeg_tensor = torch.FloatTensor(eeg_data)
        
        # Get label embedding as target
        label = item['label']
        if label in self.label_embeddings:
            target_embedding = self.label_embeddings[label]
        else:
            # Fallback: zero embedding
            target_embedding = torch.zeros(512)
        
        return {
            'eeg': eeg_tensor,
            'target_embedding': target_embedding,
            'label': label,
            'image_name': item['image'],
            'subject': item['subject'],
            'granularity': item['granularity']
        }

class EEGInferenceDataset(Dataset):
    """Dataset for EEG inference"""
    def __init__(self, data_path, granularity='both'):
        self.data_path = data_path
        
        if data_path.endswith('.pth'):
            data = torch.load(data_path, map_location='cpu')
        else:
            raise ValueError("Unsupported data format. Use .pth files")
        
        self.dataset = data['dataset']
        
        if granularity != 'both':
            self.dataset = [item for item in self.dataset if item['granularity'] == granularity]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        eeg_data = item['eeg_data']
        if isinstance(eeg_data, torch.Tensor):
            eeg_tensor = eeg_data.float()
        else:
            eeg_tensor = torch.FloatTensor(eeg_data)
        
        return {
            'eeg': eeg_tensor,
            'label': item['label'],
            'image_name': item['image'],
            'subject': item['subject'],
            'granularity': item['granularity']
        }