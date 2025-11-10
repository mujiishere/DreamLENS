import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN for EEG feature extraction
    """
    def __init__(self, input_channels=128, hidden_dim=256):
        super(CNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.conv4 = nn.Conv1d(512, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (batch, channels, time_points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        
        return x

class EEGToClipModel(nn.Module):
    """
    Map EEG signals to CLIP embedding space
    """
    def __init__(self, eeg_channels=128, eeg_timepoints=440, 
                 embedding_dim=512, pretrained_model_path=None):
        super(EEGToClipModel, self).__init__()
        
        self.eeg_encoder = CNNEncoder(input_channels=eeg_channels, hidden_dim=512)
        
        # Mapping to CLIP embedding space
        self.embedding_mapper = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim)
        )
        
        # Load pretrained weights
        if pretrained_model_path:
            self.load_pretrained_weights(pretrained_model_path)
    
    def load_pretrained_weights(self, model_path):
        """Load pretrained EEG-ImageNet weights"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter and load compatible weights
            model_state_dict = self.state_dict()
            loaded_state_dict = {}
            
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                if name in model_state_dict and model_state_dict[name].shape == v.shape:
                    loaded_state_dict[name] = v
                else:
                    # Try to handle dimension mismatches
                    if name in model_state_dict:
                        if len(v.shape) == len(model_state_dict[name].shape):
                            if v.shape[0] == model_state_dict[name].shape[0]:
                                # Same output dimension, different input
                                min_dim = min(v.shape[1], model_state_dict[name].shape[1])
                                loaded_state_dict[name] = model_state_dict[name]
                                loaded_state_dict[name][:, :min_dim] = v[:, :min_dim]
                                print(f"Partially loaded {name} with dimension adjustment")
            
            self.load_state_dict(loaded_state_dict, strict=False)
            print(f"Successfully loaded {len(loaded_state_dict)} pretrained parameters")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Initializing with random weights")
    
    def forward(self, eeg_data):
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg_data)
        
        # Map to embedding space
        embeddings = self.embedding_mapper(eeg_features)
        
        # Normalize embeddings (like CLIP)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings