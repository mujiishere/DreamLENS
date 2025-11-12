import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SpatialAttention(nn.Module):
    """Spatial attention for EEG channels - learns which channels are important"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, channels, time)
        attention = self.sigmoid(self.conv(x))  # (batch, 1, time)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention - learns importance of different EEG electrodes"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, channels, time)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * attention


class CrossAttentionBlock(nn.Module):
    """Cross-attention between different frequency bands"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        attn_out, attn_weights = self.cross_attention(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_out)), attn_weights


class TransformerBlock(nn.Module):
    """Enhanced transformer encoder block with relative position encoding"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable scaling factors for residual connections
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Self-attention with learnable residual scaling
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.alpha1 * attn_out)
        
        # Feed-forward with learnable residual scaling
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.alpha2 * ff_out)
        
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Efficient depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple temporal scales"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.scale1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.scale2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.scale3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.scale4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=9, padding=4)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        out = torch.cat([s1, s2, s3, s4], dim=1)
        return F.relu(self.bn(out))


class CNNEncoder(nn.Module):
    """
    CNN for EEG feature extraction (original implementation)
    """
    def __init__(self, input_channels=62, hidden_dim=256):
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


class HybridCNNTransformerEncoder(nn.Module):
    """
    Hybrid CNN-Transformer encoder for improved EEG feature extraction
    """
    def __init__(self, input_channels=62, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super(HybridCNNTransformerEncoder, self).__init__()
        
        # Initial CNN feature extraction
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Transformer layers for temporal modeling
        self.pos_encoding = PositionalEncoding(256, max_len=5000)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(256, num_heads, 1024, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final CNN layers
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.conv4 = nn.Conv1d(512, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (batch, channels, time_points)
        # Initial CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Reshape for transformer: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Reshape back: (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        
        # Final CNN layers
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        
        return x


class NovelEEGEncoder(nn.Module):
    """
    State-of-the-art EEG encoder with multiple innovations:
    - Multi-scale feature extraction
    - Spatial and channel attention mechanisms
    - Hybrid CNN-Transformer with cross-attention
    - Depthwise separable convolutions for efficiency
    - Learnable residual scaling
    """
    def __init__(self, input_channels=62, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super(NovelEEGEncoder, self).__init__()
        
        # Stage 1: Multi-scale feature extraction with channel attention
        self.multi_scale_conv1 = MultiScaleFeatureExtractor(input_channels, 128)
        self.channel_attn1 = ChannelAttention(128)
        self.spatial_attn1 = SpatialAttention(128)
        
        # Stage 2: Efficient depthwise separable convolution
        self.ds_conv2 = DepthwiseSeparableConv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.channel_attn2 = ChannelAttention(256)
        
        # Stage 3: Transformer for long-range temporal dependencies
        self.pos_encoding = PositionalEncoding(256, max_len=5000)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(256, num_heads, 1024, dropout) 
            for _ in range(num_layers)
        ])
        
        # Cross-attention between early and late features
        self.cross_attention = CrossAttentionBlock(256, num_heads, dropout)
        
        # Stage 4: Feature refinement
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.spatial_attn3 = SpatialAttention(512)
        
        # Stage 5: Final projection
        self.conv4 = nn.Conv1d(512, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        # Global context aggregation
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # x shape: (batch, channels, time_points)
        
        # Stage 1: Multi-scale feature extraction with attention
        x = self.multi_scale_conv1(x)
        x = self.channel_attn1(x)
        x = self.spatial_attn1(x)
        early_features = x.clone()
        
        # Stage 2: Efficient convolution with attention
        x = F.gelu(self.bn2(self.ds_conv2(x)))
        x = self.channel_attn2(x)
        
        # Stage 3: Transformer processing
        # Reshape for transformer: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        
        # Store intermediate transformer features for cross-attention
        transformer_input = x.clone()
        
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Cross-attention between transformer input and output
        x, attn_weights = self.cross_attention(x, transformer_input)
        
        # Reshape back: (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        
        # Stage 4: Feature refinement
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.spatial_attn3(x)
        
        # Stage 5: Final projection
        x = F.gelu(self.bn4(self.conv4(x)))
        
        # Global context aggregation (both avg and max pooling)
        avg_features = self.adaptive_avg_pool(x).squeeze(-1)
        max_features = self.adaptive_max_pool(x).squeeze(-1)
        
        # Fuse both pooling strategies
        combined_features = torch.cat([avg_features, max_features], dim=1)
        output = self.feature_fusion(combined_features)
        
        return output


class EEGToClipModel(nn.Module):
    """
    Map EEG signals to CLIP embedding space
    Now supports: CNN-only, Hybrid CNN-Transformer, and Novel State-of-the-art architectures
    """
    def __init__(self, eeg_channels=62, eeg_timepoints=440, 
                 embedding_dim=512, pretrained_model_path=None,
                 architecture='novel', num_transformer_layers=4, num_heads=8):
        """
        Args:
            architecture: 'cnn', 'hybrid', or 'novel'
                - 'cnn': Original CNN-only encoder
                - 'hybrid': CNN + Transformer encoder
                - 'novel': State-of-the-art encoder with all innovations
        """
        super(EEGToClipModel, self).__init__()
        
        self.architecture = architecture
        
        # Choose encoder architecture
        if architecture == 'novel':
            self.eeg_encoder = NovelEEGEncoder(
                input_channels=eeg_channels,
                hidden_dim=512,
                num_heads=num_heads,
                num_layers=num_transformer_layers,
                dropout=0.1
            )
        elif architecture == 'hybrid':
            self.eeg_encoder = HybridCNNTransformerEncoder(
                input_channels=eeg_channels, 
                hidden_dim=512,
                num_heads=num_heads,
                num_layers=num_transformer_layers,
                dropout=0.1
            )
        else:  # 'cnn'
            self.eeg_encoder = CNNEncoder(input_channels=eeg_channels, hidden_dim=512)
        
        # Enhanced embedding mapper with residual connections
        self.embedding_mapper = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.2)
            ),
            nn.Linear(1024, embedding_dim)
        ])
        
        # Learnable temperature for contrastive learning
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Load pretrained weights
        if pretrained_model_path:
            self.load_pretrained_weights(pretrained_model_path)
    
    def load_pretrained_weights(self, model_path):
        """Load pretrained EEG-ImageNet weights"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
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
                    if name in model_state_dict:
                        if len(v.shape) == len(model_state_dict[name].shape):
                            if v.shape[0] == model_state_dict[name].shape[0]:
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
        # Extract EEG features
        eeg_features = self.eeg_encoder(eeg_data)
        
        # Apply embedding mapper with residual connections
        x = self.embedding_mapper[0](eeg_features)
        residual = x
        x = self.embedding_mapper[1](x)
        x = x + residual  # Residual connection
        embeddings = self.embedding_mapper[2](x)
        
        # L2 normalization
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_temperature(self):
        """Return learnable temperature for contrastive loss"""
        return self.temperature.clamp(min=0.01, max=0.5)