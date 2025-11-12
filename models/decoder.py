import torch
import torch.nn as nn
import clip
from diffusers import StableDiffusionPipeline
from PIL import Image
import warnings
import math


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style transfer"""
    def __init__(self, num_features, latent_dim):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.scale_transform = nn.Linear(latent_dim, num_features)
        self.shift_transform = nn.Linear(latent_dim, num_features)
        
    def forward(self, x, latent):
        normalized = self.norm(x)
        scale = self.scale_transform(latent).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift_transform(latent).unsqueeze(-1).unsqueeze(-1)
        return scale * normalized + shift


class SelfAttention2D(nn.Module):
    """2D Self-attention for spatial features"""
    def __init__(self, in_channels):
        super(SelfAttention2D, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        value = self.value(x).view(batch, -1, height * width)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        return self.gamma * out + x


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block for image generation"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable residual scaling
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.alpha1 * attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.alpha2 * ff_out)
        
        return x


class PixelShuffle3D(nn.Module):
    """Pixel shuffle for upsampling with learned weights"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(PixelShuffle3D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        return self.shuffle(self.conv(x))


class ResidualBlock(nn.Module):
    """Residual block with normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return x + self.alpha * self.block(x)


class ClipTextToImage:
    """
    Wrapper for CLIP + Stable Diffusion text-to-image generation
    """
    def __init__(self, device, sd_model_name="runwayml/stable-diffusion-v1-5"):
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Load Stable Diffusion with error handling
        try:
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                sd_model_name,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)
            print("Loaded Stable Diffusion pipeline")
        except Exception as e:
            print(f"Could not load Stable Diffusion: {e}")
            self.sd_pipeline = None
    
    def embedding_to_text(self, embeddings, candidate_prompts=None):
        """Convert CLIP embeddings to text prompts by finding nearest text embeddings"""
        if candidate_prompts is None:
            candidate_prompts = [
                "a photo of an object", "a realistic image", "a clear photograph",
                "a high quality image", "a detailed picture", "a sharp photo",
                "a professional photograph", "a well lit image", "a colorful picture",
                "a beautiful image", "a natural scene", "an artistic composition"
            ]
        
        with torch.no_grad():
            text_tokens = clip.tokenize(candidate_prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            similarities = torch.matmul(embeddings, text_features.T)
            top_indices = similarities.argmax(dim=1)
            
        generated_prompts = [candidate_prompts[idx] for idx in top_indices]
        return generated_prompts
    
    def generate_images(self, embeddings, output_paths, num_inference_steps=20):
        """Generate images from CLIP embeddings via text prompts"""
        if self.sd_pipeline is None:
            print("Stable Diffusion not available. Cannot generate images.")
            return
        
        text_prompts = self.embedding_to_text(embeddings)
        generated_images = []
        
        for i, prompt in enumerate(text_prompts):
            try:
                image = self.sd_pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    height=256,
                    width=256
                ).images[0]
                
                if i < len(output_paths):
                    image.save(output_paths[i])
                    print(f"Generated: {output_paths[i]} - Prompt: '{prompt}'")
                
                generated_images.append(image)
                
            except Exception as e:
                print(f"Error generating image {i}: {e}")
                fallback_image = Image.new('RGB', (256, 256), color='black')
                generated_images.append(fallback_image)
        
        return generated_images


class VAEDecoder(nn.Module):
    """
    Simple VAE decoder for generating images from latent vectors
    Alternative to Stable Diffusion (original implementation)
    """
    def __init__(self, latent_dim=512, output_channels=3, image_size=128):
        super(VAEDecoder, self).__init__()
        
        self.image_size = image_size
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, output_channels, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.decoder(z)


class TransformerVAEDecoder(nn.Module):
    """
    Hybrid Transformer-CNN VAE decoder for improved image generation
    Uses transformer to process latent features before CNN upsampling
    """
    def __init__(self, latent_dim=512, output_channels=3, image_size=128, 
                 num_transformer_layers=4, num_heads=8, dropout=0.1):
        super(TransformerVAEDecoder, self).__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Project latent vector to sequence
        self.latent_to_seq = nn.Linear(latent_dim, 256 * 16)
        
        # Transformer decoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(256, num_heads, 1024, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Project back to spatial features
        self.seq_to_spatial = nn.Linear(256, 512 * 4 * 4)
        
        # CNN decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, output_channels, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # Project latent to sequence
        x = self.latent_to_seq(z)
        x = x.view(batch_size, 16, 256)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Project to spatial features
        x = self.seq_to_spatial(x)
        x = x.view(batch_size, 512, 4, 4)
        
        # Apply CNN decoder
        x = self.decoder(x)
        
        return x


class NovelImageDecoder(nn.Module):
    """
    State-of-the-art image decoder with multiple innovations:
    - Transformer-based latent processing
    - Adaptive Instance Normalization for style control
    - 2D Self-attention at multiple scales
    - Residual blocks for feature refinement
    - Pixel shuffle for efficient upsampling
    - Multi-scale feature fusion
    """
    def __init__(self, latent_dim=512, output_channels=3, image_size=128,
                 num_transformer_layers=6, num_heads=8, dropout=0.1):
        super(NovelImageDecoder, self).__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Latent conditioning network
        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        
        # Project latent to sequence for transformer
        self.latent_to_seq = nn.Sequential(
            nn.Linear(512, 512 * 32),
            nn.LayerNorm(512 * 32)
        )
        
        # Transformer decoder blocks with cross-attention capability
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(512, num_heads, 2048, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Project transformer output to spatial features
        self.seq_to_spatial = nn.Sequential(
            nn.Linear(512, 1024 * 4 * 4),
            nn.LayerNorm(1024 * 4 * 4)
        )
        
        # Initial spatial processing
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        
        # Residual blocks for feature refinement
        self.res_blocks_1 = nn.ModuleList([
            ResidualBlock(512) for _ in range(2)
        ])
        
        # Upsampling Stage 1: 4x4 -> 8x8
        self.upsample1 = nn.Sequential(
            PixelShuffle3D(512, 256, scale_factor=2),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.attn1 = SelfAttention2D(256)
        self.adain1 = AdaptiveInstanceNorm(256, 512)
        
        # Residual blocks
        self.res_blocks_2 = nn.ModuleList([
            ResidualBlock(256) for _ in range(2)
        ])
        
        # Upsampling Stage 2: 8x8 -> 16x16
        self.upsample2 = nn.Sequential(
            PixelShuffle3D(256, 128, scale_factor=2),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.attn2 = SelfAttention2D(128)
        self.adain2 = AdaptiveInstanceNorm(128, 512)
        
        # Upsampling Stage 3: 16x16 -> 32x32
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.attn3 = SelfAttention2D(64)
        self.adain3 = AdaptiveInstanceNorm(64, 512)
        
        # Upsampling Stage 4: 32x32 -> 64x64
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        # Upsampling Stage 5: 64x64 -> 128x128
        self.upsample5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        # Final refinement
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, output_channels, 7, 1, 3),
            nn.Tanh()
        )
        
        # Skip connection weights
        self.skip_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(3)
        ])
        
    def forward(self, z):
        batch_size = z.size(0)
        
        # Process latent vector
        processed_latent = self.latent_processor(z)
        
        # Project to sequence and apply transformer
        x = self.latent_to_seq(processed_latent)
        x = x.view(batch_size, 32, 512)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Aggregate temporal features
        x = x.mean(dim=1)  # (batch, 512)
        
        # Project to spatial features
        x = self.seq_to_spatial(x)
        x = x.view(batch_size, 1024, 4, 4)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Apply residual blocks
        for res_block in self.res_blocks_1:
            x = res_block(x)
        
        # Stage 1: 4x4 -> 8x8
        x = self.upsample1(x)
        x = self.attn1(x)
        x = self.adain1(x, processed_latent)
        skip1 = x.clone()
        
        for res_block in self.res_blocks_2:
            x = res_block(x)
        
        # Stage 2: 8x8 -> 16x16
        x = self.upsample2(x)
        x = self.attn2(x)
        x = self.adain2(x, processed_latent)
        skip2 = x.clone()
        
        # Stage 3: 16x16 -> 32x32
        x = self.upsample3(x)
        x = self.attn3(x)
        x = self.adain3(x, processed_latent)
        
        # Stage 4: 32x32 -> 64x64
        x = self.upsample4(x)
        
        # Stage 5: 64x64 -> 128x128
        x = self.upsample5(x)
        
        # Final refinement
        x = self.final_conv(x)
        
        return x


class HybridImageDecoder(nn.Module):
    """
    Flexible decoder that can use: VAE, Transformer-VAE, or Novel decoder
    Maintains backward compatibility
    """
    def __init__(self, latent_dim=512, output_channels=3, image_size=128,
                 architecture='novel', num_transformer_layers=6, num_heads=8):
        """
        Args:
            architecture: 'vae', 'transformer', or 'novel'
                - 'vae': Original simple VAE decoder
                - 'transformer': Transformer + CNN decoder
                - 'novel': State-of-the-art decoder with all innovations
        """
        super(HybridImageDecoder, self).__init__()
        
        self.architecture = architecture
        
        if architecture == 'novel':
            self.decoder = NovelImageDecoder(
                latent_dim=latent_dim,
                output_channels=output_channels,
                image_size=image_size,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads
            )
        elif architecture == 'transformer':
            self.decoder = TransformerVAEDecoder(
                latent_dim=latent_dim,
                output_channels=output_channels,
                image_size=image_size,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_heads
            )
        else:  # 'vae'
            self.decoder = VAEDecoder(
                latent_dim=latent_dim,
                output_channels=output_channels,
                image_size=image_size
            )
    
    def forward(self, z):
        return self.decoder(z)


class ProgressiveImageDecoder(nn.Module):
    """
    Progressive decoder that generates images at multiple resolutions
    Useful for progressive training and multi-scale supervision
    """
    def __init__(self, latent_dim=512, output_channels=3, max_resolution=128):
        super(ProgressiveImageDecoder, self).__init__()
        
        self.resolutions = [4, 8, 16, 32, 64, 128]
        self.decoders = nn.ModuleDict()
        
        current_channels = 512
        
        # Build decoders for each resolution
        for res in self.resolutions:
            if res <= max_resolution:
                self.decoders[f'decoder_{res}'] = NovelImageDecoder(
                    latent_dim=latent_dim,
                    output_channels=output_channels,
                    image_size=res,
                    num_transformer_layers=4,
                    num_heads=8
                )
        
    def forward(self, z, target_resolution=128):
        """Generate image at target resolution"""
        decoder_key = f'decoder_{target_resolution}'
        if decoder_key in self.decoders:
            return self.decoders[decoder_key](z)
        else:
            # Use largest available decoder
            available_resolutions = [int(k.split('_')[1]) for k in self.decoders.keys()]
            closest_res = max([r for r in available_resolutions if r <= target_resolution], default=min(available_resolutions))
            return self.decoders[f'decoder_{closest_res}'](z)
    
    def forward_all_resolutions(self, z):
        """Generate images at all resolutions for progressive training"""
        outputs = {}
        for res_key, decoder in self.decoders.items():
            resolution = int(res_key.split('_')[1])
            outputs[resolution] = decoder(z)
        return outputs