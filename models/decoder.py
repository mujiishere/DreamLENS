import torch
import torch.nn as nn
import clip
from diffusers import StableDiffusionPipeline
import warnings

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
                safety_checker=None,  # Disable for faster loading
                requires_safety_checker=False
            ).to(device)
            print("Loaded Stable Diffusion pipeline")
        except Exception as e:
            print(f"Could not load Stable Diffusion: {e}")
            self.sd_pipeline = None
    
    def embedding_to_text(self, embeddings, candidate_prompts=None):
        """
        Convert CLIP embeddings to text prompts by finding nearest text embeddings
        """
        if candidate_prompts is None:
            candidate_prompts = [
                "a photo of an object", "a realistic image", "a clear photograph",
                "a high quality image", "a detailed picture", "a sharp photo",
                "a professional photograph", "a well lit image", "a colorful picture",
                "a beautiful image"
            ]
        
        with torch.no_grad():
            # Encode candidate prompts
            text_tokens = clip.tokenize(candidate_prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Find most similar prompts for each embedding
            similarities = torch.matmul(embeddings, text_features.T)
            top_indices = similarities.argmax(dim=1)
            
        generated_prompts = [candidate_prompts[idx] for idx in top_indices]
        return generated_prompts
    
    def generate_images(self, embeddings, output_paths, num_inference_steps=20):
        """
        Generate images from CLIP embeddings via text prompts
        """
        if self.sd_pipeline is None:
            print("Stable Diffusion not available. Cannot generate images.")
            return
        
        # Convert embeddings to text prompts
        text_prompts = self.embedding_to_text(embeddings)
        
        generated_images = []
        
        for i, prompt in enumerate(text_prompts):
            try:
                # Generate image
                image = self.sd_pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    height=256,
                    width=256
                ).images[0]
                
                # Save image if path provided
                if i < len(output_paths):
                    image.save(output_paths[i])
                    print(f"Generated: {output_paths[i]} - Prompt: '{prompt}'")
                
                generated_images.append(image)
                
            except Exception as e:
                print(f"Error generating image {i}: {e}")
                # Return black image as fallback
                fallback_image = Image.new('RGB', (256, 256), color='black')
                generated_images.append(fallback_image)
        
        return generated_images

class VAEDecoder(nn.Module):
    """
    Simple VAE decoder for generating images from latent vectors
    Alternative to Stable Diffusion
    """
    def __init__(self, latent_dim=512, output_channels=3, image_size=128):
        super(VAEDecoder, self).__init__()
        
        self.image_size = image_size
        
        self.decoder = nn.Sequential(
            # Input: (latent_dim, 4, 4)
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