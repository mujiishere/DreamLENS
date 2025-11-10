import torch

class Config:
    # Data parameters
    EEG_CHANNELS = 128
    EEG_TIMEPOINTS = 440
    LATENT_DIM = 512
    EMBEDDING_DIM = 512  # CLIP embedding dimension
    
    # Model parameters
    PRETRAINED_MODEL_PATH = "EEG-ImageNet_2.pth"
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CLIP parameters
    CLIP_MODEL_NAME = "ViT-B/32"
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    OUTPUT_DIR = "generated_images"
    LOG_DIR = "logs"