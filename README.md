# DreamLens: EEG to CLIP Embedding Generator

A deep learning pipeline that maps EEG signals to CLIP semantic embeddings, enabling brainwave-to-image generation.

## Features

- **EEG Processing**: Bandpass filtering, normalization, and feature extraction
- **CLIP Integration**: Map EEG features to semantic embedding space
- **Text-to-Image**: Generate images from EEG via Stable Diffusion
- **Multi-modal Evaluation**: CLIP-based similarity metrics and accuracy

## Project Structure
dreamlens/
│
├── data/ # EEG dataset classes and loaders
│ └── eeg_dataset.py
│
├── models/ # Model architectures
│ ├── eeg_encoder.py # EEG feature encoder (CNN-based)
│ └── decoder.py # Image generation decoder
│
├── utils/ # Helper functions and metrics
│ ├── preprocessing.py
│ ├── losses.py
│ └── metrics.py
│
├── config.py # Configuration parameters
├── train.py # Model training script
├── infer.py # EEG-to-Image inference
├── evaluate.py # Model evaluation
└── requirements.txt # Project dependencies


## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt

python train.py --data_path /path/to/eeg_data.pth --granularity both

python infer.py --data_path /path/to/eeg_data.pth --checkpoint_path checkpoints/best_model.pth --generate_images

python evaluate.py --data_path /path/to/eeg_data.pth --checkpoint_path checkpoints/best_model.pth

Data Format
The EEG data should be in .pth format with the following structure:

python
{
    "dataset": [
        {
            "eeg_data": torch.Tensor,  # EEG signals
            "granularity": "coarse"/"fine",
            "subject": int,
            "label": str,  # ImageNet label
            "image": str   # Image filename
        },
        ...
    ],
    "labels": [...],
    "images": [...]
}

Model Architecture
EEG Encoder: CNN-based feature extraction from EEG signals

CLIP Mapper: Fully-connected layers mapping to CLIP embedding space

Image Generator: Stable Diffusion for text-to-image generation

Evaluation Metrics
Cosine Similarity: Semantic similarity between predicted and target embeddings

Top-1/Top-5 Accuracy: Classification accuracy in embedding space

Embedding Distance: L2 distance between embeddings

Output
CLIP embeddings from EEG signals

Generated images via text-to-image conversion

Visualization of embedding space

Comprehensive evaluation metrics


This complete implementation uses CLIP embeddings as the target space, allowing you to work with EEG data that only has labels (no actual images due to copyright). The system can generate meaningful semantic embeddings from EEG signals and optionally convert them to images using Stable Diffusion.