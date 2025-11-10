import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import clip

from models.eeg_encoder import EEGToClipModel
from models.decoder import ClipTextToImage
from data.eeg_dataset import EEGInferenceDataset
from config import Config

def generate_embeddings(model, dataloader, device, num_samples=None):
    """Generate CLIP embeddings from EEG data"""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_info = []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg_data = batch['eeg'].to(device)
            
            # Generate embeddings
            embeddings = model(eeg_data)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(batch['label'])
            all_info.append({
                'image_name': batch['image_name'],
                'subject': batch['subject'],
                'granularity': batch['granularity']
            })
            
            if num_samples and len(torch.cat(all_embeddings)) >= num_samples:
                break
    
    all_embeddings = torch.cat(all_embeddings)
    if num_samples:
        all_embeddings = all_embeddings[:num_samples]
        all_labels = all_labels[:num_samples]
    
    return all_embeddings, all_labels, all_info

def save_embeddings(embeddings, labels, info, output_dir):
    """Save generated embeddings to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy array
    np.save(os.path.join(output_dir, 'eeg_embeddings.npy'), embeddings.numpy())
    
    # Save metadata
    with open(os.path.join(output_dir, 'embedding_metadata.txt'), 'w') as f:
        f.write("EEG to CLIP Embedding Results\n")
        f.write("=============================\n")
        for i, (label, emb_info) in enumerate(zip(labels, info)):
            f.write(f"Sample {i}:\n")
            f.write(f"  Label: {label}\n")
            f.write(f"  Image: {emb_info['image_name']}\n")
            f.write(f"  Subject: {emb_info['subject']}\n")
            f.write(f"  Granularity: {emb_info['granularity']}\n")
            f.write(f"  Embedding: {embeddings[i][:5]}...\n\n")
    
    print(f"Saved embeddings and metadata to {output_dir}")

def visualize_embeddings(embeddings, labels, output_path, method='pca'):
    """Visualize embeddings using dimensionality reduction"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # Convert to numpy
    embeddings_np = embeddings.numpy()
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:  # t-sne
        reducer = TSNE(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(embeddings_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(embeddings)), cmap='viridis')
    plt.colorbar(scatter, label='Sample Index')
    plt.title(f'EEG Embeddings Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    # Add some labels
    unique_labels = list(set(labels))
    for i, label in enumerate(unique_labels[:10]):  # Show first 10 labels
        idx = labels.index(label)
        plt.annotate(label, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved embedding visualization: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='DreamLens EEG-to-CLIP Inference')
    parser.add_argument('--data_path', type=str, required=True, help='Path to EEG data .pth file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to process')
    parser.add_argument('--generate_images', action='store_true', help='Generate images from embeddings')
    parser.add_argument('--granularity', type=str, default='both', choices=['coarse', 'fine', 'both'])
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(Config.DEVICE)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model = EEGToClipModel(
        eeg_channels=Config.EEG_CHANNELS,
        eeg_timepoints=Config.EEG_TIMEPOINTS,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create dataset
    dataset = EEGInferenceDataset(
        data_path=args.data_path,
        granularity=args.granularity
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Generate embeddings
    print("Generating CLIP embeddings from EEG...")
    embeddings, labels, info = generate_embeddings(
        model, dataloader, device, args.num_samples
    )
    
    # Save embeddings
    save_embeddings(embeddings, labels, info, args.output_dir)
    
    # Visualize embeddings
    visualize_embeddings(
        embeddings, labels, 
        os.path.join(args.output_dir, 'embeddings_visualization.png')
    )
    
    # Generate images if requested
    if args.generate_images:
        print("Generating images from embeddings...")
        image_generator = ClipTextToImage(device)
        
        # Create output paths
        output_paths = [
            os.path.join(args.output_dir, f'generated_{i:04d}.png') 
            for i in range(min(10, len(embeddings)))
        ]
        
        # Generate images
        embeddings_device = embeddings[:10].to(device)  # Generate first 10
        image_generator.generate_images(embeddings_device, output_paths)
    
    print("Inference completed!")

if __name__ == '__main__':
    main()