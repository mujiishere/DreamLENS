import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from tqdm import tqdm
import clip

from models.eeg_encoder import EEGToClipModel
from data.eeg_dataset import EEGClipDataset
from utils.metrics import ClipMetrics
from config import Config

class Evaluator:
    def __init__(self, device):
        self.device = device
        self.metrics = ClipMetrics(device)
    
    def evaluate_model(self, model, dataloader):
        """Comprehensive model evaluation"""
        model.eval()
        
        results = {
            'cosine_similarity': [],
            'embedding_distance': [],
            'top1_accuracy': [],
            'top5_accuracy': []
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                eeg_data = batch['eeg'].to(self.device)
                target_embeddings = batch['target_embedding'].to(self.device)
                
                # Generate embeddings
                pred_embeddings = model(eeg_data)
                
                # Calculate metrics
                cosine_sim = self.metrics.cosine_similarity(pred_embeddings, target_embeddings)
                embedding_dist = self.metrics.embedding_distance(pred_embeddings, target_embeddings)
                top1_acc, top5_acc = self.metrics.embedding_accuracy(
                    pred_embeddings, target_embeddings, batch['label']
                )
                
                results['cosine_similarity'].append(cosine_sim)
                results['embedding_distance'].append(embedding_dist)
                results['top1_accuracy'].append(top1_acc)
                results['top5_accuracy'].append(top5_acc)
        
        # Average results
        avg_results = {k: np.mean(v) for k, v in results.items()}
        std_results = {k: np.std(v) for k, v in results.items()}
        
        return avg_results, std_results

def main():
    parser = argparse.ArgumentParser(description='DreamLens Model Evaluation')
    parser.add_argument('--data_path', type=str, required=True, help='Path to EEG data .pth file')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--granularity', type=str, default='both', choices=['coarse', 'fine', 'both'])
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(Config.DEVICE)
    
    # Load CLIP model
    clip_model, _ = clip.load(Config.CLIP_MODEL_NAME, device=device)
    
    # Load model
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model = EEGToClipModel(
        eeg_channels=Config.EEG_CHANNELS,
        eeg_timepoints=Config.EEG_TIMEPOINTS,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset
    dataset = EEGClipDataset(
        data_path=args.data_path,
        clip_model=clip_model,
        granularity=args.granularity
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Evaluate
    evaluator = Evaluator(device)
    avg_results, std_results = evaluator.evaluate_model(model, dataloader)
    
    # Print results
    print("\n" + "="*50)
    print("DreamLens Evaluation Results")
    print("="*50)
    print(f"Dataset: {args.data_path}")
    print(f"Granularity: {args.granularity}")
    print(f"Samples: {len(dataset)}")
    print("\nMetrics:")
    print(f"Cosine Similarity: {avg_results['cosine_similarity']:.4f} ± {std_results['cosine_similarity']:.4f}")
    print(f"Embedding Distance: {avg_results['embedding_distance']:.4f} ± {std_results['embedding_distance']:.4f}")
    print(f"Top-1 Accuracy: {avg_results['top1_accuracy']:.2f}% ± {std_results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {avg_results['top5_accuracy']:.2f}% ± {std_results['top5_accuracy']:.2f}%")
    
    # Save results
    results_dir = os.path.dirname(args.checkpoint_path)
    results_file = os.path.join(results_dir, 'evaluation_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("DreamLens Evaluation Results\n")
        f.write("="*40 + "\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Granularity: {args.granularity}\n")
        f.write(f"Samples: {len(dataset)}\n\n")
        f.write("Metrics:\n")
        for metric in avg_results:
            f.write(f"{metric}: {avg_results[metric]:.4f} ± {std_results[metric]:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()