import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import clip

class ClipMetrics:
    """
    Metrics for CLIP embedding evaluation
    """
    
    def __init__(self, device):
        self.device = device
    
    def embedding_accuracy(self, pred_embeddings, target_embeddings, labels):
        """Calculate accuracy in embedding space"""
        # Compute cosine similarities
        similarities = F.cosine_similarity(
            pred_embeddings.unsqueeze(1), 
            target_embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Top-1 accuracy
        pred_indices = similarities.argmax(dim=1)
        true_indices = torch.arange(len(pred_embeddings)).to(self.device)
        top1_acc = (pred_indices == true_indices).float().mean().item()
        
        # Top-5 accuracy
        top5_acc = 0
        for i in range(len(pred_embeddings)):
            top5_indices = similarities[i].topk(5).indices
            if i in top5_indices:
                top5_acc += 1
        top5_acc = top5_acc / len(pred_embeddings)
        
        return top1_acc * 100, top5_acc * 100
    
    def cosine_similarity(self, pred_embeddings, target_embeddings):
        """Average cosine similarity"""
        cosine_sim = F.cosine_similarity(pred_embeddings, target_embeddings)
        return cosine_sim.mean().item()
    
    def embedding_distance(self, pred_embeddings, target_embeddings):
        """Average L2 distance between embeddings"""
        l2_distance = torch.norm(pred_embeddings - target_embeddings, p=2, dim=1)
        return l2_distance.mean().item()

def calculate_clip_accuracy(pred_embeddings, target_embeddings, device):
    """Convenience function for CLIP accuracy"""
    metrics = ClipMetrics(device)
    return metrics.embedding_accuracy(pred_embeddings, target_embeddings, None)