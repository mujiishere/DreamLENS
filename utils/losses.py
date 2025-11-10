import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipLoss(nn.Module):
    """
    Combined loss for CLIP embedding training
    """
    def __init__(self, mse_weight=1.0, cosine_weight=1.0, temperature=0.1):
        super(ClipLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.temperature = temperature
    
    def forward(self, pred_embeddings, target_embeddings):
        # MSE loss
        mse_loss = self.mse_loss(pred_embeddings, target_embeddings)
        
        # Cosine similarity loss (we want to maximize similarity)
        cosine_sim = F.cosine_similarity(pred_embeddings, target_embeddings).mean()
        cosine_loss = 1 - cosine_sim  # Convert to loss
        
        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.cosine_weight * cosine_loss)
        
        return total_loss, mse_loss, cosine_loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for CLIP-style training
    """
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, eeg_embeddings, text_embeddings, labels):
        # Normalize embeddings
        eeg_embeddings = F.normalize(eeg_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(eeg_embeddings, text_embeddings.T) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        batch_size = eeg_embeddings.shape[0]
        labels = torch.arange(batch_size).to(eeg_embeddings.device)
        
        # Cross entropy loss
        loss_eeg_to_text = F.cross_entropy(similarity_matrix, labels)
        loss_text_to_eeg = F.cross_entropy(similarity_matrix.T, labels)
        
        total_loss = (loss_eeg_to_text + loss_text_to_eeg) / 2
        
        return total_loss