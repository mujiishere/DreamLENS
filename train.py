import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import clip

from models.eeg_encoder import EEGToClipModel
from data.eeg_dataset import EEGClipDataset
from utils.losses import ClipLoss
from utils.metrics import ClipMetrics
from config import Config

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Training loop for one epoch"""
    model.train()
    
    running_loss = 0.0
    running_mse = 0.0
    running_cosine = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')
    
    for batch_idx, batch in enumerate(pbar):
        eeg_data = batch['eeg'].to(device)
        target_embeddings = batch['target_embedding'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_embeddings = model(eeg_data)
        
        # Calculate loss
        total_loss, mse_loss, cosine_loss = criterion(pred_embeddings, target_embeddings)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += total_loss.item()
        running_mse += mse_loss.item()
        running_cosine += (1 - cosine_loss)  # Convert back to similarity
        
        pbar.set_postfix({
            'Total Loss': f'{total_loss.item():.4f}',
            'MSE': f'{mse_loss.item():.4f}',
            'Cosine Sim': f'{(1 - cosine_loss):.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_mse = running_mse / len(dataloader)
    epoch_cosine = running_cosine / len(dataloader)
    
    return epoch_loss, epoch_mse, epoch_cosine

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validation loop for one epoch"""
    model.eval()
    
    running_loss = 0.0
    running_mse = 0.0
    running_cosine = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    
    metrics_calculator = ClipMetrics(device)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} Validation')
        
        for batch_idx, batch in enumerate(pbar):
            eeg_data = batch['eeg'].to(device)
            target_embeddings = batch['target_embedding'].to(device)
            
            # Forward pass
            pred_embeddings = model(eeg_data)
            
            # Calculate loss
            total_loss, mse_loss, cosine_loss = criterion(pred_embeddings, target_embeddings)
            
            # Calculate metrics
            top1_acc, top5_acc = metrics_calculator.embedding_accuracy(
                pred_embeddings, target_embeddings, batch['label']
            )
            cosine_sim = metrics_calculator.cosine_similarity(pred_embeddings, target_embeddings)
            
            # Update statistics
            running_loss += total_loss.item()
            running_mse += mse_loss.item()
            running_cosine += cosine_sim
            running_top1 += top1_acc
            running_top5 += top5_acc
            
            pbar.set_postfix({
                'Val Loss': f'{total_loss.item():.4f}',
                'Top-1 Acc': f'{top1_acc:.2f}%',
                'Cosine Sim': f'{cosine_sim:.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_mse = running_mse / len(dataloader)
    epoch_cosine = running_cosine / len(dataloader)
    epoch_top1 = running_top1 / len(dataloader)
    epoch_top5 = running_top5 / len(dataloader)
    
    return epoch_loss, epoch_mse, epoch_cosine, epoch_top1, epoch_top5

def plot_training_history(train_history, val_history, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, train_history['loss'], label='Train Loss')
    axes[0, 0].plot(epochs, val_history['loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # MSE
    axes[0, 1].plot(epochs, train_history['mse'], label='Train MSE')
    axes[0, 1].plot(epochs, val_history['mse'], label='Val MSE')
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].legend()
    
    # Cosine Similarity
    axes[0, 2].plot(epochs, train_history['cosine_sim'], label='Train Cosine Sim')
    axes[0, 2].plot(epochs, val_history['cosine_sim'], label='Val Cosine Sim')
    axes[0, 2].set_title('Cosine Similarity')
    axes[0, 2].legend()
    
    # Top-1 Accuracy
    axes[1, 0].plot(epochs, val_history['top1_acc'])
    axes[1, 0].set_title('Validation Top-1 Accuracy (%)')
    
    # Top-5 Accuracy
    axes[1, 1].plot(epochs, val_history['top5_acc'])
    axes[1, 1].set_title('Validation Top-5 Accuracy (%)')
    
    # Learning rate curve (placeholder)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='DreamLens EEG-to-CLIP Training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to EEG data .pth file')
    parser.add_argument('--granularity', type=str, default='both', choices=['coarse', 'fine', 'both'])
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--save_dir', type=str, default=Config.CHECKPOINT_DIR)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # Load CLIP model
    clip_model, _ = clip.load(Config.CLIP_MODEL_NAME, device=device)
    
    # Create datasets
    full_dataset = EEGClipDataset(
        data_path=args.data_path,
        clip_model=clip_model,
        granularity=args.granularity
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = EEGToClipModel(
        eeg_channels=Config.EEG_CHANNELS,
        eeg_timepoints=Config.EEG_TIMEPOINTS,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained_model_path=Config.PRETRAINED_MODEL_PATH
    ).to(device)
    
    # Loss and optimizer
    criterion = ClipLoss(mse_weight=1.0, cosine_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training history
    train_history = {'loss': [], 'mse': [], 'cosine_sim': []}
    val_history = {'loss': [], 'mse': [], 'cosine_sim': [], 'top1_acc': [], 'top5_acc': []}
    
    best_val_loss = float('inf')
    
    print("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        # Training
        train_loss, train_mse, train_cosine = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_loss, val_mse, val_cosine, val_top1, val_top5 = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        train_history['loss'].append(train_loss)
        train_history['mse'].append(train_mse)
        train_history['cosine_sim'].append(train_cosine)
        
        val_history['loss'].append(val_loss)
        val_history['mse'].append(val_mse)
        val_history['cosine_sim'].append(val_cosine)
        val_history['top1_acc'].append(val_top1)
        val_history['top5_acc'].append(val_top5)
        
        # Print summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Cosine Sim: {train_cosine:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Cosine Sim: {val_cosine:.4f}')
        print(f'Val Top-1 Acc: {val_top1:.2f}%, Top-5 Acc: {val_top5:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history,
                'best_val_loss': best_val_loss,
                'config': Config.__dict__
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            # Plot training history
            plot_training_history(
                train_history, val_history,
                os.path.join(args.save_dir, f'training_history_epoch_{epoch}.png')
            )
    
    print("Training completed!")

if __name__ == '__main__':
    main()