#!/usr/bin/env python3
"""
Training script for EmoTCN-Attn model with LOSO cross-validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import argparse
import os
import time
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import json

from model import EmoTCNAttn, count_parameters
from data_loader import create_loso_data_loaders
from utils import save_checkpoint, load_checkpoint, plot_training_curves

class Config:
    """Configuration class for training parameters"""
    
    def __init__(self):
        # Model parameters
        self.num_channels = 62
        self.kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
        self.dilation_rates = [1, 2, 4]
        self.num_tcn_blocks = 3
        self.num_attention_heads = 8
        self.dropout_rate = 0.2
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.num_epochs = 100
        self.early_stopping_patience = 15
        
        # Data parameters
        self.data_path = './data/'
        self.sampling_rate = 200
        self.segment_length = 800  # 4 seconds * 200 Hz
        
        # Paths
        self.checkpoint_dir = './checkpoints/'
        self.figure_dir = './figures/'
        self.log_dir = './logs/'
        
        # Hardware
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.squeeze().to(device)
        
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.squeeze().to(device)
            logits, _ = model(data)
            loss = criterion(logits, target)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    return total_loss / len(val_loader), accuracy, all_preds, all_targets

def loso_cross_validation(config, dataset_name='SEED'):
    """Perform Leave-One-Subject-Out cross-validation"""
    
    num_subjects = 15 if dataset_name == 'SEED' else 20
    num_classes = 3 if dataset_name == 'SEED' else 5
    
    subject_accuracies = []
    all_results = {}
    
    print(f"Starting LOSO cross-validation for {dataset_name} dataset")
    print(f"Number of subjects: {num_subjects}")
    print(f"Number of classes: {num_classes}")
    
    for subject_id in range(1, num_subjects + 1):
        print(f"\n{'='*50}")
        print(f"Training for subject {subject_id} (leaving out subject {subject_id})")
        print(f"{'='*50}")
        
        # Create data loaders for this fold
        train_loader, val_loader = create_loso_data_loaders(
            config.data_path, 
            dataset_name, 
            config.batch_size, 
            subject_id
        )
        
        if train_loader is None or val_loader is None:
            print(f"Skipping subject {subject_id} - no data available")
            continue
        
        # Create model
        model = EmoTCNAttn(
            num_channels=config.num_channels,
            num_classes=num_classes,
            kernel_sizes=config.kernel_sizes,
            dilation_rates=config.dilation_rates,
            num_tcn_blocks=config.num_tcn_blocks,
            num_attention_heads=config.num_attention_heads,
            dropout_rate=config.dropout_rate
        ).to(config.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Training loop
        best_accuracy = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        start_time = time.time()
        
        for epoch in range(config.num_epochs):
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device)
            
            # Validation
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, config.device)
            
            # Update learning rate
            scheduler.step()
            
            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch+1}/{config.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(config.checkpoint_dir, f'best_model_subject_{subject_id}.pth')
                save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        training_time = time.time() - start_time
        
        # Load best model and evaluate
        best_model_path = os.path.join(config.checkpoint_dir, f'best_model_subject_{subject_id}.pth')
        if os.path.exists(best_model_path):
            load_checkpoint(model, optimizer, best_model_path)
        
        # Final evaluation
        final_loss, final_acc, predictions, targets = validate(model, val_loader, criterion, config.device)
        
        # Store results
        subject_results = {
            'subject_id': subject_id,
            'best_accuracy': best_accuracy,
            'final_accuracy': final_acc,
            'training_time': training_time,
            'predictions': predictions,
            'targets': targets
        }
        
        all_results[f'subject_{subject_id}'] = subject_results
        subject_accuracies.append(final_acc)
        
        print(f"\nSubject {subject_id} Results:")
        print(f"  Best Accuracy: {best_accuracy:.2f}%")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Training Time: {training_time:.2f} seconds")
        
        # Plot training curves
        if len(train_losses) > 0:
            plot_path = os.path.join(config.figure_dir, f'training_curves_subject_{subject_id}.png')
            plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Calculate overall statistics
    mean_accuracy = np.mean(subject_accuracies)
    std_accuracy = np.std(subject_accuracies)
    
    print(f"\n{'='*50}")
    print(f"LOSO Cross-Validation Results for {dataset_name}")
    print(f"{'='*50}")
    print(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    print(f"Individual Subject Accuracies:")
    for i, acc in enumerate(subject_accuracies, 1):
        print(f"  Subject {i}: {acc:.2f}%")
    
    # Save overall results
    results_summary = {
        'dataset': dataset_name,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'subject_accuracies': subject_accuracies,
        'detailed_results': all_results
    }
    
    results_path = os.path.join(config.log_dir, f'loso_results_{dataset_name.lower()}.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return mean_accuracy, std_accuracy, all_results

def main():
    parser = argparse.ArgumentParser(description='Train EmoTCN-Attn model with LOSO cross-validation')
    parser.add_argument('--dataset', type=str, default='SEED', choices=['SEED', 'SEED-V'], 
                       help='Dataset to use for training')
    parser.add_argument('--data_path', type=str, default='./data/', 
                       help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    print(f"Using device: {config.device}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    # Perform LOSO cross-validation
    mean_acc, std_acc, results = loso_cross_validation(config, args.dataset)
    
    print(f"\nFinal Results:")
    print(f"Dataset: {args.dataset}")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # Save final summary
    final_summary = {
        'dataset': args.dataset,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
                    'model_parameters': count_parameters(EmoTCNAttn()),
        'training_config': vars(args)
    }
    
    summary_path = os.path.join(config.log_dir, f'final_summary_{args.dataset.lower()}.json')
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)

if __name__ == '__main__':
    main() 