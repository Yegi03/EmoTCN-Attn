import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json
from scipy import stats

def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['accuracy']

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Val Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_kernel_importance(kernel_sizes, importance_scores, save_path):
    """Plot kernel size importance analysis"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(kernel_sizes, importance_scores, color='skyblue', edgecolor='navy')
    
    # Highlight the most important kernels
    max_importance = max(importance_scores)
    for i, (kernel, importance) in enumerate(zip(kernel_sizes, importance_scores)):
        if importance == max_importance:
            bars[i].set_color('red')
    
    plt.title('Kernel Size Importance in Multi-Scale TCN')
    plt.xlabel('Kernel Size')
    plt.ylabel('Relative Importance')
    plt.xticks(kernel_sizes)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(importance_scores):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_component_importance(components, importance_scores, save_path):
    """Plot component ablation analysis"""
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.1, 0.05, 0.05, 0.05)  # Explode the most important component
    
    wedges, texts, autotexts = plt.pie(importance_scores, explode=explode, labels=components,
                                       autopct='%1.1f%%', startangle=90, colors=colors)
    
    plt.title('Relative Component Impact on Model Performance')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_frequency_band_comparison(band_names, accuracies, save_path):
    """Plot frequency band combination performance"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(band_names, accuracies, color='lightcoral', edgecolor='darkred')
    
    # Highlight the best performing combination
    max_acc = max(accuracies)
    for i, (band, acc) in enumerate(zip(band_names, accuracies)):
        if acc == max_acc:
            bars[i].set_color('gold')
            bars[i].set_edgecolor('orange')
    
    plt.title('Frequency Band Combination Performance')
    plt.xlabel('Frequency Band Combination')
    plt.ylabel('Accuracy (%)')
    plt.ylim(85, 95)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_processing_comparison(strategies, accuracies, save_path):
    """Plot temporal processing strategy comparison"""
    plt.figure(figsize=(8, 6))
    bars = plt.bar(strategies, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'])
    
    # Highlight the best strategy
    max_acc = max(accuracies)
    for i, (strategy, acc) in enumerate(zip(strategies, accuracies)):
        if acc == max_acc:
            bars[i].set_color('gold')
            bars[i].set_edgecolor('orange')
    
    plt.title('Temporal Processing Strategy Comparison')
    plt.xlabel('Processing Strategy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(85, 95)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model and generate reports"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze().to(device)
            logits, _ = model(data)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1, keepdim=True)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names)
    
    return accuracy, report, all_preds, all_targets, all_probs

def statistical_significance_test(accuracies_1, accuracies_2, alpha=0.05):
    """Perform paired t-test for statistical significance"""
    t_stat, p_value = stats.ttest_rel(accuracies_1, accuracies_2)
    
    is_significant = p_value < alpha
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'alpha': alpha
    }

def compute_attention_weights(model, data_loader, device):
    """Compute attention weights for interpretability analysis"""
    model.eval()
    attention_weights_list = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _, attention_weights = model(data)
            attention_weights_list.append(attention_weights.cpu().numpy())
    
    return np.concatenate(attention_weights_list, axis=0)

def plot_attention_heatmap(attention_weights, channel_names, save_path):
    """Plot attention weights heatmap"""
    # Average attention weights across samples and heads
    avg_attention = np.mean(attention_weights, axis=(0, 1))  # (num_channels,)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(avg_attention.reshape(1, -1), cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('EEG Channel Attention Weights')
    plt.xlabel('EEG Channels')
    plt.ylabel('Attention')
    plt.xticks(range(len(channel_names)), channel_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_directories():
    """Create necessary directories"""
    dirs = ['checkpoints', 'figures', 'logs', 'results']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def save_results(results_dict, filename):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)

def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_computational_efficiency(model, data_loader, device, num_runs=100):
    """Calculate computational efficiency metrics"""
    model.eval()
    
    # Warm up
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= 5:  break
            data = data.to(device)
            _ = model(data)
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_runs: break
            data = data.to(device)
            
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if device.type == 'cuda':
                start_time.record()
                _ = model(data)
                end_time.record()
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time))
            else:
                import time
                start = time.time()
                _ = model(data)
                end = time.time()
                times.append((end - start) * 1000)  # Convert to milliseconds
    
    avg_inference_time = np.mean(times)
    std_inference_time = np.std(times)
    
    return {
        'avg_inference_time_ms': avg_inference_time,
        'std_inference_time_ms': std_inference_time,
        'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test directory creation
    create_directories()
    print("Directories created successfully")
    
    # Test plotting functions with dummy data
    kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
    importance_scores = [0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02]
    
    plot_kernel_importance(kernel_sizes, importance_scores, 'test_kernel_importance.png')
    print("Kernel importance plot created")
    
    # Test component importance
    components = ['Multi-Head Attention', 'Number of TCN Blocks', 'Kernel Scale Diversity', 'Dilation Rates']
    importance_scores = [38.3, 30.2, 16.3, 15.2]
    
    plot_component_importance(components, importance_scores, 'test_component_importance.png')
    print("Component importance plot created") 