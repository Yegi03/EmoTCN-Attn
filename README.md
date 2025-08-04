# EmoTCN-Attn: Multi-Scale Temporal Convolution and Attention Framework for EEG Emotion Recognition

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of our paper **"EmoTCN-Attn: A Multi-Scale Temporal Convolution and Attention Framework for EEG Emotion Recognition"** published at IEEE Conference.

## Overview

We propose a novel deep learning framework that combines **Multi-Scale Temporal Convolutional Networks (TCNs)** with **multi-head self-attention mechanisms** for EEG-based emotion recognition. Our model achieves state-of-the-art performance on SEED and SEED-V datasets using Leave-One-Subject-Out (LOSO) cross-validation.

## Results

| Dataset | Accuracy (%) | Model |
|---------|-------------|-------|
| SEED    | **94.27**   | Proposed Model |
| SEED-V  | **91.23**   | Proposed Model |

## Architecture

- **Multi-Scale TCN**: 16 parallel streams with kernel sizes {3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33}
- **Attention Mechanism**: 8-head self-attention for adaptive EEG channel weighting
- **Dilation Rates**: {1,2,4} for multi-scale temporal modeling
- **Parameters**: ~2.3M parameters, 15ms inference latency

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/emotcn-attn.git
cd emotcn-attn
pip install -r requirements.txt
```

### Training

```bash
# Train on SEED dataset
python src/train.py --dataset SEED --epochs 100 --batch_size 32

# Train on SEED-V dataset
python src/train.py --dataset SEED-V --epochs 100 --batch_size 32
```

### Evaluation

```bash
# Evaluate trained model
python src/evaluate.py --model_path checkpoints/best_model.pth --dataset SEED
```

### Ablation Studies

```bash
# Run comprehensive ablation studies
python src/ablation_study.py --dataset SEED --study all

# Run specific ablation study
python src/ablation_study.py --dataset SEED --study attention
```

## Project Structure

```
emotcn-attn/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py              # Main EmoTCN-Attn model architecture
│   ├── data_loader.py        # EEG data loading and preprocessing
│   ├── train.py              # Training script with LOSO cross-validation
│   ├── evaluate.py           # Model evaluation script
│   ├── ablation_study.py     # Ablation studies
│   ├── generate_figures.py   # Figure generation for paper
│   ├── generate_results_csv.py # CSV result generation
│   └── utils.py              # Utility functions
├── results/                  # Training results, figures, and CSV data
└── checkpoints/              # Model checkpoints
```

## Key Features

- **Multi-Scale Temporal Modeling**: Captures EEG dynamics across multiple temporal scales
- **Attention-Driven Spatial Weighting**: Adaptively emphasizes relevant EEG channels
- **Subject-Independent Evaluation**: LOSO cross-validation for robust generalization
- **Real-Time Capability**: 15ms inference latency suitable for real-time applications
- **Comprehensive Analysis**: Ablation studies and interpretability analysis

## Methodology

### Preprocessing Pipeline
1. **Band-pass filtering** (1-50 Hz) using zero-phase Butterworth filter
2. **Independent Component Analysis (ICA)** for artifact removal
3. **Channel-wise z-score normalization**
4. **Data augmentation**: Temporal jittering (±250ms) and Gaussian noise

### Model Architecture
1. **Multi-Scale TCN**: 16 parallel streams with varying kernel sizes
2. **Dilated Convolutions**: Dilation rates {1,2,4} for multi-scale temporal modeling
3. **Global Average Pooling**: Preserves temporal context efficiently
4. **Multi-Head Attention**: 8-head self-attention for channel weighting
5. **Classification Head**: MLP with dropout for final classification

### Training Protocol
- **Optimizer**: AdamW with weight decay 1e-4
- **Scheduler**: OneCycleLR with cosine annealing
- **Early Stopping**: Patience of 15 epochs
- **Cross-Validation**: Leave-One-Subject-Out (LOSO)

## Ablation Studies

Our comprehensive ablation studies analyze:

1. **Multi-Head Attention Impact**: 38.3% contribution to performance
2. **Kernel Size Diversity**: Intermediate kernels (5,7,9) show highest importance
3. **TCN Block Depth**: 3 blocks optimal for temporal modeling
4. **Dilation Rate Selection**: {1,2,4} provides optimal temporal coverage
5. **Frequency Band Selection**: Proposed selection outperforms all bands (94.3% vs 93.1%)
6. **Temporal Processing**: Full temporal (0-4s) outperforms partial analysis

## Usage Examples

### Basic Training
```python
from src.model import EmoTCNAttn
from src.data_loader import create_loso_data_loaders

# Create model
model = EmoTCNAttn(num_classes=3)  # SEED dataset

# Create data loaders
train_loader, val_loader = create_loso_data_loaders('./data/', 'SEED')

# Train model
# (See train.py for complete training loop)
```

### Model Evaluation
```python
from src.utils import evaluate_model

# Evaluate model
accuracy, report, predictions, targets = evaluate_model(
    model, test_loader, device, class_names
)
print(f"Accuracy: {accuracy:.2f}%")
```

### Ablation Study
```python
from src.ablation_study import attention_ablation_study

# Run attention ablation study
results = attention_ablation_study(config, 'SEED')
print(f"Attention ablation results: {results}")
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{abdollahinejad2024emotcn,
  title={EmoTCN-Attn: A Multi-Scale Temporal Convolution and Attention Framework for EEG Emotion Recognition},
  author={Abdollahinejad, Yeganeh and Mousavi, Ahmad and Boukouvalas, Zois},
  booktitle={IEEE Conference},
  year={2024}
}
```

## Authors

- **Yeganeh Abdollahinejad** - Pennsylvania State University (yza5171@psu.edu)
- **Ahmad Mousavi** - American University (mousavi@american.edu)
- **Zois Boukouvalas** - American University (boukouva@american.edu)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/emotcn-attn.git
cd emotcn-attn

# Create virtual environment
python -m venv emotcn_env
source emotcn_env/bin/activate  # On Windows: emotcn_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Documentation

For detailed documentation, see the docstrings in each Python file or refer to our paper for theoretical details.

## Issues

If you encounter any issues, please open an issue on GitHub with a detailed description of the problem.

## Contact

For questions or collaborations, please contact:
- **Yeganeh Abdollahinejad**: yza5171@psu.edu 
