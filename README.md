# CN7023-Brain-Tumor-Classification

Deep Learning Based Custom Image Multi-Class Classifier for Brain Tumor MRI Classification - CN7023 Artificial Intelligence & Machine Vision Assignment

## Overview

This project implements a Convolutional Neural Network (CNN) for classifying brain tumor MRI images into 4 categories: glioma, meningioma, no_tumor, and pituitary.

## Project Structure

```
CN7023-Brain-Tumor-Classification/
├── src/
│   ├── model.py          # BrainTumorCNN model definition
│   ├── dataset.py        # Data loading and preprocessing
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── data/
│   ├── train/            # Training data
│   └── test/             # Test data
├── models/               # Saved model checkpoints
├── results/              # Evaluation results and plots
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

Organize your data in the following structure:

```
data/
├── train/
│   ├── glioma/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

## Usage

### Training

Train the model with default parameters:
```bash
cd src
python train.py
```

With custom parameters:
```bash
python train.py --data_dir ../data --epochs 50 --batch_size 32 --lr 0.001
```

### Evaluation

Evaluate the trained model:
```bash
cd src
python evaluate.py
```

With custom parameters:
```bash
python evaluate.py --model_path models/best_model.pth --data_dir ../data
```

## Model Architecture

The `BrainTumorCNN` class implements a convolutional neural network with:
- 4 convolutional blocks with batch normalization and max pooling
- Adaptive pooling layer
- 3 fully connected layers with dropout
- Designed for 128x128 RGB input images
- 4-class output for tumor classification

## Features

- Automatic train/validation split (default 80/20)
- Data augmentation for training
- Model checkpointing with best model saving
- Comprehensive evaluation with classification report and confusion matrix
- GPU support when available
- Detailed logging and progress tracking

## Results

After training, evaluation results are saved in `results/`:
- `confusion_matrix.png` - Confusion matrix visualization
- `class_distribution.png` - Test set class distribution
- `detailed_results.txt` - Detailed performance metrics
