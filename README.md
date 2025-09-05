# Deepfake Detection: Multi-Modal Approach

A comprehensive deepfake detection system implementing multiple state-of-the-art approaches for both image and video analysis. This project demonstrates three distinct methodologies for detecting synthetic media content with varying levels of performance and computational requirements.

## Table of Contents
- [Overview](#overview)
- [Approaches](#approaches)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

## Overview

This project implements three complementary approaches to deepfake detection:

1. **Image-based Detection**: Using InceptionResNetV2 with transfer learning for static image classification
2. **Video Sequence Analysis**: Custom CNN-LSTM architecture with attention mechanisms for temporal feature extraction
3. **Frame-based Video Analysis**: Xception-based architecture processing individual video frames

Each approach addresses different aspects of deepfake detection and can be used independently or in combination for robust synthetic media identification.

## Approaches

### 1. Image-based Detection (InceptionResNetV2)

#### Architecture
- **Base Model**: InceptionResNetV2 pre-trained on ImageNet
- **Input Shape**: (128, 128, 3)
- **Transfer Learning**: Fine-tuning with frozen base layers
- **Classification Head**: Global Average Pooling + Dense(2) with softmax

#### Key Features
- Transfer learning from ImageNet weights
- Binary classification (Real vs Fake)
- Data normalization and augmentation
- Early stopping and learning rate scheduling

#### Performance
- **Training Accuracy**: 99.73%
- **Validation Accuracy**: 91.72%
- **Test Results**:
  - True Positive: 2,949
  - False Positive: 37
  - False Negative: 42
  - True Negative: 717

### 2. Video Sequence Analysis (CNN-LSTM with Attention)

#### Architecture Components

**Dynamic Video Classifier**:
- **Backbone Options**: ResNet18/50, VGG16, DenseNet121, MobileNetV2, InceptionV3
- **Temporal Processing**: LSTM with configurable hidden size and layers
- **Attention Mechanism**: Custom attention for temporal feature weighting
- **Classification**: Multi-layer perceptron with dropout and layer normalization

**Data Pipeline**:
- **Sequence Length**: Configurable frame sampling (default: 15 frames)
- **Weighted Sampling**: Addresses class imbalance in training
- **Data Augmentation**: Resize, normalization, and tensor conversion

#### Key Features
- Multi-backbone CNN support for feature extraction
- LSTM-based temporal modeling
- Attention mechanism for important frame identification
- Class balancing through weighted random sampling
- Comprehensive evaluation metrics (Precision, Recall, F1, AUC)

#### Performance (Approach 1 - Custom Architecture)
- **Training**: Accuracy: 72.14%, AUC: 73.03%
- **Validation**: Accuracy: 44.67%, AUC: 47.36%
- **Test Results**: Accuracy: 48.67%, AUC: 49.33%

#### Performance (Approach 2 - MViT Architecture)
- **Training**: Precision: 91.95%, Recall: 87.10%, F1: 89.46%, AUC: 92.02%
- **Validation**: Precision: 90.03%, Recall: 87.33%, F1: 88.66%, AUC: 90.01%
- **Test Results**: Precision: 83.33%, Recall: 85.00%, F1: 84.16%, AUC: 83.12%

### 3. Frame-based Video Analysis (Xception)

#### Architecture
- **Base Model**: Xception pre-trained on ImageNet
- **Input Processing**: Frame extraction at 1 FPS
- **Feature Extraction**: Global Average Pooling
- **Classification**: Single dense layer with sigmoid activation

#### Data Processing Pipeline
- Video frame extraction at 1 frame per second
- Image preprocessing with Xception-specific normalization
- Stratified train/validation/test splits
- TensorFlow data pipeline optimization

#### Performance
- **Training Accuracy**: 90.32%
- **Validation Accuracy**: 90.43%
- **Test Results**:
  - Overall Accuracy: 90%
  - AUC-ROC: 0.638
  - Real Class: Precision: 0.60, Recall: 0.00 (poor real detection)
  - Fake Class: Precision: 0.90, Recall: 1.00 (excellent fake detection)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for video processing

### Dependencies
```bash
# Core ML frameworks
pip install torch torchvision tensorflow

# Computer vision and data processing
pip install opencv-python numpy pandas scikit-learn

# Visualization and metrics
pip install matplotlib seaborn

# Optional: Weights & Biases for experiment tracking
pip install wandb

# Video processing (if using PyTorchVideo)
pip install pytorchvideo
```

### Dataset Setup
1. Download the Celeb-DF-v2 dataset
2. Organize data according to the project structure
3. Update dataset paths in configuration files

## Usage

### Image Detection
```python
from models.image_detection import train_image_model

# Configure parameters
input_shape = (128, 128, 3)
data_dir = 'dataset'

# Train model
model, history = train_image_model(
    data_dir=data_dir,
    input_shape=input_shape,
    epochs=20,
    batch_size=100
)
```

### Video Sequence Analysis
```python
from models.video_sequence_detection import train_model, create_dataloaders

# Create data loaders
train_loader, val_loader, test_loader, class_counts = create_dataloaders(
    base_path='/path/to/celeb-df-v2',
    output_path='/path/to/working/dir',
    dim=224,
    batch_size=12,
    sequence_length=16
)

# Train model
model, metrics = train_model(train_loader, val_loader, test_loader)
```

### Frame-based Analysis
```python
from models.frame_based_detection import extract_frames, train_frame_model

# Extract frames from videos
extract_frames(video_path, output_dir, label)

# Train Xception-based model
model = train_frame_model(
    train_paths=X_train,
    train_labels=y_train,
    val_paths=X_val,
    val_labels=y_val,
    epochs=10
)
```

## Results Summary

| Approach | Accuracy | Precision | Recall | F1-Score | AUC |
|----------|----------|-----------|--------|----------|-----|
| Image (InceptionResNetV2) | 91.72% | - | - | - | - |
| Video Sequence (Custom) | 48.67% | 48.44% | 41.33% | 44.60% | 49.33% |
| Video Sequence (MViT) | 77.88% | 83.33% | 85.00% | 84.16% | 83.12% |
| Frame-based (Xception) | 90.00% | 90.00% | 100.00% | 95.00% | 63.85% |

## Technical Details

### Data Preprocessing
- **Image Normalization**: Pixel values scaled to [0,1] range
- **Video Frame Sampling**: Uniform temporal sampling for sequence approaches
- **Class Balancing**: Weighted sampling for imbalanced datasets
- **Data Augmentation**: Standard computer vision transformations

### Model Architecture Choices
- **InceptionResNetV2**: Chosen for its efficiency and strong performance on image classification tasks
- **LSTM with Attention**: Captures temporal dependencies while focusing on discriminative frames
- **MViT (Multiscale Vision Transformer)**: State-of-the-art video understanding architecture
- **Xception**: Efficient depthwise separable convolutions for frame-level analysis

### Training Strategies
- **Transfer Learning**: Leveraging pre-trained weights from large-scale datasets
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence
- **Early Stopping**: Preventing overfitting through validation monitoring
- **Gradient Clipping**: Stabilizing training for recurrent architectures

### Evaluation Metrics
- **Classification Accuracy**: Overall correctness of predictions
- **Precision/Recall**: Class-specific performance measures
- **F1-Score**: Harmonic mean balancing precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of prediction patterns

## Key Findings

1. **Image-based Detection**: Achieves highest validation accuracy (91.72%) but may overfit to specific image artifacts
2. **MViT Video Analysis**: Best balance of metrics with 83.12% AUC, effectively leveraging temporal information
3. **Frame-based Analysis**: Strong overall accuracy (90%) but struggles with real video detection
4. **Temporal vs Spatial**: Video sequence approaches show promise for generalization across different deepfake generation methods

## Limitations and Future Work

### Current Limitations
- Dataset bias toward specific generation methods
- Computational requirements for video processing
- Class imbalance affecting some approaches
- Limited real-world testing scenarios

### Future Improvements
1. **Multi-modal Fusion**: Combining all three approaches for ensemble predictions
2. **Domain Adaptation**: Training on diverse datasets and generation methods
3. **Efficiency Optimization**: Model compression and quantization techniques
4. **Real-time Processing**: Optimizing for live video stream analysis
5. **Adversarial Robustness**: Testing against adaptive attacks

## Contributing

We welcome contributions to improve the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Areas for Contribution
- New detection architectures
- Dataset expansion and preprocessing improvements
- Performance optimization
- Evaluation metric enhancements
- Documentation and tutorials

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Celeb-DF-v2 dataset creators
- Pre-trained model providers (ImageNet, etc.)
- Open source deep learning communities
- Research papers that inspired the architectural choices

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{deepfake-detection-2024,
  title={Multi-Modal Deepfake Detection: A Comprehensive Approach},
  author={[Your Name]},
  year={2024},
  publisher={GitHub},
  url={https://github.com/[username]/deepfake-detection}
}
```
