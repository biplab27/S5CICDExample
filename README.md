# MNIST Neural Network CI/CD Pipeline

![ML Pipeline](https://github.com/<your-username>/S5CICDExample/actions/workflows/ml-pipeline.yml/badge.svg)

This project demonstrates a complete CI/CD pipeline for a machine learning project using GitHub Actions. It includes a simple convolutional neural network (CNN) for MNIST digit classification with automated training, testing, and model validation.

## Project Structure

```
├── model/
│ └── mnist_model.py # Neural network architecture
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # GitHub Actions workflow
├── train.py # Training script
├── test_model.py # Testing and validation script
├── requirements.txt # Project dependencies
└── .gitignore # Git ignore rules
```

## Model Architecture

The model is a lightweight CNN with the following architecture:
- 2 Convolutional layers with max pooling
- 2 Fully connected layers
- Total parameters: ~26,746 (optimized for efficiency)

Architecture details:
- Input: 28x28 grayscale images
- Conv1: 8 filters (3x3)
- Conv2: 16 filters (3x3)
- FC1: 32 features
- Output: 10 classes (digits 0-9)

## Data Augmentation

The training pipeline includes comprehensive image augmentation techniques to improve model robustness:

### PIL-based Transforms (Pre-tensor):
- Random Rotation (±15 degrees)
- Random Affine Transformations:
  - Translation: up to 10% in any direction
  - Scale: 90% to 110% of original size
  - Shear: -10 to 10 degrees
- Random Perspective (20% distortion, 50% probability)

### Tensor-based Transforms (Post-tensor):
- Random Inversion (10% probability)
- Random Erasing (10% probability)
- Normalization (mean=0.1307, std=0.3081)

These augmentations help:
- Improve model generalization
- Reduce overfitting
- Increase robustness to variations in digit appearance
- Better handle real-world data variations

## CI/CD Pipeline

The pipeline automatically:
1. Trains the model on MNIST dataset
2. Validates model architecture
3. Tests model performance
4. Archives trained models

### Validation Checks
- Input shape compatibility (28x28)
- Output shape verification (10 classes)
- Parameter count (< 100,000)
- Model accuracy (> 80%)

## Setup and Running Locally

1. Create and activate virtual environment:
```
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Train the model:
```
python train.py
```
4. Run tests:
```
python -m pytest test_model.py -v
```
## GitHub Actions Integration

The pipeline automatically runs on every push to the repository:
- Uses CPU-only PyTorch for compatibility
- Runs on Ubuntu latest
- Stores trained models as artifacts
- Retention period: 90 days

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- torchvision
- pytest

## Model Artifacts

Trained models are saved with timestamps in the format:
```
mnist_model_YYYYMMDD_HHMMSS.pth
```

## Notes

- The model is trained for 1 epoch to demonstrate the pipeline
- Uses CPU-only PyTorch to ensure compatibility with GitHub Actions
- All computations are performed on CPU for consistency
- Extensive data augmentation is applied during training only

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.




