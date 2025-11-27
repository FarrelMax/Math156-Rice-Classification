# Math156-Rice-Classification

A deep learning project classifying rice grain images into five varieties using a custom Convolutional Neural Network (CNN) with hyperparameter optimization.

## Dataset

Rice Image Dataset: 75,000 images across 5 rice varieties (Arborio, Basmati, Ipsala, Jasmine, Karacadag)  
Download from [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)

- Training: 60%  
- Validation: 20%  
- Testing: 20%  
- Original resolution: 250×250 pixels  

> **Note:** Update dataset path in the code:

```python
ROOT = '/path/to/your/Rice_Image_Dataset'
```

## Installation

Clone the repository:

```bash
git clone https://github.com/FarrelMax/Math156-Rice-Classification.git
cd Math156-Rice-Classification
```

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas seaborn tqdm optuna torchsummary
```

> For Google Colab users, mount your Drive where the dataset is stored.

## Model Architecture

Custom CNN (~54K trainable parameters):

1. Conv2D → ReLU → MaxPool
2. Conv2D → ReLU → MaxPool
3. AdaptiveAvgPool → Flatten
4. Fully connected layer → 5-class output

- Supports tunable hidden size and learning rate.

## Key Features

- Data augmentation: random flips, rotation, color jitter
- Dataset-specific normalization
- Hyperparameter tuning with Optuna
- Evaluation: confusion matrix, classification report, PCA, and K-Means clustering


## Inference

To use a trained model for inference:

```python
import torch
from rice_classification import CNN  # or the appropriate module

# Set device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load trained model
model = CNN(num_classes=5, hidden_size=32)
model.load_state_dict(torch.load('rice_cnn_best_model.pth', map_location=device))
model.to(device)
model.eval()

## Results

| Metric                   | Value             |
| ------------------------ | ----------------- |
| Test Accuracy            | 98.8%             |
| Best Validation Accuracy | 99.10% (epoch 14) |

Seed = 42 for reproducibility.

## Key Hyperparameters

```python
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
NUM_TRIALS = 20  # Optuna
```

Tuned parameters:

* Learning rate: 1e-4 → 1e-2 (log scale)
* Hidden layer size: 16, 32, 64
* Weight decay: 1e-6 → 1e-3 (log scale)


## Citations

### Dataset
Koklu, M. (2021). *Rice Image Dataset*. Kaggle.

### Core Methods & Theory
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

### Libraries & Frameworks
- PyTorch Documentation — Data utilities: `torch.utils.data`
- GeeksforGeeks tutorial: "Building a Convolutional Neural Network Using PyTorch"
- GeeksforGeeks tutorial: "Hyperparameter Tuning with Optuna in PyTorch"

### Implementation Resources
- GeeksforGeeks: "How Do You Use PyTorch's Dataset and DataLoader Classes for Custom Data?"
- Hosseini, G. (2023). "Beginner Tutorial: Image Classification Using PyTorch." *Medium*.
- Rath, S. R. (2021). "PyTorch ImageFolder for Training CNN Models." *DebuggerCafe*.

---

## Authors

Sophia Palomares  
Farrel Gomargana  
Diana Tolu  
MATH 156 – UCLA  
November 26, 2025

---

For complete methodology and analysis, refer to the project report.

