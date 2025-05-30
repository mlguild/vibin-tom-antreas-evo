# Dataset Preparation: EMNIST

This document describes the EMNIST dataset preparation process implemented in `tinkering/emnist_dataset.py`.

## 1. Overview

The `get_emnist_dataloaders` function handles the downloading, processing, and augmentation of the EMNIST dataset, providing `torch.utils.data.DataLoader` instances for training and testing.

## 2. Key Functionality

Located in `tinkering.emnist_dataset.py`:

```python
def get_emnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    emnist_split: str = "byclass",
) -> tuple:
    # ... implementation ...
```

### Parameters:
*   `data_dir (str)`: Root directory to store/load the EMNIST data (e.g., `./emnist_data/byclass`).
*   `batch_size (int)`: Batch size for the DataLoaders.
*   `num_workers (int)`: Number of worker processes for parallel data loading.
*   `emnist_split (str)`: Specifies which EMNIST split to use. Options include:
    *   `'byclass'`: 62 classes (0-9, A-Z, a-z).
    *   `'letters'`: 26 classes (case-insensitive A-Z, mapped to 1-26).
    *   `'digits'`: 10 classes (0-9).
    *   `'mnist'`: 10 classes (0-9, MNIST-compatible format).
    *   `'balanced'`: 47 balanced classes.
    *   `'bymerge'`: 47 classes, merged by visual similarity.

### Returns:
*   `train_loader (DataLoader)`: DataLoader for the training set.
*   `test_loader (DataLoader)`: DataLoader for the test set.
*   `num_classes (int)`: The number of classes in the chosen split.
*   `class_to_idx (dict)`: A mapping from class name/index to integer label (may be a placeholder like `{i:i}` if not directly available from `torchvision`).

## 3. Data Augmentation and Preprocessing

### 3.1. Training Data (`transform_train`):
1.  **`transforms.RandomAffine`**: Applies random affine transformations:
    *   `degrees=10`: Rotation by +/- 10 degrees.
    *   `translate=(0.1, 0.1)`: Horizontal and vertical translation by up to 10% of image size.
    *   `scale=(0.9, 1.1)`: Scaling between 90% and 110%.
    *   `shear=10`: Shear by +/- 10 degrees.
2.  **`transforms.RandAugment`**: Applies RandAugment, a scheme for automated data augmentation.
    *   `num_ops=2`: Number of augmentation operations to apply sequentially.
    *   `magnitude=9`: Magnitude for all transformations (range typically 0-30).
3.  **`transforms.ToTensor()`**: Converts PIL Image or NumPy `ndarray` to a PyTorch tensor and scales pixel values from `[0, 255]` to `[0.0, 1.0]`.
4.  **`transforms.Normalize(mean, std)`**: Normalizes the tensor image with specified mean and standard deviation.
    *   Currently uses MNIST-like values: `mean=(0.1307,)`, `std=(0.3081,)`. These are placeholders and ideally should be calculated per EMNIST split for optimal performance.

### 3.2. Test/Validation Data (`transform_test`):
1.  **`transforms.ToTensor()`**
2.  **`transforms.Normalize(mean, std)`**
    *   No aggressive augmentation is applied to the test set to ensure consistent evaluation.

## 4. Usage Example

From `run_gradient_based.py`:

```python
from tinkering.emnist_dataset import get_emnist_dataloaders

data_dir = Path("./emnist_data") / "byclass"
train_loader, test_loader, num_classes, _ = get_emnist_dataloaders(
    data_dir=str(data_dir),
    batch_size=64,
    emnist_split='byclass',
    num_workers=4
)
```

## 5. Output and Logging

The script `tinkering/emnist_dataset.py`, when run directly (`python tinkering/emnist_dataset.py`), will:
*   Print information about the data directory, transforms, chosen split, dataset sizes, and number of classes using `rich`.
*   Attempt to fetch and display information about a sample batch from the training loader for the 'byclass' and 'letters' splits. 