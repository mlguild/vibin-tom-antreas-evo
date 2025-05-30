# tom_antreas_evolution_vibe_coding

Project with .venv, requirements.txt, and src/main.py.

## Project Structure

*   `papers/`: Contains markdown notes and summaries of relevant research papers.
*   `tinkering/`:
    *   `emnist_dataset.py`: A script to download and prepare EMNIST datasets and DataLoaders using PyTorch and Torchvision. It includes data augmentation (RandAugment) for training.
*   `install-conda-python-deps.sh`: Shell script for setting up dependencies.
*   `pyproject.toml`: Project metadata and dependencies.
*   `README.md`: This file.

## Tinkering Module

The `tinkering` module houses experimental code and utilities. 

### `emnist_dataset.py`

This script provides a function `get_emnist_dataloaders` to easily load EMNIST data for different splits (e.g., 'byclass', 'letters'). It applies `RandAugment` and other standard transformations for training data.

To use it, you can import the function:

```python
from tinkering.emnist_dataset import get_emnist_dataloaders

# Example for 'byclass' split
train_loader, test_loader, num_classes, class_to_idx = get_emnist_dataloaders(
    data_dir="./emnist_data_byclass",
    batch_size=128,
    emnist_split='byclass'
)
```

Running the script directly (`python tinkering/emnist_dataset.py`) will also demonstrate its usage and download the data for the 'byclass' and 'letters' splits.
