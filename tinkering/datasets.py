import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from rich.console import Console
from rich.pretty import pprint
from rich.traceback import install as install_rich_traceback

console = Console()


# --- EMNIST ---
def get_emnist_dataloaders(
    data_dir: str = "./data_emnist",  # Changed default data_dir name
    batch_size: int = 64,
    num_workers: int = 4,
    emnist_split: str = "byclass",
):
    """
    Prepares EMNIST dataset and DataLoaders.
    """
    data_path = Path(data_dir).resolve()
    console.print(
        f"[info]EMNIST data directory ('{emnist_split}' split): {data_path}"
    )
    data_path.mkdir(parents=True, exist_ok=True)

    emnist_mean = (0.1307,)
    emnist_std = (0.3081,)

    transform_train_emnist = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(emnist_mean, emnist_std),
        ]
    )

    transform_test_emnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(emnist_mean, emnist_std)]
    )

    train_dataset = torchvision.datasets.EMNIST(
        root=str(data_path),
        split=emnist_split,
        train=True,
        download=True,
        transform=transform_train_emnist,
    )
    test_dataset = torchvision.datasets.EMNIST(
        root=str(data_path),
        split=emnist_split,
        train=False,
        download=True,
        transform=transform_test_emnist,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    try:
        class_to_idx = train_dataset.class_to_idx
        num_classes = len(class_to_idx)
    except AttributeError:
        if emnist_split == "byclass":
            num_classes = 62
        elif emnist_split == "letters":
            num_classes = 26
        elif emnist_split == "digits":
            num_classes = 10
        elif emnist_split == "mnist":
            num_classes = 10
        elif emnist_split == "balanced":
            num_classes = 47
        elif emnist_split == "bymerge":
            num_classes = 47
        else:
            raise ValueError(f"Unknown EMNIST split: {emnist_split}")
        class_to_idx = {i: i for i in range(num_classes)}

    console.print(
        f"[info]EMNIST ('{emnist_split}'): Train {len(train_dataset)}, Test {len(test_dataset)}, Classes {num_classes}"
    )
    return train_loader, test_loader, num_classes, class_to_idx


# --- CIFAR-100 ---
def get_cifar100_dataloaders(
    data_dir: str = "./data_cifar100",
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Prepares CIFAR-100 dataset and DataLoaders.
    """
    data_path = Path(data_dir).resolve()
    console.print(f"[info]CIFAR-100 data directory: {data_path}")
    data_path.mkdir(parents=True, exist_ok=True)

    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    transform_train_cifar = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ]
    )
    transform_test_cifar = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root=str(data_path),
        train=True,
        download=True,
        transform=transform_train_cifar,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=str(data_path),
        train=False,
        download=True,
        transform=transform_test_cifar,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_classes = 100
    class_to_idx = {i: i for i in range(num_classes)}

    console.print(
        f"[info]CIFAR-100: Train {len(train_dataset)}, Test {len(test_dataset)}, Classes {num_classes}"
    )
    return train_loader, test_loader, num_classes, class_to_idx


# --- Generic Loader ---
def get_dataloaders(dataset_name: str, **kwargs):
    """
    Generic function to get dataloaders for a specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'emnist', 'cifar100').
        **kwargs: Arguments specific to the dataset loader function.
                  For EMNIST, this includes `emnist_split`.
                  Common args: `data_dir`, `batch_size`, `num_workers`.
    Returns:
        tuple: (train_loader, test_loader, num_classes, class_to_idx)
    """
    if dataset_name.lower() == "emnist":
        console.print(
            f"[info]Preparing EMNIST dataloaders with args: {kwargs}"
        )
        return get_emnist_dataloaders(**kwargs)
    elif dataset_name.lower() == "cifar100":
        console.print(
            f"[info]Preparing CIFAR-100 dataloaders with args: {kwargs}"
        )
        # CIFAR-100 specific args are handled, emnist_split is not relevant
        cifar_kwargs = {k: v for k, v in kwargs.items() if k != "emnist_split"}
        return get_cifar100_dataloaders(**cifar_kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    install_rich_traceback()
    console.rule("[bold green]Testing Combined Dataset Loaders[/bold green]")

    console.rule("[bold blue]EMNIST byclass Test[/bold blue]")
    get_dataloaders(
        dataset_name="emnist",
        data_dir="./emnist_data_byclass",
        emnist_split="byclass",
        batch_size=32,
    )

    console.rule("[bold blue]CIFAR-100 Test[/bold blue]")
    get_dataloaders(
        dataset_name="cifar100", data_dir="./cifar100_data", batch_size=32
    )

    console.print(
        "\n[info]Attempting to fetch a batch from CIFAR-100 loader..."
    )
    try:
        c_train, _, _, _ = get_cifar100_dataloaders(
            data_dir="./cifar100_data_maintest", batch_size=4
        )
        images, labels = next(iter(c_train))
        console.print(
            f"[success]CIFAR-100 batch: images {images.shape}, labels {labels.shape}"
        )
    except Exception as e:
        console.print(f"[error]Error with CIFAR-100 batch: {e}")
        raise

    console.print(
        "\n[info]Attempting to fetch a batch from EMNIST (letters) loader..."
    )
    try:
        e_train, _, _, _ = get_emnist_dataloaders(
            data_dir="./emnist_data_letters_maintest",
            emnist_split="letters",
            batch_size=4,
        )
        images, labels = next(iter(e_train))
        console.print(
            f"[success]EMNIST batch: images {images.shape}, labels {labels.shape}"
        )
    except Exception as e:
        console.print(f"[error]Error with EMNIST batch: {e}")
        raise

    console.rule(
        "[bold green]Dataset Loader Test Script Finished[/bold green]"
    )
