"""
src/dataset.py
──────────────
Loads the PlantVillage dataset from HuggingFace Hub and wraps it in a
PyTorch-compatible Dataset with the appropriate transforms.

Usage:
    from src.dataset import load_plantvillage
    train_ds, test_ds, class_names = load_plantvillage(config="color")
"""

from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics used for pre-trained model normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Return torchvision transforms for the given split (train / test)."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class PlantVillageDataset(Dataset):
    """
    Wraps a HuggingFace PlantVillage split as a PyTorch Dataset.

    Each item returns (image_tensor, label_int).
    The raw HuggingFace sample also contains: leaf_id, crop, disease.
    """

    def __init__(self, hf_split, transform=None):
        self.data      = hf_split
        self.transform = transform
        self.classes   = hf_split.features["label"].names
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image  = sample["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, sample["label"]

    def get_raw(self, idx):
        """Return the full raw HuggingFace sample (PIL image + metadata)."""
        return self.data[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Public loader
# ──────────────────────────────────────────────────────────────────────────────

def load_plantvillage(config: str = "color", num_proc: int = 2):
    """
    Download (or use cached) PlantVillage from HuggingFace Hub and return
    ready-to-use PyTorch datasets.

    Args:
        config:   One of 'color', 'grayscale', 'segmented'.
        num_proc: Parallelism for dataset preparation.

    Returns:
        (train_dataset, test_dataset, class_names)
        where class_names is a list of 38 strings like 'Apple___Black_rot'.
    """
    from datasets import load_dataset  # lazy import — avoids startup cost

    print(f"[dataset] Loading PlantVillage ({config}) from HuggingFace Hub …")
    hf = load_dataset("mohanty/PlantVillage", config, num_proc=num_proc)

    train_ds = PlantVillageDataset(hf["train"], transform=get_transforms("train"))
    test_ds  = PlantVillageDataset(hf["test"],  transform=get_transforms("test"))

    print(f"[dataset]   train: {len(train_ds):,} samples")
    print(f"[dataset]   test : {len(test_ds):,} samples")
    print(f"[dataset]   classes ({train_ds.num_classes}): "
          f"{train_ds.classes[:3]} … {train_ds.classes[-1]}")

    return train_ds, test_ds, train_ds.classes


def load_raw_split(config: str = "color", split: str = "train"):
    """
    Return a raw HuggingFace DatasetDict split (without PyTorch wrapping).
    Useful for EDA notebooks that need direct access to PIL images + metadata.
    """
    from datasets import load_dataset
    hf = load_dataset("mohanty/PlantVillage", config)
    return hf[split]
