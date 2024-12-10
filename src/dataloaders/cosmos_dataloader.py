# dataloaders/cosmos_dataloader.py

from torch.utils.data import DataLoader
from ..datasets.cosmos_datasets import CosmosDataset
from typing import Optional, Any
from torchvision import transforms
from typing import List

def identity_collate(batch: List[Any]) -> List[Any]:
    """
    Simple collate function that returns the batch as-is
    """
    return batch

def get_cosmos_dataloader(
    data_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    transform: Optional[Any] = None
):
    dataset = CosmosDataset(data_path=data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=identity_collate)

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
