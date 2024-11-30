# dataloaders/visualnews_dataloader.py

from torch.utils.data import DataLoader
from datasets.visualnews_datasets import VisualNewsDataset
from typing import Optional, Any
from torchvision import transforms

def get_visualnews_dataloader(
    data_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    transform: Optional[Any] = None
):
    dataset = VisualNewsDataset(data_path=data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
