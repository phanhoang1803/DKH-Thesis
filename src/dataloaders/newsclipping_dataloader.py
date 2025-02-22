# dataloaders/newsclipping_dataloader.py

from torch.utils.data import DataLoader
from mdatasets.newsclipping_datasets import NewsClippingDataset
from typing import Optional, Any, List


def identity_collate(batch: List[Any]) -> List[Any]:
    """
    Simple collate function that returns the batch as-is.
    Args:
        batch: List of data items from the dataset
    Returns:
        The unmodified batch
    """
    return batch


def get_newsclipping_dataloader(
    data_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2, 
    transform: Optional[Any] = None
):
    """
    Creates a DataLoader for the NewsClippingDataset.
    
    Args:
        data_path (str): Path to the JSON file containing dataset information
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of subprocesses for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader for the NewsClippingDataset
    """
    dataset = NewsClippingDataset(data_path=data_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=identity_collate,
        # pin_memory=True,
        # prefetch_factor=2
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/newsclipping_dataset.json",
        help="Path to the dataset JSON file."
    )
    args = parser.parse_args()
    
    try:
        dataloader = get_newsclipping_dataloader(
            data_path=args.data_path,
            batch_size=2,  # Small batch size for testing
            shuffle=True,
            num_workers=0  # Use 0 for easier debugging
        )
        
        print(f"DataLoader created successfully")
        print(f"Number of batches: {len(dataloader)}")
        
        # Test first batch
        first_batch = next(iter(dataloader))
        print("\nSample batch contents:")
        for item in first_batch:
            print(f"Title: {item['title']}")
            print(f"Source: {item['source']}")
            print(f"Topic: {item['topic']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error setting up dataloader: {e}")