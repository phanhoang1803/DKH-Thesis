import json
from typing import Optional, Dict, Any
from datetime import datetime
from torch.utils.data import Dataset
from argparse import ArgumentParser
from PIL import Image


class NewsClippingDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Any] = None):
        """
        Args:
            data_path (str): Path to the JSON file containing dataset information.
        """
        self.data = self._load_data(data_path)
        self.transform = transform
        self.ids = list(self.data.keys())

    @staticmethod
    def _load_data(data_path: str):
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {data_path}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item_id = self.ids[idx]
        item = self.data[item_id]

        required_fields = ("image_path", "caption", "article_path", "title", "topic", "source", "timestamp")
        if not all(key in item for key in required_fields):
            raise ValueError(f"Missing required fields in data item at index {idx}: {item}")

        # Parse timestamp
        try:
            timestamp = datetime.strptime(item["timestamp"], '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            raise ValueError(f"Invalid timestamp format in item {idx}")

        # Load image
        try:
            image = Image.open(item["image_path"])
        except FileNotFoundError:
            # raise FileNotFoundError(f"Image file not found at: {item['image_path']}")
            image = None

        if self.transform:
            image = self.transform(image)

        # Load article content
        try:
            with open(item["article_path"], 'r') as f:
                content = f.read()
        except FileNotFoundError:
            # raise FileNotFoundError(f"Article file not found at: {item['article_path']}")
            content = None

        return {
            "id": int(item_id),
            "title": item["title"],
            "caption": item["caption"],
            "content": content,
            "topic": item["topic"],
            "source": item["source"],
            "timestamp": timestamp,
            "image_path": item["image_path"],
            "image": image
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="news_dataset.json",
        help="Path to the dataset JSON file."
    )
    args = parser.parse_args()

    try:
        dataset = NewsClippingDataset(data_path=args.data_path)
        print(f"Dataset loaded successfully with {len(dataset)} entries.")
        sample = dataset[0]
        print("\nSample item:")
        print(f"Title: {sample['title']}")
        print(f"Source: {sample['source']}")
        print(f"Topic: {sample['topic']}")
        print(f"Timestamp: {sample['timestamp']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")