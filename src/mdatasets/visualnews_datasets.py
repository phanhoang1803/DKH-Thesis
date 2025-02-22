import json
from typing import Optional, Dict, Any
from PIL import Image
from torch.utils.data import Dataset
from argparse import ArgumentParser


class VisualNewsDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Any] = None):
        """
        Args:
            data_path (str): Path to the JSON file containing dataset information.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.data = self._load_data(data_path)
        self.transform = transform

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
        item = self.data[idx]

        if not all(key in item for key in ("image_path", "caption", "article_path")):
            raise ValueError(f"Missing required fields in data item at index {idx}: {item}")

        # Load image
        try:
            image = Image.open(item["image_path"])
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at: {item['image_path']}")

        if self.transform:
            image = self.transform(image)

        # Load content
        try:
            with open(item["article_path"], 'r') as f:
                content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Article file not found at: {item['article_path']}")

        return {
            "image": image,
            "caption": item["caption"],
            "content": content
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/visualnews_dataset.json",
        help="Path to the dataset JSON file."
    )
    args = parser.parse_args()

    try:
        dataset = VisualNewsDataset(data_path=args.data_path)
        print(f"Dataset loaded successfully with {len(dataset)} entries.")
        print(dataset[0])
    except Exception as e:
        print(f"Error loading dataset: {e}")
