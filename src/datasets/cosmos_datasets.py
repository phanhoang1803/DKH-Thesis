import json
from typing import Optional, Dict, Any
from PIL import Image
from torch.utils.data import Dataset
from argparse import ArgumentParser
from torchvision import transforms
import os
import base64

class CosmosDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Any] = None):
        """
        Dataset class for handling image-caption-content data.
        
        Args:
            data_path (str): Path to the JSON file containing dataset information.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.image_dir = os.path.dirname(data_path)
        self.data = self._load_data(data_path)
        # self.transform = transform if transform is not None else self.__get_default_transform()
        self.transform = transform

    def __get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    @staticmethod
    def _load_data(data_path: str):
        """
        Load and validate the dataset from JSON file.
        
        Args:
            data_path (str): Path to the JSON file.
            
        Returns:
            List of validated data entries.
            
        Raises:
            FileNotFoundError: If the data file is not found.
            json.JSONDecodeError: If the JSON format is invalid.
        """
        try:
            with open(data_path, 'r', encoding='utf-8' ) as f:
                data = json.load(f)
            
            # Validate required fields
            required_fields = {'img_local_path', 'caption1', 'article_url'}
            
            for idx, item in enumerate(data):
                missing_fields = required_fields - set(item.keys())
                if missing_fields:
                    raise ValueError(
                        f"Missing required fields {missing_fields} in data item at index {idx}"
                    )
            
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {data_path}")

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to get.
            
        Returns:
            Dict containing:
                - image: PIL Image or transformed image
                - caption: Caption string
                - content: Content text
        """
        item = self.data[idx]
        
        # Load and validate image
        try:
            image = Image.open(os.path.join(self.image_dir, item["img_local_path"])).convert('RGB').resize((224, 224))
            with open(os.path.join(self.image_dir, item["img_local_path"]), "rb") as image_file:
                # Read the file and encode it to Base64
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at: {item['img_local_path']}")
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "image_base64": image_base64,
            "caption": item["caption1"],
            "content": item["caption2"],
            "label": item["context_label"]
        }

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file"
    )
    args = parser.parse_args()

    try:
        dataset = CosmosDataset(data_path=args.data_path)
        print(f"Dataset loaded successfully with {len(dataset)} entries.")
        
        # Print sample item
        sample_item = dataset[0]
        print("\nSample item:")
        for key, value in sample_item.items():
            if key == "image":
                print(f"image: PIL Image of size {value.size}")
            else:
                print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")