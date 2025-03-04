import base64
import codecs
import json
from typing import Optional, Dict, Any
from datetime import datetime
from torch.utils.data import Dataset
from argparse import ArgumentParser
from PIL import Image
import os
from io import BytesIO

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
            "title": item["title"],
            "caption": item["caption"],
            "content": content,
            "image_path": item["image_path"],
            "image": image
        }


class MergedBalancedNewsClippingDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Any] = None):
        """
        Args:
            data_path (str): Path to the root directory containing dataset files
            transform (Optional[Any]): Optional transform to be applied on images
        """
        self.data_path = data_path
        self.newsclipping_annotations = self._load_data(os.path.join(data_path, "news_clippings_test.json"))["annotations"]
        self.visualnews_data_mapping = self._load_data(os.path.join(data_path, "visual_news_test.json"))
        self.transform = transform
        
    def __len__(self):
        return len(self.newsclipping_annotations)
   
    @staticmethod
    def _load_data(data_path: str):
        """Load JSON data with proper error handling."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {data_path}: {str(e)}")
        return data
    
    def _read_text_file(self, file_path: str) -> str:
        """
        Read text file with multiple encoding fallbacks.
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with codecs.open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Warning: Unexpected error reading file {file_path}: {str(e)}")
                
        # If all encodings fail, try binary read and decode with replacement
        try:
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Failed to read file {file_path} with any encoding: {str(e)}")
            return ""
    
    def __getitem__(self, idx: int):
        """
        Get a dataset item with robust error handling.
        
        Returns:
            Dict containing:
                - image: PIL Image or transformed image
                - image_base64: Image base64
                - caption: str
                - content: str
                - label: bool
                - metadata: Dict with additional information
        """
        item = self.newsclipping_annotations[idx]
        result = {
            "image": None,
            "image_base64": None,
            "caption": "",
            "content": "",
            "label": item["falsified"],
            "metadata": {
                "id": item["id"],
                "image_id": item["image_id"],
                "similarity_score": item["similarity_score"],
                "source_dataset": item["source_dataset"]
            }
        }
        
        # Load image
        try:
            image_path = os.path.join(self.data_path, 
                                    self.visualnews_data_mapping[str(item["image_id"])]["image_path"])
            image = Image.open(image_path).convert('RGB')
            
            # with open(image_path, "rb") as image_file:
            #     # Read the file and encode it to Base64
            #     image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # if self.transform:
            #     image = self.transform(image)
            # result["image"] = image
            # result["image_base64"] = image_base64
            
            # Resize before encoding to base64 (maintain aspect ratio)
            max_size = 1024  # Maximum dimension (width or height)
            width, height = image.size
            if width > height and width > max_size:
                new_width = max_size
                new_height = int(height * (max_size / width))
                image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            elif height > width and height > max_size:
                new_height = max_size
                new_width = int(width * (max_size / height))
                image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                image_resized = image  # No resize needed

            # Save resized image to buffer and encode
            buffer = BytesIO()
            image_resized.save(buffer, format='JPEG', quality=90)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Apply transform for tensor operations (if needed)
            if self.transform:
                image = self.transform(image)

            result["image"] = image
            result["image_base64"] = image_base64
        except Exception as e:
            # print(f"Warning: Failed to load image for item {item['id']}: {str(e)}")
            raise e
        
        # Load caption
        try:
            result["caption"] = self.visualnews_data_mapping[str(item["id"])]["caption"]
        except KeyError:
            # print(f"Warning: Caption not found for item {item['id']}")
            raise e
        
        # Load article content
        try:
            article_path = os.path.join(self.data_path, 
                                      self.visualnews_data_mapping[str(item["id"])]["article_path"])
            result["content"] = self._read_text_file(article_path)
        except Exception as e:
            # print(f"Warning: Failed to load article for item {item['id']}: {str(e)}")
            raise e
            
        return result
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    from torchvision import transforms
    
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="test_dataset",
        help="Path to the dataset root directory"
    )
    args = parser.parse_args()

    try:
        # Initialize dataset without transforms for visualization
        dataset = MergedBalancedNewsClippingDataset(data_path=args.data_path)
        print(f"Dataset loaded successfully with {len(dataset)} entries.")
        
        # Get and display sample
        sample = dataset[0]
        
        # Print text information
        print("\nSample item:")
        print(f"Caption: {sample['caption']}")
        print(f"Content length: {len(sample['content'])} characters")
        print(f"Label: {sample['label']}")
        print(f"Metadata: {sample['metadata']}")
        
        # Display image if available
        if sample['image'] is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(sample['image'])
            plt.axis('off')
            plt.title(f"Sample Image\nCaption: {sample['caption'][:100]}...")
            plt.show()
        else:
            print("No image available for this sample")
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        
    # Example with transforms
    print("\nLoading dataset with transforms...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset_transformed = MergedBalancedNewsClippingDataset(
            data_path=args.data_path,
            transform=transform
        )
        sample_transformed = dataset_transformed[0]
        
        if sample_transformed['image'] is not None:
            # Convert tensor to image for display
            img_transformed = transforms.ToPILImage()(sample_transformed['image'])
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img_transformed)
            plt.axis('off')
            plt.title("Transformed Image (224x224)")
            plt.show()
            
    except Exception as e:
        print(f"Error loading transformed dataset: {str(e)}")