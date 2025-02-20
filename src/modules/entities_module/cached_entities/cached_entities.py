import json
from typing import List, Dict, Union, Optional

class EntitiesModule:
    def __init__(self, json_file_path: str):
        """
        Initialize the EntitiesModule with a JSON file path.
        
        Args:
            json_file_path (str): Path to the JSON file containing image entities data
        """
        self.json_file_path = json_file_path
        self.data = self._load_json()
    
    def _load_json(self) -> Dict:
        """
        Load and parse the JSON file.
        
        Returns:
            Dict: Parsed JSON data
        """
        try:
            with open(self.json_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at {self.json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file {self.json_file_path}")
    
    def get_entities_by_index(self, index: Union[int, str]) -> Optional[List[str]]:
        """
        Retrieve entities for a specific image index.
        
        Args:
            index (Union[int, str]): The index of the image in the JSON data
            
        Returns:
            Optional[List[str]]: List of entities for the specified index, or None if not found
        """
        index_str = str(index)
        if index_str in self.data:
            return self.data[index_str].get('entities', [])
        return None
    
# Example usage:
if __name__ == "__main__":
    # Initialize the module
    entities_module = EntitiesModule("test_dataset\links_test.json")
    
    # Get entities for sample 0
    entities = entities_module.get_entities_by_index(20)
    print(f"Entities for sample 0: {entities}")
    