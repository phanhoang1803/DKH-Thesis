from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from src.utils.logger import Logger

import torch
import time
import numpy as np

class NERConnector:
    def __init__(
        self, model_name: str, tokenizer_name: str, device: str = "cpu", torch_dtype: str = "bfloat16"
    ):
        """
        Initialize the Hugging Face NER connector.

        Args:
            model_name (str): The name of the Hugging Face model 
            device (str): Device to run the model on. Use 'cpu' for CPU or 'cuda' for GPU if available.
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.logger = Logger(name="NERConnector")
        self.pipeline = None

    def connect(self):
        """
        Initialize the Hugging Face pipeline.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.model =  AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline("ner",
                                      model=self.model, 
                                    tokenizer=self.tokenizer
            )
            self.logger.info(f"Successfully connected to model '{self.model_name}'")
        except Exception as e:
            self.logger.error(f"Failed to connect to model '{self.model_name}': {e}")
            raise

    def extract_text_entities(self, text: str):
        """
        Generate a conversational response based on structured input.

        Args:
            messages (list): List of dictionaries representing the conversation,
                             with keys 'role' and 'content'.
            max_new_tokens (int): Maximum number of tokens for the response.

        Returns:
            dict: A dictionary containing the generated response.
        """
        if not self.pipeline:
            self.logger.error("Model is not connected. Call `connect()` first.")
            raise Exception("Model is not connected. Call `connect()` first.")

        try:
            # Generate text entities
            self.logger.info("Generating entities for the given text.")
            ner_results = self.pipeline(text)

            self.logger.info(f"Generated entities: {ner_results}")
            return ner_results
        except Exception as e:
            self.logger.error(f"Failed to generate conversational response: {e}")
            return {"error": str(e)}
    @staticmethod
    def serialize_ner_results(entities):
        """
        Static method to convert NER results with numpy datatypes to JSON serializable format.
        
        Args:
            entities (list): List of dictionaries containing NER results
            
        Returns:
            list: JSON serializable version of the NER results
        """
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        return convert_numpy_types(entities)