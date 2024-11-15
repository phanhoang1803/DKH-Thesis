from utils.logger import Logger
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional

import torch
import time


class LLMConnector:
    def __init__(
        self, model_name: str, device: str = "cpu", torch_dtype: str = "bfloat16"
    ):
        """
        Initialize the Hugging Face LLM connector.

        Args:
            model_name (str): The name of the Hugging Face model (e.g., 'gpt2', 'facebook/opt-1.3b').
            device (str): Device to run the model on. Use 'cpu' for CPU or 'cuda' for GPU if available.
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.logger = Logger(name="LLMConnector")
        self.pipeline = None

    def connect(self):
        """
        Initialize the Hugging Face pipeline.
        """
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=(
                    torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float32
                ),
            )
            self.logger.info(f"Successfully connected to model '{self.model_name}'")
        except Exception as e:
            self.logger.error(f"Failed to connect to model '{self.model_name}': {e}")
            raise

    def answer(self, messages, max_new_tokens: int = 256):
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
            # Format the input messages into a single prompt
            formatted_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted_prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    formatted_prompt += f"User: {msg['content']}\n"
            formatted_prompt += "Pirate Chatbot:"

            self.logger.info(f"Generating response for prompt: {formatted_prompt}")

            # Use the pipeline to generate text
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
            )

            # Extract the generated response
            generated_text = outputs[0]["generated_text"]
            completion_text = generated_text[len(formatted_prompt) :].strip()

            self.logger.info(f"Generated text: {completion_text}")

            # Return the result in OpenAI-like format
            response = {
                "id": "generated_id_placeholder",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "text": completion_text,
                        "finish_reason": (
                            "length"
                            if len(completion_text) >= max_new_tokens
                            else "stop"
                        ),
                    }
                ],
            }
            return response
        except Exception as e:
            self.logger.error(f"Failed to generate conversational response: {e}")
            return {"error": str(e)}
