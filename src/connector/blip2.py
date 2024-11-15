from utils.logger import Logger
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import time


class BLIP2Connector:
    def __init__(
        self, model_name: str, device: str = "cpu", torch_dtype: str = "bfloat16"
    ):
        """
        Initialize the BLIP2 connector for image captioning.

        Args:
            model_name (str): The name of the BLIP2 model (e.g., 'Salesforce/blip2-opt-2.7b').
            device (str): Device to run the model on. Use 'cpu' for CPU or 'cuda' for GPU if available.
            torch_dtype (str): Data type for Torch operations ('bfloat16' or 'float32').
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.logger = Logger(name="BLIP2Connector")
        self.processor = None
        self.model = None

    def connect(self):
        """
        Initialize the BLIP2 model and processor.
        """
        try:
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float32
                ),
            )

            self.model.to(self.device)
            self.logger.info(
                f"Successfully connected to BLIP2 model '{self.model_name}'"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to connect to BLIP2 model '{self.model_name}': {e}"
            )
            raise

    def caption(self, image, max_new_tokens: int = 20):
        """
        Generate a caption for the given image.

        Args:
            image (PIL.Image): The input image.
            max_new_tokens (int): Maximum number of tokens for the caption.

        Returns:
            dict: A dictionary containing the generated caption.
        """
        if not self.processor or not self.model:
            self.logger.error("Model is not connected. Call `connect()` first.")
            raise Exception("Model is not connected. Call `connect()` first.")

        try:
            # Preprocess the image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Generate caption
            self.logger.info("Generating caption for the image.")
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Decode the caption
            caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            self.logger.info(f"Generated caption: {caption}")

            # Return the result in a structured format
            response = {
                "id": "generated_id_placeholder",
                "object": "image_caption",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "text": caption,
                        "finish_reason": (
                            "length"
                            if len(caption.split()) >= max_new_tokens
                            else "stop"
                        ),
                    }
                ],
            }
            return response
        except Exception as e:
            self.logger.error(f"Failed to generate caption: {e}")
            return {"error": str(e)}
