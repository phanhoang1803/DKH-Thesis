from PIL import Image
from dotenv import load_dotenv
from src.modules.reasoning_module.connector.blip2 import BLIP2Connector
from huggingface_hub import login
from io import BytesIO

import requests
import json
import os

load_dotenv()
login(token=os.environ["HF_TOKEN"])
DEVICE = os.environ["DEVICE"]

# Initialize the connector
blip2_connector = BLIP2Connector(
    model_name="Salesforce/blip2-opt-2.7b", device=DEVICE, torch_dtype="bfloat16"
)
blip2_connector.connect()

# Load an image
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIh9xqJqItd4IfLJZ7yTyJ43erRAEbob8BrA&s"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = Image.open(BytesIO(requests.get(image_url).content))

# Generate a caption
response = blip2_connector.caption(image)
print(response)

with open("example_blip2_output.json", "w") as f:
    json.dump(response, f, indent=4)
