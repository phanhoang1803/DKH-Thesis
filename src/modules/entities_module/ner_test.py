from PIL import Image
from dotenv import load_dotenv
from connector.NER import NERConnector
from huggingface_hub import login
from io import BytesIO

import requests
import json
import os

load_dotenv()
login(token=os.environ["HF_TOKEN"])

DEVICE = os.environ.get["DEVICE"]

# Initialize the connector
ner_connector = NERConnector(
    model_name="dslim/bert-large-NER",
    tokenizer_name="dslim/bert-large-NER",
)
ner_connector.connect()

# Load an image
example = """Donald Trump is the latest president of American"""
# Generate a caption
response = ner_connector.extract_text_entities(example)
print(response)

serializable_results = ner_connector.serialize_ner_results(response)
with open("example_ner_output.json", 'w') as f:
    json.dump(serializable_results, f, indent=4)
