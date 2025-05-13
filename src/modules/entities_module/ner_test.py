from PIL import Image
from dotenv import load_dotenv
from textual_entity_extractor import TextualEntityExtractor
from huggingface_hub import login
from io import BytesIO

import requests
import json
import os

load_dotenv()
login(token=os.environ["HF_TOKEN"])

# DEVICE = os.environ.get["DEVICE"]

# Initialize the connector
ner_connector = TextualEntityExtractor(
    # model_name="dslim/bert-large-NER",
    # tokenizer_name="dslim/bert-large-NER",
    model_name="dbmdz/bert-large-cased-finetuned-conll03-english",
    tokenizer_name="dbmdz/bert-large-cased-finetuned-conll03-english"
)
ner_connector.connect()

# Load an image
example = """Donald Trump is the latest president of American"""
# Generate a caption
response = ner_connector.extract_textual_entities(example)
print(response)

serializable_results = ner_connector.serialize_ner_results(response)
with open("example_ner_output.json", 'w') as f:
    json.dump(serializable_results, f, indent=4)
