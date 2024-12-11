from connector.llm import LLMConnector
from dotenv import load_dotenv
from huggingface_hub import login

import os
import json

load_dotenv()
login(token=os.environ["HF_TOKEN"])

DEVICE = os.environ.get["DEVICE"]

# Example usage with logging enabled
llm_connector = LLMConnector(
    model_name="meta-llama/Llama-3.2-1B-Instruct", device=DEVICE
)
llm_connector.connect()

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

response = llm_connector.answer(messages, max_new_tokens=256)

print(response)

with open("example_llm_output.json", "w") as f:
    json.dump(response, f, indent=4)
