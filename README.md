# DKH-Thesis

## Installation

1. Clone the repository
```bash
git clone https://github.com/phanhoang1803/DKH-Thesis.git
cd DKH-Thesis
```

2. Switch to the scraper module branch. If you plan to work with the scraper module, switch to the scraper.dev branch:
```bash
git checkout scraper.dev
```

3. Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
.\venv\Scripts\activate
```

4. Install the required dependencies. Run the following command to install all necessary Python libraries:
```bash
pip install -r requirements.txt
```

5. Set up the .env file. Create a .env file in the root directory of the project and add your Google API key and custom search engine ID (CX):
```
GOOGLE_API_KEY=<your_google_api_key>
CX=<your_custom_search_engine_id>
```

## Usage

The scraper module is located in scraper/scraper.py. To run an example:
```bash
python scraper/scraper.py
```

## Features
- Text-based search: The current version supports searching by text input.

## TODO

There are some parts still need to optimize and install. Currently, we just provide search by text.