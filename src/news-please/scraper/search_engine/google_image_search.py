import os
from dotenv import load_dotenv
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.oauth2 import service_account

class GoogleImageSearch:
    def __init__(self, api_key=None, cx=None, news_sites=None):
        self.news_sites = news_sites or []
        self.api_key = api_key
        self.cx = cx
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
        self.client = vision.ImageAnnotatorClient()
        
    def search(self, query, num_results=10):
        if os.path.exists(query):
            # If the query is a local image path, use it
            image = self._load_image(query)
        else:
            # Text-based queries are not supported for web detection, raise an error
            raise ValueError("For web detection, please provide a valid image path.")
        
        return self._detect_web_entities(image)

    def _search(self, image, num_results=10):
        response = self.client.image_search(image=image, max_results=num_results)
        return response.image_search_results.results

    def _detect_web_entities(self, image):
        # Perform web detection on the image
        response = self.client.web_detection(image=image)
        annotations = response.web_detection
        
        results = {
            "best_guess_labels": [],
            "pages_with_matching_images": [],
            "web_entities": [],
            "visually_similar_images": []
        }
        
        # Process best guess labels
        if annotations.best_guess_labels:
            results["best_guess_labels"] = [label.label for label in annotations.best_guess_labels]
        
        # Process pages with matching images
        for page in annotations.pages_with_matching_images:
            page_data = {"url": page.url, "full_matches": [], "partial_matches": []}
            
            # Full matches
            page_data["full_matches"] = [image.url for image in page.full_matching_images]
            
            # Partial matches
            page_data["partial_matches"] = [image.url for image in page.partial_matching_images]
            
            results["pages_with_matching_images"].append(page_data)
        
        # Process web entities
        results["web_entities"] = [
            {"score": entity.score, "description": entity.description} 
            for entity in annotations.web_entities
        ]
        
        # Process visually similar images
        results["visually_similar_images"] = [image.url for image in annotations.visually_similar_images]
        
        if response.error.message:
            raise Exception(
                f"{response.error.message}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors"
            )
        
        return results
    
    def _load_image(self, image_path):
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        return types.Image(content=content)

    def _filter_news_sites(self, search_results):
        filtered_items = []

        for item in search_results:
            url = item.get("link")
            if any(news_site in url for news_site in self.news_sites):
                filtered_items.append(item)

        return filtered_items
    
def main():
    load_dotenv()
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("CX")
    
    if not api_key or not cx:
        raise ValueError("API_KEY or CX is not set in the environment variables")
    
    # Set up the Google Image Search client
    search = GoogleImageSearch(
        api_key=api_key,
        cx=cx,
        news_sites=["nytimes.com", "cnn.com", "washingtonpost.com", "foxnews.com", "usatoday.com", "wsj.com"]
    )

    # Search for images using a file path
    image_path = "images.jpg"
    image_results = search.search(image_path)
    
    # Display search results
    print("Best Guess Labels:")
    for label in image_results["best_guess_labels"]:
        print(f"- {label}")

    print("\nPages with Matching Images:")
    for page in image_results["pages_with_matching_images"]:
        print(f"\nPage URL: {page['url']}")
        print("Full Matches:")
        for url in page["full_matches"]:
            print(f"  - {url}")
        print("Partial Matches:")
        for url in page["partial_matches"]:
            print(f"  - {url}")
            
    print("\nWeb Entities:")
    for entity in image_results["web_entities"]:
        print(f"- Score: {entity['score']}, Description: {entity['description']}")
        
    print("\nVisually Similar Images:")
    for url in image_results["visually_similar_images"]:
        print(f"- {url}")

if __name__ == "__main__":
    main()