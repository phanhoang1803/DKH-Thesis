# erm_test.py

from dotenv import load_dotenv
import os
from src.modules.evidence_retrieval_module import ExternalRetrievalModule
from src.dataloaders.cosmos_dataloader import get_cosmos_dataloader
import time

if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.environ.get("GOOGLE_API_KEY")
    CX = os.environ.get("CX")

    from src.config import NEWS_SITES, FACT_CHECKING_SITES

    if not API_KEY or not CX:
        raise ValueError("API_KEY or CX is not set in the environment variables")
    
    # Initialize the module
    retriever = ExternalRetrievalModule(API_KEY, CX, news_sites=NEWS_SITES, fact_checking_sites=FACT_CHECKING_SITES)

    # # Load data
    # dataloader = get_cosmos_dataloader(data_path="data/public_test_acm.json")
    
    # total_time = 0
    # total_items = 0
    # for batch_idx, batch in enumerate(dataloader):
    #     for item in batch:
    #         start_time = time.time()
    #         articles = retriever.retrieve(
    #             text_query=item["caption"],
    #             num_results=10,
    #             threshold=0.7,
    #             news_factcheck_ratio=0.5
    #         )
    #         end_time = time.time()
    #         print(f"Time taken: {end_time - start_time:.2f}s")
    #         total_time += end_time - start_time
    #         total_items += 1
        
    #     if total_items > 100:
    #         break

    # print(f"Total time: {total_time:.2f}s")
    # print(f"Average time: {total_time / total_items:.2f}s")
    
    
    # Example search
    articles = retriever.retrieve(
        text_query="A man lost his testicles while attempting to fill a scuba tank with marijuana smoke.",
        num_results=20,
        threshold=0.7,
        news_factcheck_ratio=0.5
    )
    
    # Print results
    for article in articles:
        print(f"Title: {article.title}")
        print(f"URL: {article.url}")
        print(f"Image URL: {article.image_url}")
        print(f"Content: {article.content}")
        print("---")