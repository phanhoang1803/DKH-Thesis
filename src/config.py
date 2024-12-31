# config.py

NEWS_SITES = [
    'reuters.com',
    'apnews.com', 
    'bbc.com',
    'npr.org',
    'wsj.com',
    'nytimes.com',
    'economist.com',
    'bloomberg.com',
    # 'washingtonpost.com',
    'ft.com'  # Financial Times
    'nbcnews.com',
    'cnn.com'
]

# Fact Checking Sources
FACT_CHECKING_SITES = [
    'snopes.com',
    'factcheck.org',
    'politifact.com',
    'apfactcheck.org',  # AP Fact Check
    'reuters.com/fact-check',
    'washingtonpost.com/news/fact-checker'
]

# URL Filtering
EXCLUDED_FILE_TYPES = [
    '.pdf',
    '.mp4',
    '.mp3',
    '.mov',
    '.avi',
    '.wmv',
    '.doc',
    '.docx',
    '.ppt',
    '.pptx',
    '.xls',
    '.xlsx'
]

EXCLUDED_URL_KEYWORDS = [
    'video',
    'videos',
    'watch',
    'streaming',
    'podcast',
    'podcasts',
]

def is_valid_url(url: str) -> bool:
    """
    Check if URL is valid based on filtering rules.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if URL is valid, False if it should be filtered out
    """
    # Check file types
    if any(ext in url.lower() for ext in EXCLUDED_FILE_TYPES):
        return False
        
    # Check keywords
    if any(keyword in url.lower() for keyword in EXCLUDED_URL_KEYWORDS):
        return False
        
    return True