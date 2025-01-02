# config.py

NEWS_SITES = [
    # International News Agencies (Primary Sources)
    'reuters.com',     # Known for factual, unbiased reporting
    'apnews.com',      # Associated Press - gold standard in journalism
    
    # Major Global News Organizations
    'bbc.com',         # Strong editorial standards and fact-checking
    'economist.com',   # In-depth analysis and fact-based reporting
    'ft.com',          # Financial Times - rigorous business/economic coverage
    
    # US Elite News Organizations
    'nytimes.com',     # Extensive fact-checking department
    'wsj.com',         # Wall Street Journal - strong business reporting
    'washingtonpost.com',  # Robust fact-checking operation
    
    # US Public Media
    'npr.org',         # National Public Radio - high journalistic standards
]

# Fact Checking Sources
FACT_CHECKING_SITES = [
    # Independent Fact-Checking Organizations
    'factcheck.org',     # Project of Annenberg Public Policy Center
    'politifact.com',    # Pulitzer Prize-winning fact-checker
    'snopes.com',        # Longest-running fact-checking site
    
    # News Agency Fact-Checking Divisions
    'apfactcheck.org',           # AP Fact Check
    'reuters.com/fact-check',    # Reuters Fact Check
    
    # Major News Organization Fact-Checkers
    'washingtonpost.com/news/fact-checker',  # Rigorous methodology
    
    # International Verification Organizations
    'poynter.org/ifcn'  # International Fact-Checking Network
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
    '.xlsx',
    '.txt'
]

EXCLUDED_URL_KEYWORDS = [
    'video',
    'videos',
    'watch',
    'streaming',
    'podcast',
    'podcasts',
    'audio',
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