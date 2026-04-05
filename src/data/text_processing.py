import re

def clean_text(text):
    """
    Lightweight aggressive tokenization text cleaning.
    Lowercases, removes non-alphanumeric, strips extra whitespaces.
    We don't remove stopwords yet, allowing TF-IDF or Word2Vec 
    libraries to handle specific vocabulary controls.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove HTML tags if any exist
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove non-alphabetical characters, keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text
