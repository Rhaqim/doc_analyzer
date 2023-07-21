import spacy
from logger import logger

# Load the spaCy language model (make sure you have already installed the 'en_core_web_sm' model)
nlp = spacy.load("en_core_web_sm")

def analyze_loan_agreement(text_content: str) -> dict:
    """
    Analyze the text content of a loan agreement and return a dictionary of keyword occurrences.
    """
    # Process the text using spaCy
    doc = nlp(text_content)

    # Define a list of key words related to small loan agreements
    keywords = ["loan", "agreement", "borrower", "lender", "amount", "interest", "repayment", "due date", "default", "term"]

    logger.info(f"Analyzing text content for keywords: {keywords}")

    # Initialize a dictionary to store the occurrences of key words
    keyword_occurrences = {keyword: 0 for keyword in keywords}

    # Loop through the text and count occurrences of key words
    for token in doc:
        if token.text.lower() in keywords:
            keyword_occurrences[token.text.lower()] += 1

    return keyword_occurrences
