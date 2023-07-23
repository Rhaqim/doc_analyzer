import spacy
from logger import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

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


# # Load spaCy language model
# nlp = spacy.load("en_core_web_sm")

# # Prepare your labeled document dataset
# documents = [
#     "This is a positive document.",
#     "Another positive review.",
#     "Negative sentiment in this one.",
#     "Yet another negative document.",
#     # Add more documents and corresponding labels here
# ]

# labels = ["positive", "positive", "negative", "negative"]

# # Tokenize and vectorize the documents using spaCy
# def tokenize_and_vectorize(text: str) -> str:
#     doc = nlp(text)
#     return " ".join(token.lemma_ for token in doc if not token.is_punct and not token.is_stop)

# vectorizer = TfidfVectorizer(tokenizer=tokenize_and_vectorize)
# X = vectorizer.fit_transform(documents)

# # Train the SVM classifier
# classifier = SVC(kernel="linear")
# classifier.fit(X, labels)

# # Evaluate the model
# y_pred = classifier.predict(X)
# accuracy = accuracy_score(labels, y_pred)
# report = classification_report(labels, y_pred)

# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(report)

# # Classify new documents
# new_documents = [
#     "An unknown document.",
#     "A positive review of the product.",
# ]

# X_new = vectorizer.transform(new_documents)
# predictions = classifier.predict(X_new)

# print("Predictions for new documents:")
# for doc, prediction in zip(new_documents, predictions):
#     print(f"Document: {doc}, Predicted Label: {prediction}")



# # Load spaCy language model
# nlp = spacy.load("en_core_web_sm")

# # Prepare your labeled document dataset
# small_loan_agreements = [
#     "This is a small loan agreement document.",
#     "Another small loan agreement.",
#     # Add more small loan agreement documents here
# ]

# promissory_note_security_agreements = [
#     "This is a Promissory Note Security Agreement.",
#     "Another Promissory Note Security Agreement.",
#     # Add more Promissory Note Security Agreement documents here
# ]

# # Combine the datasets and corresponding labels
# documents = small_loan_agreements + promissory_note_security_agreements
# labels = ["small_loan_agreement"] * len(small_loan_agreements) + ["promissory_note_security_agreement"] * len(promissory_note_security_agreements)

# vectorizer = TfidfVectorizer(tokenizer=tokenize_and_vectorize)
# X = vectorizer.fit_transform(documents)

# # Train the SVM classifier
# classifier = SVC(kernel="linear")
# classifier.fit(X, labels)

# # Evaluate the model
# y_pred = classifier.predict(X)
# accuracy = accuracy_score(labels, y_pred)
# report = classification_report(labels, y_pred)

# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(report)

# # Classify new documents
# new_documents = [
#     "A new small loan agreement document.",
#     "A new Promissory Note Security Agreement.",
# ]

# X_new = vectorizer.transform(new_documents)
# predictions = classifier.predict(X_new)

# print("Predictions for new documents:")
# for doc, prediction in zip(new_documents, predictions):
#     print(f"Document: {doc}, Predicted Label: {prediction}")
