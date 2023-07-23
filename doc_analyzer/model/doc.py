from typing import Any, Iterable, List, Optional

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

from doc_analyzer.config import CLASSIFIER_MODEL_PATH
from doc_analyzer.database.doc_queries import get_document_for_model_train
from doc_analyzer.logger import logger


class DocumentClassifierModel:
    def __init__(self):
        self.model_name: str = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model: Any = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )  # 2 classes: loan_agreement and guaranty_agreement
        self.model_path = CLASSIFIER_MODEL_PATH
        self.train_loader: Optional[Iterable] = None
        self.test_loader: Optional[Iterable] = None
        self.train_encodings: Optional[Iterable] = None
        self.test_encodings: Optional[Iterable] = None
        self.train_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

    def load_data(self):
        """Load the data from the database and prepare it for training"""

        try:
            # Get documents from database
            data = get_document_for_model_train()
            df = data

            # Split the dataset into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                df["content"], df["doc_category"], test_size=0.2, random_state=42
            )

            # Tokenize the data
            self.train_encodings = self.tokenizer(list(X_train), truncation=True, padding=True, return_tensors="pt")
            self.test_encodings = self.tokenizer(list(X_test), truncation=True, padding=True, return_tensors="pt")

            # Convert labels to integers or category codes
            label2id = {"loan_agreement": 0, "guaranty_agreement": 1}
            y_train = [label2id[label] for label in y_train]
            y_test = [label2id[label] for label in y_test]

            # Create PyTorch datasets
            self.train_dataset = TensorDataset(
                self.train_encodings["input_ids"],
                self.train_encodings["attention_mask"],
                torch.tensor(y_train),
            )

            self.test_dataset = TensorDataset(
                self.test_encodings["input_ids"],
                self.test_encodings["attention_mask"],
                torch.tensor(y_test),
            )

            # Create pytorch dataloaders
            self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=True)
        except Exception as e:
            logger.error(f"Error while loading data: {str(e)}")
            raise e

    def train(self):
        """Train the model"""
        try:
            logger.info("Training the model")

            # Set device as GPU if available
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model.to(device)

            # Set optimizer and loss function
            optimizer = AdamW(self.model.parameters(), lr=5e-5)
            loss_fn = CrossEntropyLoss()

            # Train the model
            for epoch in range(50):
                total_loss = 0
                for batch in self.train_loader:
                    optimizer.zero_grad()
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = (
                        input_ids.to(device),
                        attention_mask.to(device),
                        labels.to(device),
                    )
                    output = self.model(input_ids, attention_mask=attention_mask)[0]
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                logger.info(f"Epoch: {epoch}, Loss: {total_loss}")

            # Save the model
            self.save()
        except Exception as e:
            logger.error(f"Error while training the model: {str(e)}")
            raise e

    def evaluate(self):
        """Evaluate the model on the test set"""

        logger.info("Evaluating the model")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        try:
            # Evaluate the model
            self.model.eval()
            predictions = []
            actuals = []
            with torch.no_grad():
                for batch in self.test_loader:
                    input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    _, preds = torch.max(outputs.logits, dim=1)
                    predictions.extend(preds)
                    actuals.extend(labels)

            accuracy = accuracy_score(actuals, predictions)
            logger.info(f"Accuracy: {accuracy}")

            classification_report_str = classification_report(actuals, predictions)
            logger.info(f"Classification Report:\n{classification_report_str}")
        except Exception as e:
            logger.error(f"Error while evaluating the model: {str(e)}")
            raise e

    def save(self):
        """Save the model and tokenizer to the model_path"""
        try:
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info("Model and tokenizer saved successfully.")
        except Exception as e:
            logger.error(f"Error while saving the model: {str(e)}")
            raise e

    def classify_document(self, text: str):
        """Classify the document into one of the categories: loan_agreement or guaranty_agreement

        Args:
            text (str): Text content of the document

        Returns:
            predicted_label (str): Predicted label for the document
            confidence (float): Confidence score for the prediction
        """

        try:
            path = self.model_path / "pytorch_model.bin"

            # Load the trained model weights (replace 'path_to_model' with your saved model path)
            self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
            self.model.eval()

            # Tokenize and prepare input for the model
            inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")

            # Make predictions
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                self.model.to(device)
                self.model.eval()
                outputs = self.model(**inputs)
                logits = outputs.logits  # logits are the unnormalized predictions generated by the model
                probabilities = torch.softmax(
                    logits, dim=1
                )  # dim=1 is the dimension along which softmax will be computed

            # Get predicted class and probability
            predicted_class = torch.argmax(probabilities).item()
            class_labels: List = ["loan_agreement", "guaranty_agreement"]  # 0: loan_agreement, 1: guaranty_agreement
            predicted_label = class_labels[predicted_class]  # type: ignore
            confidence = probabilities[0][predicted_class].item() # type: ignore

            logger.info(f"Predicted Label: {predicted_label}, Confidence: {confidence}")

            return predicted_label, confidence
        except Exception as e:
            logger.error(f"Error while classifying the document: {str(e)}")
            raise e
