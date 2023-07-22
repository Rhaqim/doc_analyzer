import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import tensor
from transformers import BertForSequenceClassification, BertTokenizer

# Prepare your labeled document dataset
small_loan_agreements = [
    "This is a small loan agreement document.",
    "Another small loan agreement.",
    # Add more small loan agreement documents here
]

promissory_note_security_agreements = [
    "This is a Promissory Note Security Agreement.",
    "Another Promissory Note Security Agreement.",
    # Add more Promissory Note Security Agreement documents here
]

# Combine the datasets and corresponding labels
documents = small_loan_agreements + promissory_note_security_agreements
labels = ["small_loan_agreement"] * len(small_loan_agreements) + ["promissory_note_security_agreement"] * len(promissory_note_security_agreements)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the data
train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt")
test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt")

# Create PyTorch datasets
train_dataset = data.TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(y_train))
test_dataset = data.TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], tensor(y_test))

# Create PyTorch data loaders
train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=16)

# Set up the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Train the model
model.train()
for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        output = model(input_ids, attention_mask=attention_mask)[0]
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = []
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        output = model(input_ids, attention_mask=attention_mask)[0]
        _, predicted_labels = torch.max(output, dim=1)
        predictions.extend(predicted_labels.tolist())

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
