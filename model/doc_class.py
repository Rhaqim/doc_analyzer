# %%
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

from doc_analyzer.config import CLASSIFIER_MODEL_PATH

# %%
# Sample dataset (replace with your actual data)
data = {
    "text": [
        "This is a business plan for our company.",
        "Another business plan for the new project.",
        "A pitch deck for our startup idea.",
        "Pitch deck with our growth strategy.",
    ],
    "label": ["business_plan", "business_plan", "pitch_deck", "pitch_deck"],
}

df = pd.DataFrame(data)
# %%
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
# %%
# Load a pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
# %%
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)  # 2 classes: business_plan and pitch_deck
# %%
# Tokenize the data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors="pt")
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors="pt")
# %%
# Convert labels to integers or category codes
label2id = {"business_plan": 0, "pitch_deck": 1}
y_train = [label2id[label] for label in y_train]
y_test = [label2id[label] for label in y_test]
# %%
# Create PyTorch datasets
train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(y_train))
# %%
test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(y_test))
# %%
# Create pytorch dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
# %%
# Setup optimizer, loss function and evaluation metrics
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()
# %%
# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(50):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        output = model(input_ids, attention_mask=attention_mask)[0]
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")
# %%
# Evaluate the model
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        output = model(input_ids, attention_mask=attention_mask)[0]
        _, predicted_labels = torch.max(output, dim=1)
        predictions.extend(predicted_labels.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
# %%
# export model if accuracy is good enough
if accuracy > 0.9:
    model.save_pretrained(CLASSIFIER_MODEL_PATH)
    tokenizer.save_pretrained("model")
# %%
