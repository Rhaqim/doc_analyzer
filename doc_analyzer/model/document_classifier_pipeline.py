import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

# Define the pipeline components


def prepare_data(new_data_path):
    # Read the new data
    df = pd.read_csv(new_data_path)  # Assuming the data is in a CSV file
    X = df["text"]  # Input texts
    y = df["label"]  # Corresponding labels
    return X, y


def train_model(X_train, y_train):
    # Load a pre-trained BERT model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )  # 2 classes: business_plan and pitch_deck

    # Tokenize the data
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors="pt")

    # Convert labels to integers or category codes
    label2id = {"business_plan": 0, "pitch_deck": 1}
    y_train = [label2id[label] for label in y_train]

    # Create PyTorch datasets
    train_dataset = TensorDataset(
        train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(y_train)
    )

    # Create PyTorch data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Set up the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(5):
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

    return model


def update_model(existing_model, new_data_path):
    # Step 1: Prepare the new data
    X_train, y_train = prepare_data(new_data_path)

    # Step 2: Train the model with the new data
    updated_model = train_model(X_train, y_train)

    # Step 3: Update the existing model with the new model's weights
    existing_model.load_state_dict(updated_model.state_dict())

    return existing_model


# Example pipeline function


def pipeline(existing_model_path, new_data_path):
    # Load the existing model from a file
    existing_model = BertForSequenceClassification.from_pretrained(existing_model_path)

    # Update the model with new data
    updated_model = update_model(existing_model, new_data_path)

    # Save the updated model to a file
    updated_model.save_pretrained(existing_model_path)


# Run the pipeline
existing_model_path = "path_to_existing_model"  # Replace with the path to your existing model
new_data_path = "path_to_new_data.csv"  # Replace with the path to your new data file

# pipeline(existing_model_path, new_data_path)
