# %%

import cv2
import pytesseract
import torch
from config import CLASSIFIER_MODEL_PATH
from transformers import BertForSequenceClassification, BertTokenizer

# %%

# %%
# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
model_path = CLASSIFIER_MODEL_PATH / "pytorch_model.bin"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)  # 2 classes: business_plan and pitch_deck

# Load the trained model weights (replace 'path_to_model' with your saved model path)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()


# %%
# OCR function to extract text from the uploaded image/PDF
def perform_ocr(image_path):
    image = cv2.imread(image_path)
    text_content = pytesseract.image_to_string(image)
    return text_content


# %%
# Document classification function
def classify_document(text):
    # Tokenize and prepare input for the model
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    # Get predicted class and probability
    predicted_class = torch.argmax(probabilities).item()
    class_labels = ["business_plan", "pitch_deck"]
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence


# %%

# %%
# %%

# %%
