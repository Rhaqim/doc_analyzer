import io

from database.doc_queries import insert_into_documents_table
from flask import Flask, jsonify, render_template, request
from logger import logger
from model.doc import DocumentClassifierModel
from pdf2image.pdf2image import convert_from_bytes
from PIL import Image
from preprocess import normalize_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        doc_name, doc_type = "", ""

        # Check if the post request has the file part
        if "file" not in request.files:
            logger.error("No file part in the request")
            return render_template("upload.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            logger.error("No selected file")
            return render_template("upload.html", error="No selected file")

        if file:
            filename: str = file.filename  # type: ignore
            doc_name = filename
            doc_type = filename.split(".")[-1]

            logger.info(f"Received file: {filename}")

            # Preprocess based on the file type
            text_content = ""
            if filename.lower().endswith(".pdf"):
                logger.info("PDF file detected")

                # Convert the PDF to images (if scanned) and perform OCR
                pdf_bytes = file.read()
                images = convert_from_bytes(pdf_bytes)

                logger.info(f"Number of pages: {len(images)}")
                logger.info("Image file detected, performing OCR")
                for i, image in enumerate(images):
                    # Perform OCR directly on the image
                    text_content += normalize_image(image)

            elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                logger.info("Image file detected, performing OCR")

                # Read the image as bytes and convert to PIL image
                image_bytes = file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Perform OCR directly on the image
                text_content = normalize_image(pil_image)
            else:
                # For other file types like docx or text, read the content directly
                text_content = file.read().decode("utf-8")

            # Classify the extracted text
            classifier = DocumentClassifierModel()
            predicted_label, confidence = classifier.classify_document(text_content)

            # Insert the processed document into the database
            # insert_into_documents_table(doc_name, doc_type, predicted_label, text_content)

            # Return the processed file to the upload page
            # return render_template("upload.html", predicted_label=predicted_label, confidence=confidence)

            return jsonify(
                {
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                }
            )

    return render_template("upload.html")


@app.route("/train", methods=["GET", "POST"])
def train_model():
    training = False

    if request.method == "POST":
        training = True

        train = DocumentClassifierModel()

        train.load_data()
        train.train()

        return render_template("train.html", training=training)

    return render_template("train.html")


# Route to fetch documents from the backend
@app.route("/get_documents", methods=["GET"])
def get_documents():
    # Example data for documents (you can replace this with data from your database)
    documents = [
        {"id": 1, "name": "Document 1"},
        {"id": 2, "name": "Document 2"},
        {"id": 3, "name": "Document 3"},
        # Add more documents as needed
    ]

    return jsonify({"documents": documents}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
