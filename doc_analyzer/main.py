import os

from analysis import analyze_loan_agreement
from classifier import classify_document
from database.doc_queries import (
    get_document_for_model_train,
    insert_into_documents_table,
)
from flask import Flask, render_template, request
from logger import logger
from model.doc import TrainModel
from pdf2image.pdf2image import convert_from_path
from preprocess import normalize

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Route to handle file upload and preprocessing
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        doc_name, doc_type, doc_category = "", "", ""

        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("upload.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No selected file")

        if file:
            filename: str = file.filename  # type: ignore
            doc_name = filename
            doc_type = filename.split(".")[-1]
            doc_category = "guaranty_agreement"

            logger.info(f"Received file: {filename}")

            # Save the uploaded file to the UPLOAD_FOLDER
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            logger.info(f"Saving file to: {file_path}")

            try:
                file.save(file_path)
            except FileNotFoundError:
                os.mkdir(app.config["UPLOAD_FOLDER"])
                file.save(file_path)

            # Preprocess based on the file type
            if filename.lower().endswith(".pdf"):
                logger.info("PDF file detected")

                # Convert the PDF to images (if scanned) and perform OCR
                images = convert_from_path(file_path)
                text_content = ""

                logger.info(f"Number of pages: {len(images)}")
                logger.info("Image file detected, performing OCR")
                for i, image in enumerate(images):
                    image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"page_{i}.png")
                    image.save(image_path, "PNG")

                    text_content += normalize(image_path)

                    # Delete the image file after processing
                    os.remove(image_path)
            elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                logger.info("Image file detected, performing OCR")

                # Perform OCR directly on the image file
                text_content = normalize(file_path)
            else:
                # For other file types like docx or text, read the content directly
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()

            # Do further processing with the extracted text_content
            # For this example, we'll just print it
            # processed_doc = analyze_loan_agreement(text_content)

            # Insert the processed document into the database
            # insert_into_documents_table(doc_name, doc_type, doc_category, text_content)

            # Classify the extracted text
            predicted_label, confidence = classify_document(text_content)

            # return the processed file to the upload page
            return render_template(
                "upload.html",
                processed_doc=text_content,
                predicted_label=predicted_label,
                confidence=confidence
            )

    return render_template("upload.html")


@app.route("/train", methods=["GET", "POST"])
def train_model():
    train = TrainModel()

    training = False

    if request.method == "POST":
        training = True

        train.load_data()
        train.train()

        return render_template("train.html", training=training)

    return render_template("train.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
