import os

from analysis import analyze_loan_agreement
from classifier import classify_document
from flask import Flask, render_template, request
from logger import logger
from pdf2image.pdf2image import convert_from_path
from preprocess import normalize

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Route to handle file upload and preprocessing
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("upload.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No selected file")

        if file:
            filename: str = file.filename  # type: ignore

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
                for i, image in enumerate(images):
                    image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"page_{i}.png")
                    image.save(image_path, "PNG")

                    logger.info("Image file detected, performing OCR")

                    text_content += normalize(image_path)
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
            processed_doc = analyze_loan_agreement(text_content)

            # Classify the extracted text
            predicted_label, confidence = classify_document(text_content)

            # return the processed file to the upload page
            return render_template(
                "upload.html", processed_doc=processed_doc, predicted_label=predicted_label, confidence=confidence
            )

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
