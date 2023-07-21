import os

from analysis import analyze_loan_agreement
from flask import Flask, render_template, request
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

            # Save the uploaded file to the UPLOAD_FOLDER
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            try:
                file.save(file_path)
            except FileNotFoundError:
                os.mkdir(app.config["UPLOAD_FOLDER"])
                file.save(file_path)

            # Preprocess based on the file type
            if filename.lower().endswith(".pdf"):
                # Convert the PDF to images (if scanned) and perform OCR
                images = convert_from_path(file_path)
                text_content = ""
                for i, image in enumerate(images):
                    image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"page_{i}.png")
                    image.save(image_path, "PNG")
                    # text_content += perform_ocr(image_path)
                    text_content += normalize(image_path)
            elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                # Perform OCR directly on the image file
                # text_content = perform_ocr(file_path)
                text_content = normalize(file_path)
            else:
                # For other file types like docx or text, read the content directly
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()

            # Do further processing with the extracted text_content
            # For this example, we'll just print it
            print(analyze_loan_agreement(text_content))

            return render_template("upload.html", message="File successfully uploaded and processed.")

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
