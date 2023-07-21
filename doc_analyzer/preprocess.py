import os

import cv2
import pytesseract


def normalize(image_path: str) -> str:
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Resize the image to a standard size for better OCR results (you can adjust the size)
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Convert the image to grayscale for better text visibility
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to improve text visibility against various backgrounds
    _, threshold_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the normalized image temporarily to a new file (use a unique name to avoid conflicts)
    normalized_image_path = image_path + "_normalized.png"
    cv2.imwrite(normalized_image_path, threshold_image)

    # Perform OCR on the normalized image
    normalized_text = perform_ocr(normalized_image_path)

    # Delete the normalized image file to free up storage space
    os.remove(normalized_image_path)

    return normalized_text

# Function to perform OCR on scanned images
def perform_ocr(image_path: str) -> str:
    return pytesseract.image_to_string(image_path)