import os

import cv2
import numpy as np
import pytesseract
from flask import current_app
from PIL import Image

from doc_analyzer.logger import logger


def normalize(image_path: str) -> str:
    logger.info("Normalizing image")

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

    logger.info(f"Normalized image saved to: {normalized_image_path}")

    # Perform OCR on the normalized image
    normalized_text = perform_ocr(normalized_image_path)

    # Delete the normalized image file to free up storage space
    os.remove(normalized_image_path)

    return normalized_text


def perform_ocr(image_path: str) -> str:
    logger.info("Performing OCR")

    # Perform OCR using Tesseract
    custom_config = r"--oem 3 --psm 6"
    ocr_text = pytesseract.image_to_string(Image.open(image_path), config=custom_config)

    return ocr_text


def normalize_image(image: Image.Image) -> str:
    logger.info("Normalizing image")

    # Convert the image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Resize the image to a standard size for better OCR results (you can adjust the size)
    resized_image = cv2.resize(image_cv, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Convert the image to grayscale for better text visibility
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to improve text visibility against various backgrounds
    _, threshold_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform OCR on the normalized image
    normalized_text = perform_ocr_image_byte(threshold_image)

    return normalized_text


def perform_ocr_image_byte(image: np.ndarray) -> str:
    logger.info("Performing OCR")

    # Perform OCR using Tesseract
    custom_config = r"--oem 3 --psm 6"
    ocr_text = pytesseract.image_to_string(Image.fromarray(image), config=custom_config)

    return ocr_text
