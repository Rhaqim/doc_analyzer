# Use a base image with Python 3.11
FROM python:3.11

# Install system dependencies required for OpenCV, pytesseract, poppler, and tesseract
RUN apt-get update \
    && apt-get install -y poppler-utils tesseract-ocr libgl1-mesa-glx transformers \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry's bin to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Install spaCy language model 'en_core_web_sm'
RUN pip install spacy
RUN python -m spacy download en_core_web_sm

# Copy the project files into the container
COPY doc_analyzer /app/doc_analyzer
COPY pyproject.toml poetry.lock README.md ./

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Expose the Flask app's port (assuming your Flask app is using port 5000)
# EXPOSE 5000

# Start the Flask app
CMD ["python", "doc_analyzer/main.py"]
