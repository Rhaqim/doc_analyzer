# Document Analyzer

## Description

This is a simple document analyzer app that can be used to train a model to classify documents into different categories.
It is separated into three parts:

1. The Upload (Test data) section, where you can upload documents and get the classification results.
2. The Upload (Train data) section, where you can upload documents and save them to the database.
3. The Training section, where you can select documents from the database and train a model to classify them.

## Installation

1. Run the following commands in the root directory of the project:

```bash
docker build -t document-analyzer .
```

```bash
docker run -p 5001:5000 document-analyzer
```

1. With Docker-compose:

```bash
docker-compose up
```

## Usage

1. Navigate to `http://localhost:5001` in your browser.
