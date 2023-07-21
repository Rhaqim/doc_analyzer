# Document Analyzer

## Description

This is a simple document analyzer app that accepts a file upload and outputs the total word count, the count of each word, and the top 10 words by count in descending order.

## Installation

1. Clone the repository
2. run the following commands in the root directory of the project:

```bash
docker build -t document-analyzer .
```

```bash
docker run -p 5001:5000 document-analyzer
```

## Usage

1. Navigate to `http://localhost:5001/upload`
2. Select a file to upload
3. Check terminal for output
