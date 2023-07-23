import json

import pandas as pd

from doc_analyzer.logger import logger
from doc_analyzer.utils import remove_special_characters, to_snack_case

from .connection import create_database, create_tables, insert_data, select_data

__all__ = ["create_documents_table", "insert_into_documents_table"]

table_name = "documents"


def create_documents_table():
    table_columns = """
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        doc_type VARCHAR(255) NOT NULL,
        doc_category VARCHAR(255) NOT NULL,
        content VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    """
    create_tables(table_name, table_columns)


def insert_into_documents_table(doc_name: str, doc_type: str, doc_category: str, content: str):
    """Insert data into documents table

    Args:
        doc_name (str): Document name
        doc_type (str): File extension e.g. pdf, docx
        doc_category (str): Category of document e.g. loan_agreement
        content (str): Content of the document from OCR
    """

    # create the table if it does not exist
    # create_documents_table()

    # Format the content for insertion into PostgreSQL
    content = remove_special_characters(content)
    doc_name = to_snack_case(doc_name.split(".")[0])

    table_columns = "name, doc_type, doc_category, content"
    table_values = f"'{doc_name}', '{doc_type}', '{doc_category}', '{content}'"
    insert_data(table_name, table_columns, table_values)


def get_documents():
    """Get all documents from the documents table

    Returns:
        List[Dict[str, Any]]: List of documents
    """

    table_columns = "id, name, doc_type, doc_category, content, created_at"
    result = select_data(table_name, table_columns, None)

    documents = []
    for row in result:
        document = {
            "id": row[0],
            "name": row[1],
            "doc_type": row[2],
            "doc_category": row[3],
            "content": row[4],
            "created_at": row[5],
        }
        documents.append(document)

    return documents


def get_document_for_model_train():
    """Get all documents from the documents table and create a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the documents data.
    """
    # Assuming you have a working database connection
    # and the select_data function is defined elsewhere
    table_name = "documents"
    table_columns = "doc_category, content"

    # Assuming you have a working select_data function
    result = select_data(table_name, table_columns, None)

    # for each row in the result, create a dictionary with the column names as keys table_columns.split(", ")
    # and the row values as values
    documents = []
    for row in result:
        document = {
            "doc_category": row[0],
            "content": row[1],
        }
        documents.append(document)
        
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(documents)

    return df