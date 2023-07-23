import json
import re


def format_content_for_postgres(content):
    """
    Format content to be inserted into PostgreSQL JSON column.

    Parameters:
        content (dict or list): The Python object to be converted into a JSON string.

    Returns:
        str: The formatted JSON string ready for insertion into PostgreSQL.
    """
    try:
        json_string = json.dumps(content)
        return json_string
    except Exception as e:
        raise ValueError(f"Error formatting content as JSON: {e}")


def to_snack_case(string: str) -> str:
    """
    Converts a strings of characters to snake case.
    replaces all spaces with underscores and
    makes all characters lowercase.
    """
    return re.sub(r"\s+", "_", string).lower()


def remove_special_characters(string: str) -> str:
    """
    Removes special characters from a sentence and leave aplhanumeric characters and spaces, remove brackets and \n
    """

    return re.sub(r"[^A-Za-z0-9 ]+", "", string).replace("\n", " ")
