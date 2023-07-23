# Connect to postgresql database

from typing import Dict, Optional

from pg8000 import Error, connect

from doc_analyzer.config import (
    DATABASE_CONN,
    DATABASE_NAME,
    DATABASE_PASSWORD,
    DATABASE_PORT,
    DATABASE_USER,
)
from doc_analyzer.logger import logger

DictStrStr = Dict[str, str]


def connect_database():
    """Create database if it does not exist"""
    try:
        connection = connect(
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            host=DATABASE_CONN,
            port=DATABASE_PORT,
            database=DATABASE_NAME,
        )

        return connection
    except (Exception, Error) as error:
        logger.error(f"Error while connecting to PostgreSQL: {error}")
        raise error


def create_database():
    """Create database if it does not exist"""
    try:
        connection = connect_database()

        cursor = connection.cursor()

        #  check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DATABASE_NAME}'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {DATABASE_NAME}")
            logger.info(f"Database {DATABASE_NAME} created successfully in PostgreSQL")
        else:
            logger.info(f"Database {DATABASE_NAME} already exists in PostgreSQL")
        cursor.close()
        connection.close()
    except (Exception, Error) as error:
        logger.error(f"Error while connecting to PostgreSQL: {error}")
        raise error


def create_tables(table_name: str, table_columns: str):
    """Create tables if it does not exist"""
    try:
        connection = connect_database()
        cursor = connection.cursor()
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({table_columns});"
        cursor.execute(create_table_query)
        connection.commit()
        logger.info(f"Table {table_name} created successfully in PostgreSQL")
        cursor.close()
        connection.close()
    except (Exception, Error) as error:
        logger.error(f"Error while connecting to PostgreSQL: {error}")
        raise error


def insert_data(table_name: str, table_columns: str, table_values: str):
    """Insert data into table"""
    try:
        connection = connect_database()
        cursor = connection.cursor()
        insert_query = f"INSERT INTO {table_name} ({table_columns}) VALUES ({table_values});"
        logger.info(f"Insert query: {insert_query}")
        cursor.execute(insert_query)
        connection.commit()
        logger.info(f"Data inserted successfully in PostgreSQL")
        cursor.close()
        connection.close()
    except (Exception, Error) as error:
        logger.error(f"Error while connecting to PostgreSQL: {error}")
        raise error


def select_data(table_name: str, table_columns: str, where_clause: Optional[DictStrStr]):
    """Select data from table"""
    try:
        connection = connect_database()
        cursor = connection.cursor()

        if where_clause is None:
            select_query = f"SELECT {table_columns} FROM {table_name};"
        else:
            where = " AND ".join([f"{k} = '{v}'" for k, v in where_clause.items()])
            select_query = f"SELECT {table_columns} FROM {table_name} WHERE {where};"
        logger.info(f"Select query: {select_query}")
        cursor.execute(select_query)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return result
    except (Exception, Error) as error:
        logger.error(f"Error while connecting to PostgreSQL: {error}")
        raise error
