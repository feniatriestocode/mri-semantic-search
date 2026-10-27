# src/db.py
# This module provides a function to establish a connection
#  to the PostgreSQL database with pgvector support.
import psycopg2
from pgvector.psycopg2 import register_vector

# Connection parameters as defined in docker-compose.yml
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vectordb",
    "user": "postgres",
    "password": "password"
}

def get_connection():
    """
    Creates and returns a new Connection object 
    according to the Python DB-API 2.0, with pgvector support.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    # Register the vector type in the connection session
    register_vector(conn)
    return conn