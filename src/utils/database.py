import os
import psycopg2
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def get_connection():
    """
    Creates and returns a new PostgreSQL connection using credentials
    from environment variables (.env).

    Returns:
        A psycopg2 connection object, or raises an exception if the
        connection fails (we want this to fail loudly — silently
        continuing without a database connection would cause confusing
        errors deeper in the pipeline).
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
