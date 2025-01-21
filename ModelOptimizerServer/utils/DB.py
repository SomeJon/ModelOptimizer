from dotenv import load_dotenv
import pymysql
import os


class DB:
    _instance = None  # Class-level variable to hold the singleton instance
    _connection = None  # Class-level variable to hold the database connection

    @classmethod
    def get_connection(cls):
        """
        Class-level method to get the database connection.
        Ensures only one connection exists (singleton).
        """
        if cls._connection and cls._connection.open:
            return cls._connection
        else:
            # Load environment variables and initialize the connection
            if not cls._instance:
                load_dotenv()
                cls._instance = cls()
            cls._connection = pymysql.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                port=3306
            )
            return cls._connection

    @classmethod
    def close_connection(cls):
        """Class-level method to close the database connection."""
        if cls._connection and cls._connection.open:
            cls._connection.close()
            cls._connection = None
