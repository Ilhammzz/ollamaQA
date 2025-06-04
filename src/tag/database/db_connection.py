import psycopg2
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

def connect_db():
    """
    Membuka koneksi ke database PostgreSQL menggunakan DB_URI dari .env.
    """
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise ValueError("DB_URI not found in environment variables.")

    result = urlparse(db_uri)
    
    return psycopg2.connect(
        dbname=result.path.lstrip("/"),
        user=result.username,
        password=result.password,
        host=result.hostname,
        port=result.port
    )
