
"""
Database connection configuration for the Government Orders system.
Uses Docker PostgreSQL as the primary database.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import PROJECT_ROOT


# Load database credentials from the project root .env file.
load_dotenv(PROJECT_ROOT / ".env")


# Read database configuration.
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

if not all([DB_NAME, DB_USER, DB_PASSWORD]):
    raise RuntimeError(
        "Database credentials are missing. Check the .env file."
    )


# Build the connection URL safely.
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=DB_USER,
    password=DB_PASSWORD,
    host="localhost",
    port=5433,
    database=DB_NAME,
)


# Create the SQLAlchemy engine.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# Create database sessions.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# Base class for future database models.
class Base(DeclarativeBase):
    pass


def test_connection() -> None:
    """Check whether the application can connect to Docker PostgreSQL."""

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user")
        )
        database_name, username = result.fetchone()

        print("Database connection successful!")
        print(f"Database: {database_name}")
        print(f"User: {username}")
        print("Host: localhost")
        print("Port: 5433")
