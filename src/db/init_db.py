import os
import sqlite3
from src.utils.config import DB_PATH


def init_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT UNIQUE,
            platform TEXT,
            date TEXT,
            brand TEXT,
            text TEXT,
            cleaned_text TEXT,
            engagement INTEGER,
            sentiment_label TEXT,
            sentiment_score REAL,
            pos_score REAL,
            neu_score REAL,
            neg_score REAL,
            topic_id INTEGER,
            topic TEXT,
            topic_keywords TEXT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_database()
    print("Database created successfully")