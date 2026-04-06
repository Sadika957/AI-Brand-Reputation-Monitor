import sqlite3
from src.utils.config import DB_PATH


def load_dataframe_to_sqlite(df):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    insert_query = """
        INSERT OR REPLACE INTO sentiment_results (
            post_id, platform, date, brand, text, cleaned_text, engagement,
            sentiment_label, sentiment_score, pos_score, neu_score, neg_score,
            topic_id, topic, topic_keywords
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    rows = [
        (
            str(row["post_id"]),
            row["platform"],
            str(row["date"]),
            row["brand"],
            row["text"],
            row["cleaned_text"],
            int(row["engagement"]),
            row["sentiment_label"],
            float(row["sentiment_score"]),
            float(row["pos_score"]),
            float(row["neu_score"]),
            float(row["neg_score"]),
            int(row["topic_id"]),
            row["topic"],
            row["topic_keywords"],
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany(insert_query, rows)
    conn.commit()
    conn.close()