import pandas as pd

from src.ingest.load_raw_data import load_raw_data
from src.nlp.sentiment import get_sentiment
from src.nlp.bertopic_model import extract_topics_bertopic
from src.db.init_db import init_database
from src.db.load_data import load_dataframe_to_sqlite


def run_pipeline():
    print("Initializing database")
    init_database()

    print("Loading CSV data")
    df = load_raw_data()
    print("Rows loaded:", len(df))

    print("Running sentiment analysis")
    sentiment_results = df["text"].apply(get_sentiment)
    sentiment_df = pd.DataFrame(sentiment_results.tolist())

    print("Running BERTopic topic modeling")
    topic_df = extract_topics_bertopic(df["text"].tolist())

    final_df = pd.concat(
        [df.reset_index(drop=True), sentiment_df, topic_df],
        axis=1
    )

    print("Saving results")
    load_dataframe_to_sqlite(final_df)

    print("Pipeline finished")
    print(final_df[["post_id", "sentiment_label", "topic", "topic_keywords"]].head())


if __name__ == "__main__":
    run_pipeline()