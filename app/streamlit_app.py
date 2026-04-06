import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

DB_PATH = "database/sentiment_monitor.db"

st.set_page_config(page_title="Sentiment Intelligence Dashboard", layout="wide")


@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sentiment_results", conn)
    conn.close()
    return df


def clean_topic_label(topic_value) -> str:
    """
    Clean BERTopic-style labels like:
    '-1_netflix_the_is_too' -> 'Other'
    '0_price_subscription_cost' -> 'price, subscription, cost'
    """
    if pd.isna(topic_value):
        return "Other"

    topic_str = str(topic_value).strip()

    if topic_str == "" or topic_str == "-1":
        return "Other"

    if topic_str.startswith("-1_"):
        return "Other"

    # Split BERTopic-like labels
    parts = topic_str.split("_")

    # If first part is numeric topic id, remove it
    if parts and parts[0].lstrip("-").isdigit():
        parts = parts[1:]

    # Remove very common noisy words
    noisy_words = {
        "netflix", "the", "is", "too", "very", "really", "just",
        "this", "that", "and", "for", "with", "what"
    }

    cleaned_parts = [p for p in parts if p and p.lower() not in noisy_words]

    if not cleaned_parts:
        return "Other"

    return ", ".join(cleaned_parts[:4]).title()


def generate_summary(df: pd.DataFrame) -> str:
    total_mentions = len(df)
    positive = (df["sentiment_label"] == "Positive").sum()
    neutral = (df["sentiment_label"] == "Neutral").sum()
    negative = (df["sentiment_label"] == "Negative").sum()

    if df.empty:
        return "No data is available for the selected filters."

    top_topic = df["topic_clean"].value_counts().idxmax() if not df["topic_clean"].empty else "N/A"

    negative_topics = df[df["sentiment_label"] == "Negative"]["topic_clean"]
    top_negative_topic = negative_topics.value_counts().idxmax() if not negative_topics.empty else "None"

    top_platform = df["platform"].value_counts().idxmax() if not df["platform"].empty else "N/A"

    return (
        f"Out of {total_mentions} total mentions, {positive} are positive, "
        f"{neutral} are neutral, and {negative} are negative. "
        f"The most discussed topic is '{top_topic}'. "
        f"The main negative concern appears to be '{top_negative_topic}'. "
        f"The most active platform in the selected view is '{top_platform}'."
    )


def build_wordcloud(text_series):
    text = " ".join(text_series.dropna().astype(str).tolist()).strip()
    if not text:
        return None

    custom_stopwords = STOPWORDS.union({
        "netflix", "really", "just", "very", "too", "one", "get",
        "got", "also", "still", "even", "would", "could", "month",
        "today", "recently", "actually", "honestly"
    })

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        stopwords=custom_stopwords,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


st.title("Automated Social Media Sentiment Intelligence Pipeline")

try:
    df = load_data()

    if df.empty:
        st.warning("No data found. Run the pipeline first.")
    else:
        # Basic cleaning
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "topic" not in df.columns:
            df["topic"] = "Other"

        if "topic_keywords" not in df.columns:
            df["topic_keywords"] = ""

        # Clean topic labels for display
        df["topic_clean"] = df["topic"].apply(clean_topic_label)

        # Sidebar filters
        st.sidebar.header("Filters")

        platform_options = sorted(df["platform"].dropna().unique().tolist())
        topic_options = sorted(df["topic_clean"].dropna().unique().tolist())

        platform_filter = st.sidebar.multiselect(
            "Select Platform",
            options=platform_options,
            default=platform_options
        )

        topic_filter = st.sidebar.multiselect(
            "Select Topic",
            options=topic_options,
            default=topic_options
        )

        filtered_df = df.copy()

        if platform_filter:
            filtered_df = filtered_df[filtered_df["platform"].isin(platform_filter)]
        else:
            filtered_df = filtered_df.iloc[0:0]

        if topic_filter:
            filtered_df = filtered_df[filtered_df["topic_clean"].isin(topic_filter)]
        else:
            filtered_df = filtered_df.iloc[0:0]

        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
        else:
            # KPIs
            total_mentions = len(filtered_df)
            positive_pct = (filtered_df["sentiment_label"] == "Positive").mean() * 100 if len(filtered_df) else 0
            neutral_pct = (filtered_df["sentiment_label"] == "Neutral").mean() * 100 if len(filtered_df) else 0
            negative_pct = (filtered_df["sentiment_label"] == "Negative").mean() * 100 if len(filtered_df) else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Mentions", total_mentions)
            col2.metric("Positive %", f"{positive_pct:.1f}%")
            col3.metric("Neutral %", f"{neutral_pct:.1f}%")
            col4.metric("Negative %", f"{negative_pct:.1f}%")

            st.divider()

            # Summary
            st.subheader("AI-Style Summary")
            st.info(generate_summary(filtered_df))

            st.divider()

            # Main charts
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Sentiment Distribution")
                sentiment_counts = filtered_df["sentiment_label"].value_counts()
                st.bar_chart(sentiment_counts)

            with c2:
                st.subheader("Mentions by Platform")
                platform_counts = filtered_df["platform"].value_counts()
                st.bar_chart(platform_counts)

            st.divider()

            c3, c4 = st.columns(2)

            with c3:
                st.subheader("Topic Distribution")
                topic_counts = filtered_df["topic_clean"].value_counts()
                st.bar_chart(topic_counts)

            with c4:
                st.subheader("Sentiment Trend Over Time")
                trend_df = (
                    filtered_df.groupby(["date", "sentiment_label"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                )
                st.line_chart(trend_df)

            st.divider()

            # Word cloud
            st.subheader("Word Cloud")
            wc_fig = build_wordcloud(filtered_df["cleaned_text"])
            if wc_fig is not None:
                st.pyplot(wc_fig)
            else:
                st.info("No text available for word cloud.")

            st.divider()

            # Heatmap table
            st.subheader("Platform × Sentiment Heatmap")
            heatmap_df = pd.crosstab(filtered_df["platform"], filtered_df["sentiment_label"])

            if not heatmap_df.empty:
                st.dataframe(
                    heatmap_df.style.background_gradient(cmap="Blues"),
                    use_container_width=True
                )
            else:
                st.info("No data available for heatmap.")

            st.divider()

            # Negative mentions
            st.subheader("Top Negative Mentions")
            negative_posts = filtered_df[filtered_df["sentiment_label"] == "Negative"].copy()

            if negative_posts.empty:
                st.info("No negative mentions found.")
            else:
                negative_posts = negative_posts.sort_values(by="sentiment_score", ascending=True)
                st.dataframe(
                    negative_posts[
                        ["date", "platform", "topic_clean", "text", "sentiment_score", "engagement"]
                    ].rename(columns={"topic_clean": "topic"}),
                    use_container_width=True
                )

            st.divider()

            # Full records
            st.subheader("All Mentions")
            st.dataframe(
                filtered_df[
                    [
                        "date",
                        "platform",
                        "brand",
                        "topic_clean",
                        "topic_keywords",
                        "text",
                        "sentiment_label",
                        "sentiment_score",
                        "engagement",
                    ]
                ]
                .rename(columns={"topic_clean": "topic"})
                .sort_values(by="date", ascending=False),
                use_container_width=True
            )

except Exception as e:
    st.error(f"Error loading dashboard: {e}")