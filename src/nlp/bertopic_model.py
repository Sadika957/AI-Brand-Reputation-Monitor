from bertopic import BERTopic
import pandas as pd


def clean_topic_words(words, max_words=4):
    """
    Clean BERTopic words and return a readable topic label.
    """
    noisy_words = {
        "netflix", "the", "is", "too", "very", "really", "just",
        "this", "that", "and", "for", "with", "what", "have",
        "has", "had", "been", "are", "was", "were", "will",
        "would", "could", "should", "can", "all", "some", "more"
    }

    cleaned = []
    for word, _ in words:
        word = str(word).strip().lower()
        if word and word not in noisy_words and word not in cleaned:
            cleaned.append(word)

    if not cleaned:
        return "Other", ""

    label_words = cleaned[:max_words]
    topic_label = ", ".join(label_words).title()
    topic_keywords = ", ".join(label_words)

    return topic_label, topic_keywords


def extract_topics_bertopic(texts):
    # Keep original order and handle blanks safely
    processed_texts = [str(t).strip() if pd.notna(t) else "" for t in texts]

    valid_mask = [bool(t) for t in processed_texts]
    valid_texts = [t for t in processed_texts if t]

    # Fallback for very small datasets
    if len(valid_texts) < 5:
        result_rows = []
        for text in processed_texts:
            if text:
                result_rows.append({
                    "topic_id": -1,
                    "topic": "Other",
                    "topic_keywords": ""
                })
            else:
                result_rows.append({
                    "topic_id": -1,
                    "topic": "Other",
                    "topic_keywords": ""
                })
        return pd.DataFrame(result_rows)

    topic_model = BERTopic(
        language="english",
        verbose=False,
        calculate_probabilities=False
    )

    topics, _ = topic_model.fit_transform(valid_texts)

    # Build clean mappings
    topic_label_map = {}
    topic_keyword_map = {}

    unique_topics = sorted(set(topics))

    for topic_id in unique_topics:
        if topic_id == -1:
            topic_label_map[topic_id] = "Other"
            topic_keyword_map[topic_id] = ""
            continue

        words = topic_model.get_topic(topic_id)

        if words:
            topic_label, topic_keywords = clean_topic_words(words, max_words=4)
            topic_label_map[topic_id] = topic_label
            topic_keyword_map[topic_id] = topic_keywords
        else:
            topic_label_map[topic_id] = "Other"
            topic_keyword_map[topic_id] = ""

    # Rebuild output in original row order
    result_rows = []
    valid_index = 0

    for is_valid in valid_mask:
        if is_valid:
            topic_id = topics[valid_index]
            result_rows.append({
                "topic_id": int(topic_id),
                "topic": topic_label_map.get(topic_id, "Other"),
                "topic_keywords": topic_keyword_map.get(topic_id, "")
            })
            valid_index += 1
        else:
            result_rows.append({
                "topic_id": -1,
                "topic": "Other",
                "topic_keywords": ""
            })

    return pd.DataFrame(result_rows)