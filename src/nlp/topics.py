def detect_topic(text: str):

    if not isinstance(text, str):
        return {"topic": "Other", "topic_keywords": ""}

    text_lower = text.lower()

    topic_map = {
        "Pricing": ["price", "pricing", "subscription", "cost", "expensive"],
        "Content Quality": ["show", "series", "movie", "documentary", "content"],
        "Cancellation Issues": ["cancel", "cancelled", "remove", "removed"],
        "Recommendations": ["recommendation", "recommendations", "suggest"],
        "User Experience": ["ui", "interface", "app", "smooth"],
        "Customer Support": ["support", "helpful", "service"],
    }

    for topic, keywords in topic_map.items():
        for keyword in keywords:
            if keyword in text_lower:
                return {
                    "topic": topic,
                    "topic_keywords": keyword
                }

    return {
        "topic": "Other",
        "topic_keywords": ""
    }