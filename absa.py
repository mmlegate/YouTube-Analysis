from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bertopic import BERTopic
import pandas as pd

def infer_aspect(video_id):
    df = pd.read_csv(f"data/comments_v2/{video_id}_comments")

    # Extract aspects using BERTopic
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(df["comment"].tolist())

    # Assign inferred aspects to dataset
    df["aspect"] = topics
    df.to_csv("comments_with_aspects.csv", index=False)

    print(df.head())  # Check inferred aspects
    return df

# Load DeBERTa ABSA model
MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def infer_aspect_sentiment(comment, aspect):
    """
    Predicts sentiment score (-1: Negative, 0: Neutral, 1: Positive)
    for a given comment and inferred aspect.
    """
    input_text = f"{aspect} [SEP] {comment}"  # Aspect before comment
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        logits = model(**inputs).logits

    sentiment_class = torch.argmax(logits).item()
    sentiment_score = sentiment_class - 1  # (-1 = Negative, 0 = Neutral, 1 = Positive)

    return sentiment_score

# Apply ABSA to each comment using inferred aspects
df["sentiment"] = df.apply(lambda row: infer_aspect_sentiment(row["comment"], row["aspect"]), axis=1)

# Save results
df.to_csv("comments_with_aspects_and_sentiments.csv", index=False)

print(df.head())

# Group by video and aspect, then average sentiment scores
video_aspect_sentiment = df.groupby(["video_id", "aspect"])["sentiment"].mean().reset_index()

# Save the summary
video_aspect_sentiment.to_csv("video_aspect_sentiment.csv", index=False)

print(video_aspect_sentiment.head())  # Check summary
