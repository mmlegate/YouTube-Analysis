import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

metadata_path = "data/video_metadata.csv"
comments_path = "data/comments_v2/"

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Load your comments dataset
df = pd.read_csv("comments.csv")  # Make sure it has 'video_id' and 'comment'

# Apply VADER sentiment analysis
df["vader_score"] = df["comment"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

# Define sentiment categories based on VADER score
def classify_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_label"] = df["vader_score"].apply(classify_sentiment)

# Group by video_id to get average sentiment per video
video_sentiment = df.groupby("video_id")["vader_score"].mean().reset_index()
video_sentiment.rename(columns={"vader_score": "video_positivity_score"}, inplace=True)

# Save both datasets
df.to_csv("labeled_comments.csv", index=False)  # Comment-level data
video_sentiment.to_csv("video_sentiment_scores.csv", index=False)  # Video-level data

print(df.head())  # Check comment-level sentiment
print(video_sentiment.head())  # Check video-level sentiment
