import pandas as pd

def concatenate_replies(df):
    """
    Concatenates replies into their parent comment with [COMMENT] and [REPLY] tokens.
    """
    df["comment_train"] = "[COMMENT] " + df["comment"]  # Start with parent comment prefix

    for index, row in df.iterrows():
        if pd.isna(row["parent_id"]):  # This is a parent comment
            reply_count = row["reply_count"]
            if reply_count:
                for i in range(index + 1, index + reply_count + 1):
                    # Add [REPLY] prefix for each reply
                    reply_text = f" [REPLY] {df.at[i, 'comment']}"
                    df.at[index, "comment_train"] += reply_text  # Append to parent comment

    # Keep only top-level comments
    return df[df["parent_id"].isna()]

# Loop through video IDs and process comments
video_ids = pd.read_csv("data/youtube_video_ids.csv")

for video_id in video_ids["video_id"]:
    # Load comments for each video
    df = pd.read_csv(f"data/comments_v2/{video_id}_comments.csv")
    
    # Concatenate replies with tokens
    df = concatenate_replies(df)
    
    # Save updated dataset
    df.to_csv(f"data/processed_comments/{video_id}_concatenated.csv", index=False)