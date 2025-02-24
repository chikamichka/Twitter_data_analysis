from sentiment import predict_sentiment
import csv

def read_tweets(file_path: str) -> list[dict[str, str]]:
    """Reads tweets from a CSV file and returns them as a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)

# Read tweets from the CSV file
tweets = read_tweets("./tweets.csv")
print(f"Loaded {len(tweets)} tweets.")

# Open a new CSV file to store results
output_file = "./tweets_with_sentiment.csv"
with open(output_file, "w", encoding="utf-8", newline="") as file:
    fieldnames = ["Tweet_count", "Text", "Category", "Sentiment"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header

    # Process each tweet and write results
    for index, tweet in enumerate(tweets, start=1):
        tweet_count = tweet["Tweet_count"]  # Unique identifier
        tweet_text = tweet["Text"]  # Tweet content
        category = tweet.get("Category", "Unknown")  # Get Category (default to "Unknown" if missing)

        # Predict sentiment
        sentiment = predict_sentiment(tweet_text)

        # Save result to file
        writer.writerow({"Tweet_count": tweet_count, "Text": tweet_text, "Category": category, "Sentiment": sentiment})

        # Print progress every 100 tweets
        if index % 100 == 0 or index == len(tweets):
            print(f"Processed {index}/{len(tweets)} tweets...")

print(f"Sentiment analysis completed. Results saved in {output_file}.")
