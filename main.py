import csv
from sentiment import predict_sentiment

# Function to read the CSV file and return a list of tweets/replies
def read_replies(file_path: str) -> list[dict[str, str]]:
    """Reads replies from a CSV file and returns them as a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)

# Function to save replies with sentiment to a new CSV file
def save_replies_with_sentiment(replies: list[dict[str, str]], output_file: str):
    """Save replies along with sentiment analysis to a new CSV file."""
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        fieldnames = ["Tweet Text", "Category", "Username", "Reply Text", "Reply Likes", "Reply Reposts", "Reply Sentiment"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row

        # Process each reply and add sentiment analysis
        for reply in replies:
            tweet_text = reply["Tweet Text"]
            category = reply["Category"]
            username = reply["Username"]
            reply_text = reply["Reply Text"]

            # Analyze sentiment of the reply text
            reply_sentiment = predict_sentiment(reply_text)

            # Save the processed data in the new file
            writer.writerow({
                "Tweet Text": tweet_text,
                "Category": category,
                "Username": username,
                "Reply Text": reply_text,
                "Reply Likes": reply["Reply Likes"],
                "Reply Reposts": reply["Reply Reposts"],
                "Reply Sentiment": reply_sentiment
            })

# Main function to read replies, analyze sentiment, and save results
def main():
    # Read the replies from the CSV file
    replies = read_replies("replies.csv")
    print(f"Loaded {len(replies)} replies.")

    # Output file path
    output_file = "replies_with_sentiment.csv"

    # Save the replies with sentiment analysis
    save_replies_with_sentiment(replies, output_file)

    print(f"Sentiment analysis completed. Results saved in {output_file}.")

# Run the main function
if __name__ == "__main__":
    main()
