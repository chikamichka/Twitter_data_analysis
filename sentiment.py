from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"  # Updated model
ROBERTA_SUPPORTED_LANGUAGES = ('ar', 'en', 'fr', 'de', 'hi', 'it', 'es', 'pt')

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# Save the model locally (if necessary)
# model.save_pretrained(MODEL)
# tokenizer.save_pretrained(MODEL)

# Preprocess text to handle mentions and URLs
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def predict_sentiment(text: str) -> str:
    processed_text = preprocess(text)
    encoded_input = tokenizer(processed_text, return_tensors='pt')
    output = model(**encoded_input)
    index_of_sentiment = output.logits.argmax().item()

    # Mapping the numeric label to human-readable sentiment labels
    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}

    sentiment = sentiment_labels.get(index_of_sentiment, "unknown")  # Default to "unknown" if invalid
    return sentiment
