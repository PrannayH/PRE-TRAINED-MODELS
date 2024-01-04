biobert


import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Replace 'dmis-lab/biobert-v1.1' with the correct model identifier
model_name = "dmis-lab/biobert-v1.1"

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Define the sentiment analysis function
def analyze_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Perform sentiment analysis
    outputs = model(**inputs)
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)

    # Set a threshold (e.g., 0.5) for positive sentiment
    threshold = 0.5
    predicted_class = (probabilities[:, 1] > threshold).int().item()

    # Map labels to more understandable terms
    if predicted_class == 0:
        sentiment = 'Negative'
    elif predicted_class == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'  # You can add more cases if needed

    return sentiment

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder="Enter text...", label="Text"),
    outputs=gr.Textbox(placeholder="Sentiment will be displayed here...", label="Sentiment"),
    live=True,
    title="BioBERT Sentiment Analysis",
    description="Enter a text, then click Submit to get the sentiment analysis result.",
)

iface.launch()
