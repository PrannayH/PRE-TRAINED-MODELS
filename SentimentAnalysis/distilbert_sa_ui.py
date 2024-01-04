import gradio as gr
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification

# Load the DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Define the sentiment analysis function
def analyze_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Perform sentiment analysis
    outputs = model(**inputs)
    logits = outputs.logits

    # Predict sentiment label
    predicted_class = logits.argmax().item()

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
    title="DistilBERT Sentiment Analysis",
    description="Enter a text, then click Submit to get the sentiment analysis result.",
)

iface.launch()
