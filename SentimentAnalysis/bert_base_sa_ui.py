import gradio as gr
from transformers import pipeline

# Load the sentiment analysis pipeline with 'bert-base' model
sentiment_pipeline = pipeline('sentiment-analysis', model='bert-base-uncased', tokenizer='bert-base-uncased')

def analyze_sentiment(text):
    # Perform sentiment analysis
    result = sentiment_pipeline(text)
    sentiment_label = result[0]['label']
    
    # Map labels to more understandable terms
    if sentiment_label == 'LABEL_0':
        sentiment = 'Negative'
    elif sentiment_label == 'LABEL_1':
        sentiment = 'Positive'
    else:
        sentiment = 'Neutral'  # You can add more cases if needed
    
    return sentiment

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(placeholder="Enter text...", label="Text"),
    outputs=gr.Textbox(placeholder="Sentiment will be displayed here...", label="Sentiment"),
    live=True,
    title="BERT Sentiment Analysis",
    description="Enter a text, then click Submit to get the sentiment analysis result.",
)

iface.launch()
