import gradio as gr
from transformers import pipeline

# Creating a fill-mask pipeline with the specified RoBERTa model
unmasker = pipeline('fill-mask', model='roberta-base')  # Change the model name here

# Define the function for predicting the word with the highest score
def predict_highest_scoring_word(sentence):
    # Replace placeholder (_) with <mask> for RoBERTa
    sentence = sentence.replace('_', '<mask>')
    print(sentence)
    
    # Using the pipeline to fill in the masked token
    result = unmasker(sentence)

    # Extracting the token with the highest score
    highest_scoring_token = result[0]['token_str']

    return highest_scoring_token

# Create Gradio interface
iface = gr.Interface(
    fn=predict_highest_scoring_word,
    inputs=gr.Textbox(placeholder="Enter a sentence with a placeholder (e.g., 'I am _ hungry')"),
    outputs="text",
    live=True,
    title="RoBERTa Highest Scoring Word Prediction",  # Update the title
    description="Enter a sentence with a placeholder (_), and the model will predict the word with the highest score using RoBERTa.",
)

# Launch the Gradio interface
iface.launch()
