import gradio as gr
from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')

def predict_highest_scoring_word(sentence):
    sentence = sentence.replace('_', '[MASK]')
    
    print(sentence)

    result = unmasker(sentence)

    highest_scoring_token = result[0]['token_str']

    return highest_scoring_token

iface = gr.Interface(
    fn=predict_highest_scoring_word,
    inputs=gr.Textbox(placeholder="Enter a sentence with a placeholder (e.g., 'I am _ hungry')"),
    outputs="text",
    live=True,
    title="BERT Highest Scoring Word Prediction",
    description="Enter a sentence with a placeholder (_), and the model will predict the word with the highest score.",
)

iface.launch()