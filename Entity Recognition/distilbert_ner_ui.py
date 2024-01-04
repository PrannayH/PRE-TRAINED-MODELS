import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the NER pipeline with 'HooshvareLab/distilbert-fa-zwnj-base-ner' model
ner_tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/distilbert-fa-zwnj-base-ner")
ner_model = AutoModelForTokenClassification.from_pretrained("HooshvareLab/distilbert-fa-zwnj-base-ner")
ner_nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

def perform_ner(context):
    # Perform NER on the context
    ner_result = ner_nlp(context)
    formatted_result = "\n".join([f"{entity['word']}: {entity['entity']}" for entity in ner_result])
    return formatted_result

iface = gr.Interface(
    fn=perform_ner,
    inputs=gr.Textbox(placeholder="Enter text...", label="Text"),
    outputs=gr.Textbox(placeholder="NER results will be displayed here...", label="NER Results", type="text"),
    live=True,
    title="DistilBERT Named Entity Recognition",
    description="Enter text, then click Submit to get the Named Entity Recognition results.",
)

iface.launch()
