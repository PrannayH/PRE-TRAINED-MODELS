import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the NER pipeline with 'StivenLancheros/biobert-base-cased-v1.2-finetuned-ner-CRAFT_Augmented_EN' model
ner_tokenizer = AutoTokenizer.from_pretrained("StivenLancheros/biobert-base-cased-v1.2-finetuned-ner-CRAFT_Augmented_EN")
ner_model = AutoModelForTokenClassification.from_pretrained("StivenLancheros/biobert-base-cased-v1.2-finetuned-ner-CRAFT_Augmented_EN")
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
    title="BioBERT Named Entity Recognition (Fine-tuned)",
    description="Enter a text, then click Submit to get the Named Entity Recognition results using the fine-tuned BioBERT model.",
)

iface.launch()
