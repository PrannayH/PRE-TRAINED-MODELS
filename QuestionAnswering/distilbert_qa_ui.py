import gradio as gr
from transformers import pipeline

# Load the question-answering pipeline with 'distilbert-base-cased-distilled-squad' model
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased')

def answer_question(context, question=""):
    # Default question if left empty
    if not question:
        question = "What is the main idea?"

    # Perform question answering
    result = qa_pipeline(context=context, question=question)
    return result['answer']

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(placeholder="Enter context...", label="Context"),
        gr.Textbox(placeholder="Enter question...", label="Question")
    ],
    outputs=gr.Textbox(placeholder="Answer will be displayed here...", label="Answer"),
    live=True,
    title="DistilBERT Question Answering",
    description="Enter a context and a question, then click Submit to get the answer.",
)

iface.launch()
