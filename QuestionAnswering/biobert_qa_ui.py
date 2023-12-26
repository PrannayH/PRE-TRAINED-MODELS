import gradio as gr
from transformers import pipeline

question_answerer = pipeline("question-answering", model='dmis-lab/biobert-large-cased-v1.1-squad', tokenizer='dmis-lab/biobert-large-cased-v1.1-squad')

def answer_question(context, question=""):
    # Default question if left empty
    if not question:
        question = "What is the main idea?"

    # Perform question answering
    result = question_answerer(question=question, context=context)
    
    answer = result['answer']

    return answer

# Example input
example_context = "BioBERT is a pre-trained biomedical language representation model. It has been fine-tuned on various biomedical NLP tasks, including the SQuAD dataset."

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(placeholder="Enter context...", label="Context", default=example_context),
        gr.Textbox(placeholder="Enter question...", label="Question")
    ],
    outputs=gr.Textbox(placeholder="Answer will be displayed here...", label="Answer"),
    live=True,
    title="BioBERT Question Answering",
    description="Enter a context and a question, then click Submit to get the answer.",
)

iface.launch()
