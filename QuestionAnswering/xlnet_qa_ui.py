import gradio as gr
from transformers import pipeline

question_answerer = pipeline("question-answering", model='xlnet-base-cased')

def answer_question(context, question=""):
    # Default question if left empty
    if not question:
        question = "What is the main idea?"

    # Perform question answering
    result = question_answerer(question=question, context=context)
    
    answer = result['answer']

    return answer

iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(placeholder="Enter context...", label="Context"),
        gr.Textbox(placeholder="Enter question...", label="Question")
    ],
    outputs=gr.Textbox(placeholder="Answer will be displayed here...", label="Answer"),
    live=True,
    title="XLNet Question Answering",
    description="Enter a context and a question, then click Submit to get the answer.",
)

iface.launch()
