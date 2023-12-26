# BERT Highest Scoring Word Prediction

This project utilizes the Hugging Face Transformers library and Gradio to predict the word with the highest score in a sentence with a placeholder. The model used is BERT (Bidirectional Encoder Representations from Transformers) with the 'bert-base-uncased' configuration.

## Usage

1. **Install the required libraries:**

   ```bash
   pip install gradio transformers

2. **Run the provided script:**
   ```bash
   python3 script_name.py

3. **Open the provided Gradio interface:**
A Gradio interface will be launched, allowing you to input sentences with placeholders.
The model will predict the word with the highest score in the given context.

## Example
For instance, if you input the sentence "I am _ hungry," the model will predict the word that best fits in the placeholder.

## Dependencies
1. Gradio
2. Transformers

## Acknowledgements
1. HuggingFaceTransformers: https://huggingface.co/bert-base-uncased
2. Gradio: https://www.gradio.app/docs/interface
