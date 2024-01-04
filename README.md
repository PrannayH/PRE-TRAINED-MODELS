# Pre-Trained models 

## TASKS
1. Masked Word Prediction
2. Question Answering
3. Sentiment Analysis
4. Named Entity Recognition
   
## Usage

1. **Install the required libraries:**

   ```bash
   pip install gradio transformers

2. **Run the provided script:**
   ```bash
   python3 script_name.py

3. **Open the provided Gradio interface:**
A Gradio interface will be launched, allowing you to input sentences with placeholders.
The model will perform the respective task using the model API given.
## Example
1. For masked word prediction, if you input the sentence "I am _ hungry," the models will predict the word that best fits in the placeholder.   
2. For question answering, if you give context and question like: 
   Context:
   "The Eiffel Tower is a famous landmark in Paris, France. It was completed in 1889 and has become a symbol of the city."
   Question:
   "What is the Eiffel Tower?"
   It gives an appropriate answer.
3. For Sentiment Analysis, for a given input statement it returns either POSITIVE or NEGATIVE or NEUTRAL label depending on the sentiment of the statement.
4. For named entity recognition, for a given input statement it returns the labels like LOC,PER,ORG,MISC etc depending on the model beig used and thier respective     classes. 


## Dependencies
1. Gradio
2. Transformers
   
## Acknowledgements
1. bert-base-uncased
2. dmis-lab/biobert-large-cased-v1.1-squad
3. roberta-base
4. distilbert-base-cased-distilled-squad
5. VMware/minilmv2-l12-h384-from-roberta-large-mrqa
6. xlnet-base-cased
7. Gradio: https://www.gradio.app/docs/interface
