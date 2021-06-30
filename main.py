'''
Q&A task using a pre-trained BERT base model.
Dataset: SQuAD 2.0

Steps:
1. Install HuggingFace's Transformers
    First install pytorch then,
    pip install transformers
2. Build a Q&A pipeline
    (- Data preprocessing --> not presented here
    - Context retrieval --> not presented here)
    - Initialization
    - Tokenizer
    - Pipeline
    - Prediction 
'''

# Package declaration
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline
import time

def main(question, context, nlp):
    
    results = nlp({
        'question': question,
        'context': context,
    })

    print(results)

def create_pipeline(model_name=None, device=0):
    if model_name is None:
        model_name = "deepset/bert-base-cased-squad2"
        print('\nUsing default model BERT for Q&A task. \n')

    # Get tokenizer and model from pretrained model name 
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Build pipepine
    nlp = pipeline(task='question-answering', 
                    model=model,
                    tokenizer=tokenizer,
                    )                    
    return nlp

if __name__=='__main__':
    start = time.time()
    # model = 'deepset/electra-base-squad2'
    model_name = "deepset/minilm-uncased-squad2"
    # model_name = "distilbert-base-cased-distilled-squad"
    # model_name = "deepset/bert-base-cased-squad2"

    nlp = create_pipeline(model_name=model_name, device=-1)
    
    end1 = time.time()    
    print('\nElapsed time to load model: {}s. \n'.format(end1 - start))
    
    context = "The Intergovernmental Panel on Climate Change (IPCC) is a scientific intergovernmental body under the auspices of the United Nations, set up at the request of member governments. It was first established in 1988 by two United Nations organizations, the World Meteorological Organization (WMO) and the United Nations Environment Programme (UNEP), and later endorsed by the United Nations General Assembly through Resolution 43/53. Membership of the IPCC is open to all members of the WMO and UNEP. The IPCC produces reports that support the United Nations Framework Convention on Climate Change (UNFCCC), which is the main international treaty on climate change. The ultimate objective of the UNFCCC is to \"stabilize greenhouse gas concentrations in the atmosphere at a level that would prevent dangerous anthropogenic [i.e., human-induced] interference with the climate system\". IPCC reports cover \"the scientific, technical and socio-economic information relevant to understanding the scientific basis of risk of human-induced climate change, its potential impacts and options for adaptation and mitigation.\""
    question = "What's the Intergovernmental Panel about?"
    
    n = 4
    for i in range(n):
        main(question, context, nlp)    
    
    end = time.time()
    print('\nElapsed time: {}s. \n'.format(end - end1))
    print('\nAverage elapsed time: {}s. \n'.format((end - end1)/n))
    pass 
