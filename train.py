'''
Script to fine tune BERT-based Q&A 
Steps:
    1. Download SQuAD v2.0 dataset
    2. Data preparation: extraction, encoding, initialization
    3. Fine-tuning
    4. Model evaluation
'''

import json
# package declaration
import os

import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AdamW, AutoModelForQuestionAnswering,
                          DistilBertTokenizerFast, pipeline)
from transformers.data.processors import squad
from transformers import DistilBertForQuestionAnswering
from transformers import Trainer, TrainingArguments

# config 
PATH = 'data/benchmarks/squad'


def download_squad(path=PATH):

    # create squad folder if needed
    if not os.path.exists(path):
        os.makedirs(path)

    url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
    # loop through
    for file in ['train-v2.0.json', 'dev-v2.0.json']:
        # make the request to download data over HTTP
        res = requests.get(f'{url}{file}')
        # write to file
        with open(f'{path}{file}', 'wb') as f:
            for chunk in res.iter_content(chunk_size=4):
                f.write(chunk)


def read_squad(path):
    # open JSON file and load intro dictionary
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                # check if we need to be extracting from 'answers' or 'plausible_answers'
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa[access]:
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers


def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer['answer_end'] = end_idx
        else:
            # this means the answer is off by 1-2 tokens
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n



def add_token_positions(encodings, answers, tokenizer):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def main(path, read_squad, add_end_idx):
    
    ##############################################
    ### Data preparation ###
    ##############################################
    # download SQuAD v2 dataset
    # download_squad()

    # execute our read SQuAD function for training and validation sets
    train_contexts, train_questions, train_answers = \
                read_squad(f'{path}/train-v2.0.json')
    val_contexts, val_questions, val_answers = \
                read_squad(f'{path}/dev-v2.0.json')

    # print one example
    print(train_contexts[0])
    print(train_questions[0])
    print(train_answers[0])

    # and apply the function to our two answer lists
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    print(train_answers[:3])

    # tokenization 
    tokenizer = DistilBertTokenizerFast.from_pretrained(\
        'distilbert-base-uncased')
    train_encodings = tokenizer(train_contexts, train_questions,\
        truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, \
        truncation=True, padding=True)
    
    # print(tokenizer.decode(train_encodings['input_ids'][0]))

    # apply function to our data
    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(val_encodings, val_answers, tokenizer)

    print(train_encodings.keys())

    # build datasets for both our training and validation sets
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)


    ##############################################
    ### Fine tuning ###
    ##############################################
    
    
    # training_args = TrainingArguments(
    #     output_dir='./results',          # output directory
    #     num_train_epochs=3,              # total number of training epochs
    #     per_device_train_batch_size=16,  # batch size per device during training
    #     per_device_eval_batch_size=64,   # batch size for evaluation
    #     warmup_steps=500,                # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,               # strength of weight decay
    #     logging_dir='./logs',            # directory for storing logs
    #     logging_steps=10,
    # )
    # build the model 
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    
    # trainer = Trainer(
    #     model=model,                         # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,                  # training arguments, defined above
    #     train_dataset=train_dataset,         # training dataset
    #     eval_dataset=val_dataset             # evaluation dataset
    # )

    # trainer.train()

    # # # set up GPU/ CPU
    # device = torch.device('cuda') if torch.cuda.is_available()\
    #     else torch.device('cpu')

    # # move model to the device
    # model.to(device)
    # # activate training mode of model
    # model.train()
    # # initialize adam optimizer with weight decay (reduces chance of overfitting)
    # optim = AdamW(model.parameters(), lr=5e-5)

    # # initialize data loader for training data
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # for epoch in range(3):
    #     # # set model to train mode
    #     # model.train()
    #     # setup loop (we use tqdm for the progress bar)
    #     loop = tqdm(train_loader, leave=True)
    #     for batch in loop:
    #         # initialize calculated gradients (from prev step)
    #         optim.zero_grad()
    #         # pull all the tensor batches required for training
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         start_positions = batch['start_positions'].to(device)
    #         end_positions = batch['end_positions'].to(device)
    #         # train model on batch and return outputs (incl. loss)
    #         outputs = model(input_ids, attention_mask=attention_mask,
    #                         start_positions=start_positions,
    #                         end_positions=end_positions)
    #         # extract loss
    #         loss = outputs[0]
    #         # calculate loss for every parameter that needs grad update
    #         loss.backward()
    #         # update parameters
    #         optim.step()
    #         # print relevant info to progress bar
    #         loop.set_description(f'Epoch {epoch}')
    #         loop.set_postfix(loss=loss.item())
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.eval()


    model_path = 'models/distilbert-custom'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # evaluation
    # switch model out of training mode
    model.eval()
    # initialize validation set data loader
    val_loader = DataLoader(val_dataset, batch_size=16)
    # initialize list to store accuracies
    acc = []
    # loop through batches
    for batch in val_loader:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # we will use true positions for accuracy calc
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull prediction tensors out and argmax to get predicted tokens
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            # calculate accuracy for both and append to accuracy list
            acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
    # calculate average accuracy in total
    acc = sum(acc)/len(acc)
    print('\n=============\n')
    print('Average accuracy (Exact Match ', acc)

    print('T/F \tstart \tend')
    for i in range(len(start_true)):
        print(
            f'true \t{start_true[i]} \t{end_true[i]}'
            f'pred \t{start_pred[i]} \t{end_pred[i]}'
        )

if __name__=='__main__':

    main(PATH, read_squad, add_end_idx)

    pass
