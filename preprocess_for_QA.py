import evaluation
import t5small
import torch
import argparse
import pickle
import os
from utils.data_utils import CoQADataset, CustomDataLoader
import json
import re
from utils import *

def get_arguments():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('model_path', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    return args

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def simpleTokenizer(text):
    tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False
    return tokens



if __name__ == '__main__':
    args = get_arguments()
    with open(args.input,'rb') as f:
        dev = pickle.load(f)

    data = {'input' : [], 'context' : [], 'references' : []}
    for x in dev.dataset:
        s = re.split(' *</s> *', ' '.join(x['tokens']))
        data['input'].append(re.split(' *<A> *| *<Q> *', s[0])[1:])
        data['context'].append(s[1])
        data['references'].append([''])
    

    model = torch.load(args.model_path) 
    predictions = evaluation.generate_predictions(
            model = model,
            data = data,
            dataset_maker = t5small.MODEL_DIC['dataset_maker'],
            output_tokenizer = t5small.MODEL_DIC['output_tokenizer_getter'](),
            hparams = {
                'include_story' : False,
                'history_size' : 20,
                'batch_size' : 16
            })

    for row, pred in zip(dev.dataset, predictions):
        row['coqaqr_model_paraphrase'] = simpleTokenizer(pred)

    with open(args.output, 'wb') as f:
        pickle.dump(dev, f)
    
    with open(args.output + '.json', 'w') as f:
        for row in dev.dataset:
            f.write(str(row) + '\n')
