from utils import *
import torch
import config
from torch.utils.data import Dataset
import pandas as pd
import random
from nltk.tokenize import RegexpTokenizer
import statistics

def word_count(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def distrib(l, normalize = False):
    if normalize:
        total = len(l)
        return {
            i : round(len([j for j in l if j == i]) / total, 3)
            for i in range(min(l), max(l) + 1)
        }
    else:
        return {
            i : len([j for j in l if j == i])
            for i in range(min(l), max(l) + 1)
        }

def coqar_statistics():
    path = config.COQAR_DATA_PATH
    raw_data = \
        json_get(os.path.join(config.COQAR_DATA_PATH, 'dev/coqar-dev-v1.0.json'))['data'] + \
        json_get(os.path.join(path,'train/coqar-train-v1.0.json'))['data']
               
    nb_passages = 0
    dialogue_lengths = []
    passage_lengths = []
    nb_paraphrases = []
    question_lengths = []
    paraphrase_lengths = []
    nb_questions = 0
    nb_unknown = 0
    for dialogue in raw_data:
        length = 0
        nb_passages += 1
        passage_lengths.append(word_count(dialogue['story']))
        for (questions, answer) in zip(dialogue['questions'], dialogue['answers']):
            length += 1
            if word_count(questions['input_text']) != 0:
                if answer['input_text'] == 'unknown':
                    nb_unknown += 1

                nb_questions += 1
                question_lengths.append(word_count(questions['input_text']))
                nb_paraphrases.append(len([x for x in questions['paraphrase'] if len(x) > 0]))
                for paraphrase in questions['paraphrase']:
                    if word_count(paraphrase) != 0:
                        paraphrase_lengths.append(word_count(paraphrase))
        dialogue_lengths.append(length)

    print(f'{nb_passages} dialogues')
    print(nb_questions, 'question')
    print('Dialogue length (# of questions) distribution : ' +' '.join(f'({x}, {y})' for x, y in distrib(dialogue_lengths).items()))
    print('Avg dialogue length:' ,statistics.mean(dialogue_lengths)) 
    print(f'Passage length (min, avg, max): ({min(passage_lengths)}, {statistics.mean(passage_lengths)}, {max(passage_lengths)})')
    print('# of paraphrases per question (distribution) : ' +' '.join(f'({x}, {y})' for x, y in distrib(nb_paraphrases).items()))
    print(f'Question length (min, avg, max): ({min(question_lengths)}, {statistics.mean(question_lengths)}, {max(question_lengths)})')
    print('question length (distribution) : ' +' '.join(f'({x}, {y})' for x, y in distrib(question_lengths, True).items()))
    print(f'Paraphrase length (min, avg, max): ({min(paraphrase_lengths)}, {statistics.mean(paraphrase_lengths)}, {max(paraphrase_lengths)})')
    print('paraphrase length (distribution) : ' +' '.join(f'({x}, {y})' for x, y in distrib(paraphrase_lengths, True).items()))
    print('# unknown:', nb_unknown/nb_questions)

'''
The function returns a dictionary with three keys: "input", "references", "context".
Each key is associated to a list:
    result["input"][i] is list of dialogue turns [question1, answer1, question2, answer2, ..., questionN]
    result["references"][i] is a list of human reformulations of result["input"][i][-1]
    result["context"][i] is the paragraph about which the dialogue result["input"][i] is
'''
def format_coqar_data(data, include_story=False):
    contexts, inputs, references, answers = [], [], [], []
    for dialogue in data:
        seq = []
        for (questions, answer) in zip(dialogue['questions'], dialogue['answers']):
            seq.append(questions['input_text'])
            for paraphrase in questions['paraphrase']:
                if len(paraphrase) > 0:
                    if len(inputs) == 0 or seq[-1] != inputs[-1][-1]:
                        # if seq has not been added yet as input
                        inputs.append(list(seq))
                        contexts.append(dialogue['story'] if include_story else '')
                        references.append([paraphrase])
                        answers.append([answer['span_text'], answer['input_text']])
                    else:
                        references[-1].append(paraphrase)
            seq.append(answer['input_text'])
    return {'input' : inputs, 'references' : references, 'context' : contexts, 'answer_spans' : answers}

'''Same as format_coqar_data but for canard'''
def format_CANARD_data(data, include_story):
    quac = json_get(os.path.join(config.QUAC_DATA_PATH, 'train_v0.2.json'))['data'] + \
            json_get(os.path.join(config.QUAC_DATA_PATH, 'val_v0.2.json'))['data']
    
    if include_story:
        context_dic = {
            x['paragraphs'][0]['id'] : x['paragraphs'][0]['context']
            for x in quac
        }
    else:
        context_dic = {
            x['paragraphs'][0]['id'] : ''
            for x in quac
        }

    for x in quac:
        assert len(x['paragraphs']) == 1
    
    answer_dic = {
            x['paragraphs'][0]['id'] : [
                [a['text'] for a in qa['answers']] + [qa['orig_answer']['text']]
                for qa in x['paragraphs'][0]['qas']
            ]
            for x in quac
    }

    context, input, references, answers = [], [], [], []
    for dic in data:
        context.append(context_dic[dic['QuAC_dialog_id']])
        input.append(dic['History'][2:] + [dic['Question']])
        references.append([dic['Rewrite']])
        answers.append(answer_dic[dic['QuAC_dialog_id']][int(len(dic['History'])/2) - 1])
    return {'input' : input, 'references' : references, 'context' : context, 'answer_spans' : answers}

'''Same as format_canard_data but with only first turn'''
def format_nocontext_data(data):

    context, input, references = [], [], []
    for dic in data:
        context.append('')
        if len(dic['History']) == 0:
            input.append(dic['History'] + [dic['Question']])
            references.append([dic['Rewrite']])
    return {'input' : input, 'references' : references, 'context' : context}




def split_data(data, i):
    return  {k: data[k][:-i] for k in ('input', 'references', 'context')}, \
            {k: data[k][-i:] for k in ('input', 'references', 'context')}

def shuffle_data(data, n = -1, remove_first_question = False):
    if remove_first_question:
        print(len(data['input']))
        z = [
            x for x in zip(data['input'], data['references'], data['context'], data['answer_spans'])
            if len(x[0]) > 1
        ]
        print(len(z))
    else:
        z = list(zip(data['input'], data['references'], data['context'], data['answer_spans']))
    if n == -1:
        n = len(data['input'])
    random.shuffle(z)
    z = z[:n]
    input, references, context, answers = list(zip(*z))
    return {'input' : input, 'references' : references, 'context' : context, 'answer_spans' : answers}


def get_coqar_original_train_set(include_story):
    path = config.COQAR_DATA_PATH
    raw_data = json_get(os.path.join(path,'train/coqar-train-v1.0.json'))['data']
    return format_coqar_data(raw_data, include_story)

def get_coqar_train_and_dev_sets(include_story):
    path = config.COQAR_DATA_PATH
    raw_data =  json_get(os.path.join(path,'train/coqar-train-v1.0.json'))['data']
    data = format_coqar_data(raw_data, include_story)
    i = len(get_canard_dev_set(include_story)['input'])
    return split_data(data, i)


def get_coqar_test_set(include_story):
    raw_data = json_get(os.path.join(config.COQAR_DATA_PATH,'dev/coqar-dev-v1.0.json'))['data']
    return format_coqar_data(raw_data, include_story)

def get_canard_train_set(include_story):
    raw_data = json_get(os.path.join(config.CANARD_DATA_PATH, 'train.json'))
    return format_CANARD_data(raw_data, include_story)

def get_canard_dev_set(include_story):
    raw_data = json_get(os.path.join(config.CANARD_DATA_PATH, 'dev.json'))
    return format_CANARD_data(raw_data, include_story)

def get_canard_test_set(include_story):
    raw_data = json_get(os.path.join(config.CANARD_DATA_PATH, 'test.json'))
    return format_CANARD_data(raw_data, include_story)

def get_canard_all_sets(include_story):
    raw_data = json_get(os.path.join(config.CANARD_DATA_PATH, 'test.json')) +\
            json_get(os.path.join(config.CANARD_DATA_PATH, 'dev.json')) +\
            json_get(os.path.join(config.CANARD_DATA_PATH, 'train.json'))
    return format_CANARD_data(raw_data, include_story)


'''
Returns a mix of both datasets (train and dev)
The mixed dev contains all rows of canard dev + as many rows from coqar train
The mixed train contains the remaining rows from coqar train + all rows from canard train
'''
def get_mixed_train_and_dev_sets(include_story):
    coqar_train = get_coqar_original_train_set(include_story)
    canard_train = get_canard_train_set(include_story)
    canard_dev = get_canard_dev_set(include_story)

    i = len(canard_dev['input'])

    train = {k : coqar_train[k][:-i] + canard_train[k]
             for k in ['input', 'references', 'context']}
    dev = {k: coqar_train[k][-i:] + canard_dev[k]
           for k in ['input', 'references', 'context']}

    return train, dev


def get_train_dev(dataset_name, include_story):
    if dataset_name == 'coqar':
        train, dev = get_coqar_train_and_dev_sets(include_story)
    elif dataset_name == 'canard':
        train = get_canard_train_set(include_story)
        dev = get_canard_dev_set(include_story)
    elif dataset_name == 'mixed':
        train, dev = get_mixed_train_and_dev_sets(include_story)
    return train, dev

def data_statistics():
    mixed_train, mixed_dev = get_mixed_train_and_dev_sets(False)
    coqar_train, coqar_dev = get_coqar_train_and_dev_sets(False)
    splits = {
        'COQAR train (minus dev)' : coqar_train,
        'COQAR train (part of original train)': coqar_dev,
        'COQAR test (originally dev)' : get_coqar_test_set(False),
        'CANARD train' : get_canard_train_set(True),
        'CANARD dev' : get_canard_dev_set(True),
        'CANARD test' : get_canard_test_set(True),
        'Mixed train' : mixed_train,
        'Mixed dev' : mixed_dev
    }
    index = list(splits.keys())
    nb_questions = [len(splits[k]['input']) for k in index]
    nb_rows = [sum(len(r) for r in splits[k]['references']) for k in index]

    frame = pd.DataFrame(data = {'# questions' : nb_questions, '# reformulations' : nb_rows}, index=index)

    return frame

def print_dataset_into_file(dataset, output_tokenizer, path, n = 20, predictions = None):
    decoded_data = []
    for i in range(n):
        row = dataset[i]
        decoded_row = [output_tokenizer.decode(row['labels'], skip_special_tokens=True)]
        if predictions != None:
            decoded_row.append(predictions[i])
        decoded_row.append(output_tokenizer.decode(row['input_ids'], skip_special_tokens=True))
        decoded_data.append(decoded_row)
    columns = ['output']
    if predictions != None:
        columns.append('prediction')
    columns.append('input')
    df = pd.DataFrame(data = decoded_data, columns = columns)
    with open(path, 'w') as out:
        out.write(df.to_csv(sep = '\t'))


'''
This class contains the tokenized data, ready to be fed to a transformer
'''
class QRDataset(Dataset):
    def __init__(self,
                 data,
                 input_tokenizer,
                 output_tokenizer,
                 include_story,
                 history_size,
                 cuda = False):

        X, Y = [], []
        for c, seq, refs in zip(data['context'], data['input'], data['references']):
            for y in refs:
                start = max(-len(seq), -history_size*2-1)
                assert start % 2 == 1
                x = ''.join(f'<question> {seq[i]} <answer> {seq[i+1]} '
                            for i in range(start, -2, 2)) + \
                    input_tokenizer.eos_token + ' ' + seq[-1]
                X.append(c + x if include_story else x)
                Y.append(y)
       
        X = input_tokenizer(X, padding=True, truncation=False)
        Y = output_tokenizer(Y, padding=True, truncation=False)

        self.rows = []
        if not cuda:
            for x, mask, y in zip(X['input_ids'], X['attention_mask'], Y['input_ids']):
                self.rows.append({'input_ids': torch.tensor(x[-512:]),
                                'attention_mask': torch.tensor(mask[-512:]),
                                'labels': torch.tensor(y)})
        else:
            for x, mask, y in zip(X['input_ids'], X['attention_mask'], Y['input_ids']):
                self.rows.append({'input_ids': torch.tensor(x[-512:]).cuda(),
                                'attention_mask': torch.tensor(mask[-512:]).cuda(),
                                'labels': torch.tensor(y).cuda()})
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


if __name__ == '__main__':
    print(data_statistics())
    print(coqar_statistics())
