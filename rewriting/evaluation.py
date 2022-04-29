from utils import *
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
import random
import qrdatasets
import numpy
import nltk
from rouge_metric import PyRouge


def generate_predictions(
        model,
        data,
        dataset_maker,
        output_tokenizer,
        hparams,
        save_path = None):
    
    model.train(False)

    data = dict(data)
    full_dataset = dataset_maker(data, hparams)
    old_refs = list(data['references'])

    # create a dataset from data (without the references)
    data['references'] = [[''] for r in data['references']]
    dataset = dataset_maker(data, hparams, cuda = True)

    loader = DataLoader(dataset=dataset, batch_size=hparams['batch_size'])
    model.cuda()
    model.train(False)
    predictions = []
    for dic in loader:
        output = model.generate(input_ids = dic['input_ids'], attention_mask = dic['attention_mask'])
        pred = output_tokenizer.batch_decode(output, skip_special_tokens = True)
        predictions += pred
    
    if save_path != None:
        corrected_pred_list = []
        for p, r in zip(predictions, old_refs):
            for i in range(len(r)):
                corrected_pred_list.append(p)
        qrdatasets.print_dataset_into_file(full_dataset, output_tokenizer, save_path, predictions = corrected_pred_list)
 
    return predictions

def evaluate(predictions, references):

    split_predictions = [p.lower().split(' ') for p in predictions]
    split_references = [[r.lower().split(' ') for r in refs] for refs in references]
    
    bleu = nltk.translate.bleu_score.corpus_bleu(split_references, split_predictions)
   
    all_meteor_scores = []
    for pred, refs in zip(split_predictions, split_references):
        all_meteor_scores.append(nltk.translate.meteor_score.meteor_score(refs, pred))
    
    meteor = numpy.mean(all_meteor_scores)

    rouge = PyRouge().evaluate(predictions, references)

    return {'BLEU' : bleu, 'ROUGE' : rouge['rouge-1']['r'], 'METEOR' : meteor}



def display_examples(predictions, references, inputs):
    for p, r, i in random.choices(list(zip(predictions,references,inputs)), k=10):
        print('Input: ' + str(i))
        print('Prediction: ' + str(p))
        print('References: ' + str(r))


