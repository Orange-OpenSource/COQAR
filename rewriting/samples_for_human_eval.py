from utils import *
import qrdatasets
import pandas as pd
import evaluation
import t5small
import torch
import re

dataset_maker = t5small.MODEL_DIC['dataset_maker']
output_tokenizer = t5small.MODEL_DIC['output_tokenizer_getter']()

coqar = qrdatasets.shuffle_data(qrdatasets.get_elda_test_set(True), 50, True)
canard = qrdatasets.shuffle_data(qrdatasets.get_canard_test_set(True), 50, True)

coqar_model = torch.load(f'data/trained_models/best_coqar_model')
canard_model = torch.load(f'data/trained_models/best_canard_model')
mixed_model = torch.load(f'data/trained_models/best_mixed_model')

hparams = {
    'batch_size' : 16,
    'history_size' : 1,
    'include_story' : False
}

coqar_preds = evaluation.generate_predictions(
        coqar_model, coqar, dataset_maker, output_tokenizer, hparams)

canard_preds = evaluation.generate_predictions(
        canard_model, canard, dataset_maker, output_tokenizer, hparams)

mixed_preds = evaluation.generate_predictions(
        mixed_model, canard, dataset_maker, output_tokenizer, hparams)


def clean(x):
    x = re.sub('\n', ' ', x)
    x = re.sub('\t', ' ', x)
    x = re.sub(';', ':', x)
    print(x)
    return x

coqar['input'] = [[clean(y) for y in x] for x in coqar['input']]
coqar['context'] = [clean(x) for x in coqar['context']]
coqar_preds = [clean(x) for x in coqar_preds]
canard['input'] = [[clean(y) for y in x] for x in canard['input']]
canard['context'] = [clean(x) for x in canard['context']]
canard_preds = [clean(x) for x in canard_preds]

# SIMILARITY

fluency_df = pd.DataFrame([], columns = ['dataset', 'input', 'output', 'output_source'])

for x, y, p in zip(coqar['input'], coqar_preds, coqar['context']):
    
    dialogue = '- ' + '<br>- '.join(x[:-1])
    fluency_df = fluency_df.append({
        'dataset' : 'coqar',
        'input' : x[-1],
        'output' : y,
        'output_source' : 'coqar_model',
        'dialogue' : dialogue,
        'paragraph' : p,
    }, ignore_index = True)

for x, y, p in zip(coqar['input'], coqar['references'], coqar['context']):
    dialogue = '- ' + '<br>- '.join(x[:-1])
    fluency_df = fluency_df.append({
        'dataset' : 'coqar',
        'input' : x[-1],
        'output' : y[0],
        'output_source' : 'ref',
        'dialogue' : dialogue,
        'paragraph' : p,
    }, ignore_index = True)

for x, y, p in zip(canard['input'], canard_preds, canard['context']):
    dialogue = '- ' + '<br>- '.join(x[:-1])
    fluency_df = fluency_df.append({
        'dataset' : 'canard',
        'input' : x[-1],
        'output' : y,
        'output_source' : 'canard_model',
        'dialogue' : dialogue,
        'paragraph' : p,
    }, ignore_index = True)

for x, y, p in zip(canard['input'], mixed_preds, canard['context']):
    dialogue = '- ' + '<br>- '.join(x[:-1])
    fluency_df = fluency_df.append({
        'dataset' : 'canard',
        'input' : x[-1],
        'output' : y,
        'output_source' : 'mixed_model',
        'dialogue' : dialogue,
        'paragraph' : p,
    }, ignore_index = True)

for x, y, p  in zip(canard['input'], canard['references'], canard['context']):
    dialogue = '- ' + '<br>- '.join(x[:-1])
    fluency_df = fluency_df.append({
        'dataset' : 'canard',
        'input' : x[-1],
        'output' : y[0],
        'output_source' : 'mixed_model',
        'dialogue' : dialogue,
        'paragraph' : p,
    }, ignore_index = True)

fluency_df.to_csv('data/samples_for_human_eval/similarity.csv', index=False)


# COMPLETENESS

completeness_df = pd.DataFrame([], columns = ['dataset', 'output', 'output_source', 'paragraph'])

for x, y in zip(coqar['context'], coqar_preds):
    completeness_df = completeness_df.append({
        'dataset' : 'coqar',
        'paragraph' : x,
        'output' : y,
        'output_source' : 'coqar_model'
    }, ignore_index = True)

for x, y in zip(coqar['context'], coqar['references']):
    completeness_df = completeness_df.append({
        'dataset' : 'coqar',
        'paragraph' : x,
        'output' : y[0],
        'output_source' : 'ref'
    }, ignore_index = True)

for x, y in zip(coqar['context'], coqar['input']):
    completeness_df = completeness_df.append({
        'dataset' : 'coqar',
        'paragraph' : x,
        'output' : y[-1],
        'output_source' : 'input'
    }, ignore_index = True)

for x, y  in zip(canard['context'], canard_preds):
    completeness_df = completeness_df.append({
        'dataset' : 'canard',
        'paragraph' : x,
        'output' : y,
        'output_source' : 'canard_model'
    }, ignore_index = True)

for x, y  in zip(canard['context'], mixed_preds):
    completeness_df = completeness_df.append({
        'dataset' : 'canard',
        'paragraph' : x,
        'output' : y,
        'output_source' : 'mixed_model'
    }, ignore_index = True)

for x, y  in zip(canard['context'], canard['references']):
    completeness_df = completeness_df.append({
        'dataset' : 'canard',
        'paragraph' : x,
        'output' : y[0],
        'output_source' : 'mixed_model'
    }, ignore_index = True)


for x, y  in zip(canard['context'], canard['input']):
    completeness_df = completeness_df.append({
        'dataset' : 'canard',
        'paragraph' : x,
        'output' : y[-1],
        'output_source' : 'input'
    }, ignore_index = True)

completeness_df.to_csv('data/samples_for_human_eval/completeness.csv', index=False)


# FLUENCY

fluency_df = pd.DataFrame([], columns = ['dataset', 'input', 'output', 'output_source'])

for x, y in zip(coqar['input'], coqar_preds):
    fluency_df = fluency_df.append({
        'dataset' : 'coqar',
        'output' : y,
        'output_source' : 'coqar_model'
    }, ignore_index = True)

for x, y in zip(coqar['input'], coqar['references']):
    fluency_df = fluency_df.append({
        'dataset' : 'coqar',
        'output' : y[0],
        'output_source' : 'ref'
    }, ignore_index = True)

for x, y  in zip(canard['input'], canard_preds):
    fluency_df = fluency_df.append({
        'dataset' : 'canard',
        'output' : y,
        'output_source' : 'canard_model'
    }, ignore_index = True)

for x, y  in zip(canard['input'], mixed_preds):
    fluency_df = fluency_df.append({
        'dataset' : 'canard',
        'output' : y,
        'output_source' : 'mixed_model'
    }, ignore_index = True)

for x, y  in zip(canard['input'], canard['references']):
    fluency_df = fluency_df.append({
        'dataset' : 'canard',
        'output' : y[0],
        'output_source' : 'mixed_model'
    }, ignore_index = True)

fluency_df.to_csv('data/samples_for_human_eval/fluency.csv', index=False)

