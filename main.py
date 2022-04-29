import torch
import evaluation
import sys
import argparse
import pbt
from models import MODEL_DICS
import qrdatasets
import models
import os
from utils import *
import random

def get_arguments():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('action', choices=['pbt', 'eval', 'pre-eval', 'train'])
    parser.add_argument('model_name', choices=['t5small', 'bartbase'])
    parser.add_argument('dataset_name', choices=['canard', 'elda', 'mixed'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--history_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--include_story', dest='include_story', action='store_true')
    parser.set_defaults(include_story=False)
    parser.add_argument('--smoke_test', dest='smoke_test', action='store_true')
    parser.set_defaults(smoke_test=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    random.seed(63254)
    args = get_arguments()

    model_dic = MODEL_DICS[args.model_name]

    # overrides default parameters
    hparams = model_dic['default_hparams']
    if args.batch_size != None:
        hparams['batch_size'] = args.batch_size
    if args.dropout_rate != None:
        hparams['dropout_rate'] = args.dropout_rate
    if args.learning_rate != None:
        hparams['learning_rate'] = args.learning_rate
    if args.history_size != None:
        hparams['history_size'] = args.history_size

    hparams['include_story'] = args.include_story

    if args.action in ['eval', 'pre-eval']:
        
        model = torch.load(f'data/trained_models/best_coqaqr_model')
        if args.action == 'eval':
            
            elda = qrdatasets.get_elda_test_set(args.include_story)
            canard = qrdatasets.get_canard_test_set(args.include_story)
    
        if args.action == 'pre-eval':
        
            _, elda = qrdatasets.get_elda_train_and_dev_sets(args.include_story)
            canard = qrdatasets.get_canard_dev_set(args.include_story)

        for data, dataset_name in [(elda, 'elda'), (canard, 'canard')]:
            predictions = evaluation.generate_predictions(
                model = model,
                data = data,
                dataset_maker = model_dic['dataset_maker'],
                output_tokenizer = model_dic['output_tokenizer_getter'](),
                hparams = hparams)

            print(f"{dataset_name}: {evaluation.evaluate(predictions, data['references'])}")
            #evaluation.display_examples(predictions, data['references'], data['input'])
    
    if args.action == 'train':
        dir_path = create_and_get_dir('data/logs', 'training')
        models.train_dev_loop(args, hparams, log_dir_path = dir_path) 

    if args.action == 'pbt':
        pbt.run_pbt(args)
