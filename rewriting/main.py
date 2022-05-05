import torch
import evaluation
import argparse
import qrdatasets
import models
from utils import *
import random
import t5small

def get_arguments():
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('action', choices=['eval', 'pre-eval', 'train'],
                        help='"pre-eval" evaluates the model on dev-sets, while "eval" evaluates on test sets.')
    parser.add_argument('--dataset_name', choices=['canard', 'coqar', 'mixed'],
                        help='Use this argument only when training. With "mixed", a mix of CANARD and COQAR is used.')
    parser.add_argument('--model_path', type=str, default='',
                        help='Only for eval/pre-eval. Path of the model to evaluate.')
    parser.add_argument('--epochs', type=int, help='only for training')
    parser.add_argument('--learning_rate', type=float, help='only for training')
    parser.add_argument('--history_size', type=int, help='only for training')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dropout_rate', type=float, help='only for training')
    parser.add_argument('--smoke_test', dest='smoke_test', action='store_true', help='only for training')
    parser.set_defaults(smoke_test=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    random.seed(63254)
    args = get_arguments()

    # default parameters
    hparams = {
        'epochs' : 3,
        'learning_rate' : 0.00005,
        'batch_size' : 16,
        'weight_decay' : 0.0,
        'history_size' : 20,
        'dropout_rate' : 0.1,
        'include_story' : False
    }
    # overrides default params
    if args.batch_size != None:
        hparams['batch_size'] = args.batch_size
    if args.dropout_rate != None:
        hparams['dropout_rate'] = args.dropout_rate
    if args.learning_rate != None:
        hparams['learning_rate'] = args.learning_rate
    if args.history_size != None:
        hparams['history_size'] = args.history_size

    if args.action in ['eval', 'pre-eval']:
        
        model = torch.load(args.model_path)
        if args.action == 'eval':
            
            coqar = qrdatasets.get_coqar_test_set(hparams['include_story'])
            canard = qrdatasets.get_canard_test_set(hparams['include_story'])
    
        if args.action == 'pre-eval':
        
            _, coqar = qrdatasets.get_coqar_train_and_dev_sets(hparams['include_story'])
            canard = qrdatasets.get_canard_dev_set(hparams['include_story'])

        for data, dataset_name in [(coqar, 'coqar'), (canard, 'canard')]:
            predictions = evaluation.generate_predictions(
                model = model,
                data = data,
                dataset_maker = t5small.make_dataset,
                output_tokenizer = t5small.get_output_tokenizer(),
                hparams = hparams)

            print(f"{dataset_name}: {evaluation.evaluate(predictions, data['references'])}")
            #evaluation.display_examples(predictions, data['references'], data['input'])
    
    if args.action == 'train':
        dir_path = create_and_get_dir('trained_models', 'training')
        models.train_dev_loop(args, hparams, log_dir_path = dir_path)
