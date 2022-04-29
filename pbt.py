from ray import tune
import evaluation
from ray.tune.schedulers.pbt import PopulationBasedTraining
from ray.tune.trial import ExportFormat
from utils import *
from models import MODEL_DICS
import qrdatasets
import torch
import random
import models
import os
import ray


def train_from_checkpoint(config, checkpoint_dir = None):
    
    model_dic = MODEL_DICS[config['model_name']] 

    if config['dataset_name'] == 'elda':
        train_set, dev_set = qrdatasets.get_elda_train_and_dev_sets(include_story=config['include_story'])
    if config['dataset_name'] == 'canard':
        train_set, dev_set = qrdatasets.get_canard_train_set(), qrdatasets.get_canard_dev_set()
    if config['dataset_name'] == 'mixed':
        train_set, dev_set = qrdatasets.get_mixed_train_and_dev_sets(include_story=config['include_story'])
    
    #train_dataset = model_dic['dataset_maker']({'input' : ['eh'], 'references' : [['oh']], 'context' : ['le nouveau son de manau']}, config)
    train_dataset = model_dic['dataset_maker'](train_set, config)

    step = 0
    model = model_dic['pretrained_getter'](config['dropout_rate'])
    
    if checkpoint_dir:
        print('Loading from checkpoint.')
        path = os.path.join(checkpoint_dir, 'checkpoint')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dic'])
        step = checkpoint['step']
   
    while True:
        model_dic['dropout_rate_setter'](model, config['dropout_rate'])
        models.train(model, train_dataset, config)

        predictions = evaluation.generate_predictions(
                model = model,
                data = dev_set,
                dataset_maker = model_dic['dataset_maker'],
                output_tokenizer = model_dic['output_tokenizer_getter'](),
                hparams = config)

        scores = evaluation.evaluate(predictions, dev_set['references'])
        if step % 1 == 0:
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save({
                    'step' : step,
                    'model_dic' : model_dic,
                    'hparams' : config,
                    'model_state_dic' : model.state_dict(),
                    'scores' : scores}, path)
        step += 1
        tune.report(bleu = scores['BLEU'], meteor = scores['METEOR'])


class CustomStopper(tune.Stopper):
    def __init__(self, smoke_test):
        self.smoke_test = smoke_test

    def __call__(self, trial_id, result):
        max_iter = 5 if self.smoke_test else 50
        return result['training_iteration'] >= max_iter

    def stop_all(self):
        return False
    

def run_pbt(args):

    ray.init(address=os.environ["ip_head"])
    
    scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval = 1,
        hyperparam_mutations = {
            'learning_rate': lambda: random.uniform(0.000001, 0.1),
            'dropout_rate': lambda: random.uniform(0,0.5),
            'history_size' : lambda: random.randint(1,10),
            'include_story' : [True, False]
        })

    tune.run(
        train_from_checkpoint,
        name = 'pbt_test',
        scheduler = scheduler,
        metric = 'meteor',
        mode = 'max',
        verbose = 1,
        stop = CustomStopper(args.smoke_test),
        export_formats = [ExportFormat.MODEL],
        checkpoint_score_attr = 'meteor',
        num_samples = 3 if args.smoke_test else 31,
        resources_per_trial = {'cpu':1, 'gpu':1},
        config = {
            'model_name' : args.model_name,
            'dataset_name' : args.dataset_name,
            #'learning_rate': 0.001,
            #'dropout_rate' : 0.1,
            'epochs': 1,
            'batch_size' : 16,
            'weight_decay' : 0,
            #'history_size' : 2,
            #'include_story' : args.include_story
        })

