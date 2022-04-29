import t5small
from torch.utils.data import Dataset, DataLoader
from transformers import Adafactor, Seq2SeqTrainer, TrainingArguments
from torch.optim import AdamW
import bartbase
import config
import os
import torch
import evaluation
import json
import qrdatasets
from utils import *
import pandas as pd


MODEL_DICS = {
    't5small' : t5small.MODEL_DIC,
    'bartbase' : bartbase.MODEL_DIC
}


def get_scores(model, data, dataset_maker, output_tokenizer, hparams, save_path = None):
    predictions = evaluation.generate_predictions(
            model = model,
            data = data,
            dataset_maker = dataset_maker,
            output_tokenizer = output_tokenizer,
            hparams = hparams,
            save_path = save_path)
  
    return evaluation.evaluate(predictions, data['references'])

def train_reformulation(args, hparams):
    pass

def train_dev_loop_old(args, hparams, log_dir_path = None):
    train, dev = qrdatasets.get_train_dev(
            args.dataset_name,
            args.include_story) 
        
    model_dic = MODEL_DICS[args.model_name]
    del hparams['epochs']
    output_tokenizer = model_dic['output_tokenizer_getter']()
    print(json.dumps(hparams, indent = 1))

    # initializes datasets and model
    trainset = model_dic['dataset_maker'](train, hparams)
    devset = model_dic['dataset_maker'](dev, hparams)
    qrdatasets.print_dataset_into_file(
            trainset,
            output_tokenizer,
            log_dir_path + '/train_data.csv'
        )

    model = model_dic['pretrained_getter'](hparams['dropout_rate'])
   
    trainer = Seq2SeqTrainer(
            model = model,
            args = TrainingArguments(
                "Test",
                num_train_epochs = 1.0,
                per_gpu_train_batch_size = 16),
            train_dataset = trainset,
            eval_dataset = devset)
    score_list = []
    for epoch in range(30):
        j = 0
        file_path = log_dir_path + '/predictions'
        while os.path.exists(file_path + str(j)):
            j += 1
        file_path = file_path + str(j)

        scores = get_scores(
                model,
                dev,
                model_dic['dataset_maker'],
                output_tokenizer,
                hparams,
                save_path = file_path
        )
        print(scores)
        score_list.append(scores)
        train_result = trainer.train(resume_from_checkpoint = True)
    
    print(score_list)
    json_save(score_list, log_dir_path + '/scores', indent = 1)
    trainer.save_model()


def train_dev_loop(args, hparams, log_dir_path = None):
    
    train, dev = qrdatasets.get_train_dev(
            args.dataset_name,
            args.include_story
    ) 
        
    model_dic = MODEL_DICS[args.model_name]
    del hparams['epochs']
    print(json.dumps(hparams, indent = 1))

    # initializes datasets and model
    trainset = model_dic['dataset_maker'](train, hparams, cuda=True)
    output_tokenizer = model_dic['output_tokenizer_getter']()
    qrdatasets.print_dataset_into_file(
            trainset,
            output_tokenizer,
            log_dir_path + '/train_data.csv'
    )

    model = model_dic['pretrained_getter'](hparams['dropout_rate'])
    model.cuda()

    dataloader = DataLoader(
        dataset=trainset,
        batch_size=hparams['batch_size'],
        shuffle = True
    )

    optimizer = AdamW(
        model.parameters(),
        lr = hparams['learning_rate'],
        weight_decay = hparams['weight_decay']
    )

    threshold = 10000
    counter = 0
    running_loss = 0.
    score_list = []
    while True:    
        file_path = get_file_path(log_dir_path, 'predictions')       
        
        ### >>> eval on dev set
        model.train(False)
        scores = get_scores(
                model,
                dev,
                model_dic['dataset_maker'],
                output_tokenizer,
                hparams,
                save_path = file_path
        )
       
        print('Scores on dev set : ' + str(scores))
        score_list.append(scores)
        
        if len(score_list) >= 3 \
        and scores['METEOR'] <= score_list[-2]['METEOR']\
        and scores['METEOR'] <= score_list[-3]['METEOR']:
            break
        ### <<< end eval

        ### >>> one epoch of training
        model.train()
        for dic in dataloader:
            optimizer.zero_grad()
            loss = model(**dic).loss
            loss.backward()
            optimizer.step()

            # printing stuff
            counter += dic['input_ids'].size()[0]
            running_loss += loss.item()
            if counter >= threshold:
                print('loss: %.6f' % (running_loss / counter))
                running_loss = 0.0
                counter = 0   
        ### <<< end epoch

        # save model
        torch.save(model, get_file_path(log_dir_path, 'epoch'))    
    
    json_save(score_list, log_dir_path + '/scores.json', indent = 1)        


def train(model, dataset, hparams, optimizer_name):
    
    model.cuda()
    model.train()

    dataloader = DataLoader(dataset=dataset, batch_size=hparams['batch_size'])

    if optimizer_name == 'Adafactor':
        optimizer = Adafactor(
            model.parameters(),
            weight_decay=hparams['weight_decay'],
            lr=hparams['learning_rate'],
            scale_parameter=False,
            relative_step=False
        )
    if optimizer_name == 'AdamW':
        optimizer = AdamW(
            model.parameters(),
            lr = hparams['learning_rate'],
            weight_decay = hparams['weight_decay']
        )
    threshold = 1000
    counter = 0
    running_loss = 0.
    for epoch in range(hparams['epochs']):
        for dic in dataloader:
            optimizer.zero_grad()
            loss = model(**dic).loss

            loss.backward()
            optimizer.step()

            # printing stuff
            counter += dic['input_ids'].size()[0]
            running_loss += loss.item()
            if counter >= threshold:
                print('loss: %.6f' % (running_loss / counter))
                running_loss = 0.0
                counter = 0
