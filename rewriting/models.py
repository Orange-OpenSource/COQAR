import t5small
from torch.utils.data import Dataset, DataLoader
from transformers import Adafactor, Seq2SeqTrainer, TrainingArguments
from torch.optim import AdamW
import torch
import evaluation
import qrdatasets
from utils import *


def get_scores(model, data, dataset_maker, output_tokenizer, hparams, save_path = None):
    predictions = evaluation.generate_predictions(
            model = model,
            data = data,
            dataset_maker = dataset_maker,
            output_tokenizer = output_tokenizer,
            hparams = hparams,
            save_path = save_path)
  
    return evaluation.evaluate(predictions, data['references'])


def train_dev_loop(args, hparams, log_dir_path = None):
    
    train, dev = qrdatasets.get_train_dev(
            args.dataset_name,
            args.include_story
    ) 

    del hparams['epochs']
    print(json.dumps(hparams, indent = 1))

    # initializes datasets and model
    trainset = t5small.make_dataset(train, hparams, cuda=True)
    output_tokenizer = t5small.get_output_tokenizer()
    qrdatasets.print_dataset_into_file(
            trainset,
            output_tokenizer,
            log_dir_path + '/train_data.csv'
    )

    model = t5small.get_pretrained_model(hparams['dropout_rate'])
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
                t5small.make_dataset,
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
