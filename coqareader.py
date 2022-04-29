import argparse
import json
import os
import pickle
import argparse
import time
import matplotlib.pyplot as plt
from configparser import ConfigParser

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from datasets import load_metric

from io import open
from os import path
from pathlib import Path
import numpy as np

def usarguments():

    parser = argparse.ArgumentParser(\
            description='Default evaluation of user satisfaction parser.')
    parser.add_argument('-c', help='config file path.')
    parser.add_argument('-op', help='data file path.')



    return parser.parse_args()

class CoQAExample(object):
    """
    A single training/test example for the ConvQuestion dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 dialogue_id,
                 turns,
                 source
                 ):
        self.dialogue_id=dialogue_id
        self.turns= turns
        self.source=source



    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "dialogue_id: %s" % (self.dialogue_id)
        s += ", turns: %s" % str(self.turns)
        s += " source %s" % (self.source)

        return s


def readcorpus(datapath):

    max_turn_len=0
    datatype="dev"
    if "train" in datapath:
        datatype="train"
    
    fin = open(datapath, "r")
    coqa_corpus={}
    data=[]
    dialogues={}
    with open(datapath) as json_wiki:
        coqa_corpus=json.load(json_wiki)
        data=coqa_corpus["data"]
        counter=0
        for dial in data:
            dialogue = {}
            dialogId= dial["id"]
            if dialogId in dialogues:
                dialogue = dialogues[dialogId]
                turns = dialogue['turns'].copy()

            else:
                dialogue["id"] = dialogId
                turns = []
                
            dialogue["story"]=dial["story"]
            dialogue["source"]=dial["source"]
            questions=dial["questions"]
            answers=dial["answers"]
            q=0
            for question in questions:
                turn = {}
                turn['id']=len(turns)
                turn["text"]=question["input_text"]
                turn["paraphrases"]=question["paraphrase"]
                turn['speaker']="user"
                turns.append(turn.copy())
                turn = {}
                turn['id']=len(turns)
                turn['speaker']="system"
                turn["text"]=answers[q]["input_text"]
                turn["span_text"]=answers[q]["span_text"]
                turn["span_start"]=answers[q]["span_start"]
                turn["span_end"]=answers[q]["span_end"]
                turns.append(turn.copy())
                q+=1

            dialogue['turns'] = turns
            dialogues[dialogId]=dialogue

    examples = []
    examples_dict={}
    for dialogue_id in dialogues:
        dialogue=dialogues[dialogue_id]

        example = CoQAExample(dialogue_id, dialogue['turns'],dialogue['source'])
        examples_dict[dialogue_id]=example
        examples.append(example)

    return examples_dict,examples, datatype
 
def doubleannotations(parser):
    annotator1,annotator2=None,None
    if parser.has_option('data', 'annotator1'):
        annotator1 = parser.get('data', 'annotator1')
    if parser.has_option('data', 'annotator2'):
        annotator2 = parser.get('data', 'annotator2')
    chencherry = SmoothingFunction()
    ann1_dict,exs_ann1, datatype_ann1 = readcorpus(annotator1)
    ann2_dict,exs_ann2, datatype_ann1 = readcorpus(annotator2)

    all_refs=[]
    all_preds=[]
    all_preds0=[]
    all_preds1=[]
    i=0
    for key in ann1_dict.keys() :
        example=ann1_dict[key]
        example_ann2=ann2_dict[key]
        #print(' turn annotated by annotator 1 : {}'.format(len(example.turns)))
        #print(' turn annotated by annotator 2 : {}'.format(len(example_ann2.turns)))
        for turn in example.turns:
            turn_idx= example.turns.index(turn)
            ann2_turn=example_ann2.turns[turn_idx]
            refs=[]
            preds=[]

            if "paraphrases" in turn:
                if len(turn['paraphrases']) < 1 or len(ann2_turn['paraphrases']) < 1:
                    continue

                for par in turn['paraphrases']:
                    idx_par=turn['paraphrases'].index(par)
                    par2 = ann2_turn['paraphrases'][idx_par]
                    if len(par)>0 and len(par2)>0 and len(turn['paraphrases'])>=len(ann2_turn['paraphrases']):
                        refs.append(par.split())
                        preds.append(par2.split())
                    elif  len(par)>0 and len(par2)>0 and len(turn['paraphrases'])<len(ann2_turn['paraphrases']):
                        refs.append(par2.split())
                        preds.append(par.split())

                if(len(preds)<1) or len(refs)<1:
                    continue

                print("refs: {}, preds: {}".format(refs,preds))
                all_refs.append(refs.copy())
                all_preds.append(preds.copy())

                all_preds0.append(preds[0].copy())
                if len(preds)>0:

                    print("TURN BLEU: "+ str(sentence_bleu(refs, preds[0])))

                if len(preds)>1:
                    all_preds1.append(preds[1].copy())
                    print("TURN BLEU: "+ str(sentence_bleu(refs, preds[1])))


        
    
    print("CORPUS BLEU_0: "+ str(corpus_bleu(all_refs, all_preds0, smoothing_function=chencherry.method3)))
    #print("CORPUS BLEU_1: "+ str(corpus_bleu(all_refs, all_preds1, smoothing_function=chencherry.method3)))
    from datasets import list_datasets, load_dataset, list_metrics, load_metric
    print(list_metrics())
    bertscore = load_metric("bertscore")
    results = bertscore.compute(predictions=all_preds, references=all_refs, lang="en")
    scores=[round(v, 2) for v in results["f1"]]
    berts_mean=np.array(scores).mean()
    print('bert score: ', berts_mean)
    sacrebleu = load_metric("sacrebleu")
    sb_refs=[[x[0]] for x in all_refs]
    min_refs=10
    max_refs=0
    
    for ref in all_refs:
        if len(ref)>max_refs:
            max_refs=len(ref)
        if len(ref)<min_refs:
            min_refs=len(ref)

    print('[{}-{}]'.format(min_refs,max_refs))

    results_sacre = sacrebleu.compute(predictions=all_preds0, references=sb_refs)
    print(list(results_sacre.keys()))
    sacrebleu_score=round(results_sacre["score"], 2) 
    print('sacre bleu: ', sacrebleu_score)


if __name__ == "__main__":
    args = usarguments()
    config = args.c
    parse_conf = ConfigParser()
    parse_conf.read(config)

    if args.op == "stats":
        datapath="data/"
        if parse_conf.has_option('data', 'corpuspath'):
            datapath = parser.get('data', 'corpuspath')
        ex_dict, exs, datatype = readcorpus(datapath)
        pickle.dump(exs, open(datatype+"_coqa_examples.p", "wb"))
    if args.op == "agree":
        doubleannotations(parse_conf)
