from transformers import BartTokenizer, BartForConditionalGeneration
import qrdatasets
import config

DEFAULT_HPARAMS = {
        'epochs' : 3,
        'learning_rate' : 0.00005,
        'batch_size' : 16,
        'weight_decay' : 0.0,
        'history_size' : 2,
        'dropout_rate' : 0.1,
        'include_story' : False
    }

def get_input_tokenizer():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', padding_side = 'left')
    tokenizer.add_tokens(['<turn>'])
    return tokenizer

def get_output_tokenizer():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_tokens(['<turn>'])
    return tokenizer

def get_pretrained_model(dropout_rate):
    return BartForConditionalGeneration.from_pretrained('facebook/bart-base')

def make_dataset(data, hparams, cuda = False):
    return qrdatasets.QRDataset(
            data,
            get_input_tokenizer(),
            get_output_tokenizer(),
            hparams['include_story'],
            hparams['history_size'],
            cuda = cuda)

def load_fine_tuned_model():
    return BartForConditionalGeneration.from_pretrained('data/fine-tuned-models/bart-base')

def set_dropout_rate(model, rate):
    model.config.dropout = rate
    model.config.activation_dropout = rate
    model.config.attention_dropout = rate

MODEL_DIC = {
    'name' : 'bart-base',
    'pretrained_getter': get_pretrained_model,
    'fine_tuned_loader' : load_fine_tuned_model,
    'dataset_maker' : make_dataset,
    'input_tokenizer_getter' : get_input_tokenizer,
    'output_tokenizer_getter' : get_output_tokenizer,
    'default_hparams' : DEFAULT_HPARAMS
}
