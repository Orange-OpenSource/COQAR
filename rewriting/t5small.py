from transformers import T5ForConditionalGeneration, T5Tokenizer
import config
import qrdatasets

def get_input_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained(config.T5v1_1_TOKENIZER_PATH, padding_side = 'left')
    tokenizer.add_tokens(['<question>'])
    tokenizer.add_tokens(['<answer>'])
    return tokenizer

def get_output_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained(config.T5v1_1_TOKENIZER_PATH)
    tokenizer.add_tokens(['<question>'])
    tokenizer.add_tokens(['<answer>'])
    return tokenizer 

def get_pretrained_model(dropout_rate):
    model = T5ForConditionalGeneration.from_pretrained(config.T5v1_1_PATH)
    model.config.dropout_rate = dropout_rate
    tokenizer = get_input_tokenizer()
    model.resize_token_embeddings(len(tokenizer))
    return model

def make_dataset(data, hparams, cuda = False):
    return qrdatasets.QRDataset(
            data,
            get_input_tokenizer(),
            get_output_tokenizer(),
            hparams['include_story'],
            hparams['history_size'],
            cuda = cuda)

def load_fine_tuned_model():
    return T5ForConditionalGeneration.from_pretrained('data/fine-tuned-models/t5-small')
