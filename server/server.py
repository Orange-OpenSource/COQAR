import os
import urllib
from logzero import logger
from rewriting import t5small
import torch
from rewriting import qrdatasets
from torch.utils.data import DataLoader

class Server:
    def __init__(self):
        self.model = None
        self.input_tokenizer = None
        self.output_tokenizer = None
    def init_app(self,app):
        model_repo = os.getenv('MODEL_URL','s3://di-diod-diana-fe-models/qrew/qrew.model')
        logger.info("Using model at: %s",model_repo)
        parsed_repo = urllib.parse.urlparse(model_repo)
        if parsed_repo.scheme == 's3':
            import boto3
            import tqdm
            sess = boto3.session.Session(
                profile_name=os.getenv('S3_PROFILE'))
            s3 = sess.client(
                service_name='s3',
                aws_access_key_id=os.getenv('S3_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('S3_SECRET_ACCESS_KEY'),
                endpoint_url=os.getenv(
                    'S3_ENDPOINT_URL',
                    "https://s3.flexible-datastore.orange-business.com"))
            bucket_name = parsed_repo.netloc
            file_name = parsed_repo.path[1:]
            logger.info("Retrieving model from bucket %s with name %s"%(bucket_name,file_name))
            file_object = s3.get_object(Bucket=bucket_name,Key=file_name)
            filesize = file_object['ContentLength']
            with tqdm.tqdm(total=filesize, unit='B', unit_scale=True, desc=file_name) as pbar:
                s3.download_file(Bucket=bucket_name,
                                 Key=file_name,
                                 Filename='/tmp/qrew.model',
                                 Callback=lambda transferred: pbar.update(transferred))
            model_path = '/tmp/qrew.model'
        elif parsed_repo.scheme == '':
            import shutil
            model_path = parsed_repo.path
        else:
            logger.error("Unsupported model repo url: %s"%model_repo)
            exit(1)
        self.load_model(model_path)
    def load_model(self,model_path):
        self.input_tokenizer = t5small.get_input_tokenizer()
        self.output_tokenizer = t5small.get_output_tokenizer()
        self.model = torch.load(model_path,map_location=torch.device('cpu'))
        self.model.eval()

    def format_conversation(self,sentence,context):
        conversation = ''
        flip = True
        for x in context:
            if (len(context) % 2 == 0) == flip:
                conversation += '<question> '
            else:
                conversation += '<answer> '
            conversation += x
            flip = not flip
        conversation += self.input_tokenizer.eos_token + ' ' + sentence
        return conversation

    def rewrite(self,*,sentence=None,context=[]):
        if sentence is None:
            raise Exception("Missing sentence to rewrite")
        if self.model is None:
            raise Exception("No model loaded")
        # no context, no rewrite
        if context is None:
            return {
                "sentence": sentence
            }

        conversation = self.format_conversation(sentence, context)
        encoding = self.input_tokenizer(conversation, return_tensors='pt').input_ids
        output = self.model.generate(encoding)
        rewriting = self.output_tokenizer.decode(output[0], skip_special_tokens = True)
        return {
            "sentence": rewriting
        }

    def rewrite_batch(self,*,instances=[]):
        # XXX: each instance is a dict same as rewrite() input:
        # XXX: {'sentence': 'string', 'context': ['string']}

        convs = [self.format_conversation(inst['sentence'], inst['context']) for inst in instances]
        dataset = qrdatasets.QRDatasetForServer(convs,self.input_tokenizer,self.output_tokenizer)

        loader = DataLoader(dataset=dataset, batch_size=16)

        predictions = []
        for dic in loader:
            output = self.model.generate(input_ids=dic['input_ids'], attention_mask=dic['attention_mask'])
            pred = self.output_tokenizer.batch_decode(output, skip_special_tokens=True)
            predictions += pred

        return {
            "outputs": predictions
        }
