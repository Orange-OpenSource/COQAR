# QREW

This repository contains the dataset CoQAR and the models trained for question rewriting.
CoQAR is a corpus containing 4.5K conversations from the Conversational Question-Answering dataset CoQA, for a total of 53K follow-up question-answer pairs, in which each original question was manually annotated with at least 2 at most 3 out-of-context rewritings. 
COQAR can be used for three NLP tasks: question paraphrasing, question rewriting and conversational question answering.
In order to assess the quality of COQAR's rewritings, we conducted several experiments consisting in training and evaluating models for these three tasks.
Our results support the idea that question rewriting can be used as a preprocessing step for (conversational and non-conversational) question answering models, thereby increasing their performances. 
The code presents the annotations and the models for question rewriting.

We fine-tune the huggingface model T5.

This code is an implementation of the paper: CoQAR: Question Rewriting on CoQA, submitted to LREC 2022.


## Quick Start

    virtualenv --python=python3.8 env
    ./env/bin/pip install -r requirements.txt

**TODO:** Add requirements.txt file, describe how to use the code to train, eval and produce a model

## API

Install API dependencies:

    ./env/bin/pip install -r server/requirements.txt

Then use `api.py` to start the server:

```
usage: api.py [-h] [--host HOST] [--port PORT] [--static-folder STATIC_FOLDER] [--model MODEL] [--debug]

Question Rewriting API

optional arguments:
  -h, --help            show this help message and exit
  --host HOST           Interface to listen to
  --port PORT           Port number to listen to
  --static-folder STATIC_FOLDER
                        Folder to serve as static content
  --model MODEL         URI or path for model to use
  --debug, -d           Enable flask debugging
```

You can use a local model with `--model` or you can let the server download one from the S3 server.
To use S3 you'll need to provide some credentials using environment variables with either:
- `S3_PROFILE`: the `~/.aws/credentials` profile name to use
- `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY`: if you don't have a profile configured and/or
  want to use your S3 credentials directly.

The model is store in the `di-diod-diana-fe-models` DIOD S3 bucket.
