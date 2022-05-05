# Question Rewriting

This folder contains the code for training and evaluating question rewriting models using CoQAR
and [CANARD](https://sites.google.com/view/qanta/projects/canard).

Training and evaluation are both done by calling ``main.py``. For indication on arguments to use, call the ``python main.py --help`` command.

Note that you should modifiy ``config.py`` to add the path of the CANARD folder, which can be downloaded [here](https://sites.google.com/view/qanta/projects/canard).

We used Python 3.8.11 ; package versions are listed in ``requirements.txt``. 
