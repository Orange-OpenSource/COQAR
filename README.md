# COQAR
CoQAR is a corpus containing 4.5K conversations from the using open-source dataset [Conversational Question-Answering dataset CoQA](https://stanfordnlp.github.io/coqa/), for a total of 53K follow-up question-answer pairs. 
In CoQAR each original question was manually annotated with at least 2 at most 3 out-of-context rewritings. 

The corpus CoQA contains passages from seven domains, which are public under the following licenses:
 - Literature and Wikipedia passages are shared under CC BY-SA 4.0 license. 
 - Children's stories are collected from MCTest which comes with MSR-LA license. 
 - Middle/High school exam passages are collected from RACE which comes with its own license. 
 - News passages are collected from the DeepMind CNN dataset which comes with Apache license. 

We annotated each original question of CoQA with at least 2 at most 3 out-of-context rewritings. 

![image](https://user-images.githubusercontent.com/52821991/165952155-822ce743-791d-46c8-8705-0937a69df933.png)


These annotations are released as open source.
Beside the annotations, we also provide the code for question rewriting by using T5.  The datasets and code are presented in the paper: **CoQAR Question Rewriting on CoQA. Quentin Brabant, Gwenole Lecorve and Lina Rojas-Barahona, to be published in [LREC2022](https://lrec2022.lrec-conf.org/en/)**, we assess the quality of COQAR's annotations by conducting several experiments for question paraphrasing, question rewriting and conversational question answering. Our results support the idea that question rewriting can be used as a preprocessing step for  non-conversational question answering models, thereby increasing their performances. 
In this repository you can find the dataset with the annotations and the models for question rewriting.

The code is published under the licence Apache 2.0, as the models are taken from HuggingFace.
The annotations are published under the licence CC-BY-SA 4.0.
The original content of the dataset CoQA is under the distinct licences described above.
