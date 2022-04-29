# COQAR
CoQAR is a corpus containing 4.5K conversations from the using open-source dataset [Conversational Question-Answering dataset CoQA](https://stanfordnlp.github.io/coqa/), for a total of 53K follow-up question-answer pairs. 
In CoQAR each original question was manually annotated with at least 2 at most 3 out-of-context rewritings. 

The corpus CoQA contains passages from seven domains, which are public under the following licenses:
•Literature and Wikipedia passages are shared under CC BY-SA 4.0 license. 
•Children's stories are collected from MCTest which comes with MSR-LA license. 
•Middle/High school exam passages are collected from RACE which comes with its own license. 
•News passages are collected from the DeepMind CNN dataset which comes with Apache license. 

We annotated each original question of CoQA with at least 2 at most 3 out-of-context rewritings. We agreed with Celine Fontaine that these annotations would be released as open source.
Besides the annotations, we also provide the code both related to the paper "Question Rewriting on CoQA", in which we present the paraphrasing, question rewriting by using T5.  

In the paper: **CoQAR Question Rewriting on CoQA. Quentin Brabant, Gwenole Lecorve and Lina Rojas-Barahona, to be published in LREC2022**, we assess the quality of COQAR's annotations by conducting several experiments for question paraphrasing, question rewriting and conversational question answering. Our results support the idea that question rewriting can be used as a preprocessing step for  non-conversational question answering models, thereby increasing their performances. 
The code presents the annotations and the models for question rewriting.
