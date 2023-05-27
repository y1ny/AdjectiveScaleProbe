Code and dataset for the paper: "Adjective Scale Probe: Can Language Models Encode Formal Semantics Information?"

# Data
Directory data contains the NLI-style samples used in the paper.

Each sample is formulated as :

premise	hypothesis	label

see detailed descriptions in data/readme.md

# Training

Code for fine-tuning pre-trained models on the MNLI/ASP.

Training codes are forked from: https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification

Run `bash run.sh`  to fine-tuning pre-trained models on the ASP.

Change the configuration of `run.sh` to reproduce other fine-tuning procedures.



# Evaluation

Code for evaluating the models on the ASP.

Run `bash evaluation.sh` to test the ASP models on the leaveout testing sets.

Run `python NLI_ASP.py` to test the MNLI models on the ASP.

Run `python zs_ASP.py` to test the zero-shot models on the ASP.

`zs_ASP.py` is forked from https://github.com/bigscience-workshop/t-zero/tree/master/evaluation



# Human

`cloze_sample.csv`: Cloze-style questions for human annotations. We change the unit to United States customary units, since all annotators are American.

`result.csv`: Annotation results of human.

Directory `pkl`: processed human results for most tests of the degree estimation task.

# Plot

Codes for analysis the performance of models, and visualization.

Run the code in the `plot` after **evaluation**.

To measure the performance on the degree estimation task, we use the function `compute_metrics`. Apart from accuracy we used in the paper, we also provide multiple metrics, such as mse and pearson correlation. 