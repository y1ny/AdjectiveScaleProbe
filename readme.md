This code for the paper: "Adjective Scale Probe: Can Language Models Encode Formal Semantics Information?", presented at AAAI 2023 (oral). See [the paper](https://y1ny.github.io/assets/AAAI2023_ASP_paper.pdf), [the corresponding slides](https://y1ny.github.io/assets/AAAI2023_ASP_slides.pdf) and the appendix file (coming soon).

# Data
Directory `data` contains the NLI-style samples used in the paper.

see detailed descriptions in `data/readme.md`

# Training
Directory `training` contains the code for fine-tuning pre-trained models on the MNLI or the our **Adjective Scale Probe (ASP)** dataset.

Training codes are forked from [Transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)

Run `bash run.sh`  to fine-tuning pre-trained models on the ASP.

Change the configuration of `run.sh` to reproduce other fine-tuning procedures.

# Evaluation

Directory `evaluation` contains the code for evaluating the models on the ASP.

Run `bash evaluation.sh` to test the ASP models on the leaveout testing sets.

Run `python NLI_ASP.py` to test the MNLI models on the ASP.

Run `python zs_ASP.py` to test the zero-shot models on the ASP.

`zs_ASP.py` is forked from [T-zero](https://github.com/bigscience-workshop/t-zero/tree/master/evaluation)



# Human
Directory `human` contains the questions and results for the human experiment.

`cloze.csv`: Cloze-style questions for human annotations. We change the unit to United States customary units, since all annotators are American.

`result.csv`: Raw results of human.

Directory `pkl`: processed human results for most tests of the degree estimation task.

# Plot

Directory `plot` contains the code for analysis the performance of models, and visualization.

Run the code in the `plot` after **evaluation**.

To measure the performance on the degree estimation task, we use the function `compute_metrics`. Apart from accuracy we used in the paper, we also provide multiple metrics, such as `mse` and `pearson correlation`. 
