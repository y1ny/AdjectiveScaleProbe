# Entailment Inference
Samples of the entailment inference task.

Each test contains **1000** NLI samples. All samples are formulated as \<premise,    hypothesis,    label\>

# Degree Estimation
Samples of the degree estimation task.

Each test contains **2000** NLI samples.

Notably, due to the labels of some tests are determined by human annotation (see Appendix C). Some tests does not provide the gold label. 

To analysis the performance on the degree estimation task, run the code in `plot`, or use the function `compute_metrics` in `ASP_utils.py`

# ASP_fine-tuning
Datasets used for the experiment "Fine-tuning on the ASP" in the paper.

## Train
Sub-directory Train contains the NLI samples used for fine-tuning on the ASP.

The fine-tuning is done 4 times (see details in the paper). Each time, we use leave_ei.csv (generated from training vocabulary) and leave_*dim*.csv for fine-tuning. 

Each leave_*dim*.csv file contains the samples constructed from 3 dimensions (e.g., length, mass, price), and leave the samples from *dim* (e.g., temperature) for testing.

## Validation
Samples for validation set. Validation set also does not contain the leaveout adjective/dimension.

## Test
Leaveout testing sets, used for evaluating the models fine-tuning on a subset of the ASP.

## Wider
Datasets for the control experiment (see Appendix D). 
