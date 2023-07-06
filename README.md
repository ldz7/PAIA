# Prior Information Assisted Integrative Analysis of Multiple Datasets

Code for the paper *Prior Information Assisted Integrative Analysis of Multiple Datasets*.

<img src="https://user-images.githubusercontent.com/60643168/236855479-eab29db6-b155-4858-bfd3-1998538a49ca.png" width="50%" height="50%">

There are two code files, which corresponds to Section 2.2 and Section 2.3, respectively.


## PART1: `cnn_active_learning.py` for prior information extraction (Section 2.2)

### Requirements
Code is written in Python (3.6). The following packages need to be installed before running the code. 

```
re==2.2.1
logging==0.5.1.2
itertools
numpy==1.21.5
pandas==1.3.5
collections
string
torch==1.10.1
scikit-learn==1.0.2
argparse==1.1
os
```

### Input
The input is a file which consists of sentences and its labels. The excel file should be organized as follows:

| sentence      | relation |
| --------- | -----|
| sentence 1 | 1 |
| sentence 2 | 0 |
| sentence 3 | 1 |
| ... | ... |


### Output
After running this code, these files will be generated:
```
{dataset_name}_result
│  al_metrics_{dataset_name}.xlsx
│  early_stopping_model.pt
│  hyperparam_df_{dataset_name}.xlsx
│  model_iter=0.pt
│  
└─training process
        loss_iteration=0.xlsx
```
`{dataset_name}` is the value of variable `dataset` in code `cnn_active_learning.py`;

`al_metrics_{dataset_name}.xlsx` records metrics (auc, sen, spe, gmeans) in each iteration;

`early_stopping_model.py` can be ignored. It is just used to reload the best model in early stopping procedure;

`hyperparam_df_{dataset_name}.xlsx` records the best hyperparameter in each iteration;

`model_iter=0.pt` is the model trained in iteration 0;

`training process/loss_iteration=0.xlsx` records the loss during training.

Besides, the code `cnn_active_learning.py` will print a list of indices which correspond to the sentences you need to label.

*Note: Some code snippets are from https://github.com/yoonkim/CNN_sentence and related repositorys.*


## PART2: `PAIA.R` for prior information assisted integrative analysis (Section 2.3)

### Requirements
Code is written in R (4.1.2). The following packages need to be installed before running the code. 
```
grpreg==3.4.0
```

### Input
The input is $M$ datasets, e.g., "dataset1.csv", "dataset2.csv" and "dataset3.csv". In each dataset, each row represents one observation, and each column represents one variable (covariates and response).

### Output
The output is file `m1_coef.csv`, which is the estimated coefficients using the proposed method.
