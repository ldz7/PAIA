# Prior Information Assisted Integrative Analysis of Multiple Datasets*

Code for the Paper *Prior Information Assisted Integrative Analysis of Multiple Datasets*

There are two code files, which corresponds to Section 2.2 and Section 2.3, respectively.


## `cnn_active_learning.py` for prior information extraction (Section 2.2)
Some code comes from https://github.com/yoonkim/CNN_sentence and related repositorys.

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


## `PAIA.R` for prior information assisted integrative analysis (Section 2.3)
### Input
The input is $M$ datasets, e.g., "dataset1.csv", "dataset2.csv" and "dataset3.csv". In each dataset, each row represents each instance and each column represents each variable (covariables and response). 

### Output
The output is file `m1_coef.csv`, which is the estimated coefficients using the proposed method.
