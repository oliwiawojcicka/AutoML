# AutoML Hyperparameter Tunability Project

## Project description

The goal of this project is to analyze the tunability of hyperparameters for selected machine learning algorithms across multiple datasets. The experiments focus on comparing how different hyperparameter configurations affect model performance and how sampling strategies influence conclusions about algorithm and hyperparameter tunability.

The project investigates three machine learning algorithms:

- XGBoost
- Random Forest
- Elastic Net

The models were evaluated on four datasets representing different prediction problems:

1. Auto loan dataset  
2. Diabetes clinical dataset  
3. Depression dataset  
4. Weather dataset  

For each algorithm, hyperparameter tuning was performed using at least two different sampling strategies:

- **Random Search**, based on sampling configurations from predefined hyperparameter spaces
- **Bayesian Optimization**, used to guide the search towards promising hyperparameter configurations

The tuning histories obtained from these methods were used to analyze the tunability of algorithms and hyperparameters. The analysis follows the ideas described in *Tunability: Importance of Hyperparameters of Machine Learning Algorithms*. In particular, the project aims to identify new default hyperparameter configurations that achieve strong average performance across all datasets and to compare them with other tested configurations.

The analysis also considers:

- how many iterations are needed for each tuning method to obtain stable optimization results
- the motivation for selected hyperparameter ranges based on literature and common practice
- the tunability of individual algorithms
- the tunability of selected hyperparameters
- whether the sampling strategy influences conclusions about tunability, including possible sampling bias

## Datasets used

1. https://www.kaggle.com/datasets/nezukokamaado/auto-loan-dataset
2. https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset
3. https://www.kaggle.com/datasets/anthonytherrien/depression-dataset
4. https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

## Folders and files

- The folders `depression`, `diabetes`, `loan`, and `weather` contain the original datasets, the preprocessed data used in the experiments, and the resulting datasets used in later stages of the project.
- The main notebook is `tunning_all_models.ipynb`. It contains the full experimental workflow, including hyperparameter tuning, default configuration selection, and further analysis.
- The file `functions_tuning.py` contains helper functions used in the main notebook.
- The folder `results_tunning` contains the results of hyperparameter search for each dataset, model, and sampling method, including Random Search and Bayesian Search.
- The folder `raport` contains the final report and plots referenced in the report.
