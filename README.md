# Multi-Task Learning - Multi-Head Online Learning (MTL-MHOL)

This project combined a multi-head online learning model, which tackles the problem of delayed feedback, with a multi-task learning model, which handles data sparsity. Together, this model gives conversion rates predictions. The model is coded in Python.

## Data
The model can be used on the Attribution Modeling for Bidding dataset from Criteo, as well as on the data provided by a private marketing company.

## Usage
- `General_Data_Processing.py` preprocesses the raw data file such that it can be used for training and testing. Besides the preprocessed data file, it also returns the maximum time horizon H and the array of bucket cutoffs.
- `Time_Specific_Data_Processing.py` preprocesses the data file returned by `General_Data_Processing.py` for time-specific training and evaluation. It creates masks that indicate which target information is available at training and testing time. In addition, it maps unseen categorical values to "unkown" values. It outputs the data file with added mask columns.
- `Evaluation.py` provides the evaluation metrics which are used in the evaluation of MTL-MHOL against several benchmarks. It computes Negative Log Loss (NLL) and Relative Cross Entropy (RCE).
- `HPTuning.py` perform hyperparameter tuning using Tree-structured Parzen Estimator (TPE) with Optuna. It uses inner fold cross validation through a rolling window and selects the configuration of hyperparameters that achieve the best aggregated RCE score.
- `HPTuner_RF.py` ...
- `Random_Forest.py` ...
- `Logistic_Regression.py` trains and evaluates a logistic regression, which is used as a benchmark model in our paper. The model makes CVR predictions and returns the evaluation metrics of these predictions.
- `DeepFM.py` implements the MTL-MHOL framework with a DeepFM workhorse model. This is used to evaluate whether our framework is agnostic to the choice of workhorse model.
- `Flag_Models.py` implements the main MTL-MHOL model. This MLP-based architecture supports single-head (MLP), multi-task learning (MTL), mutli-head learning (MHOL), and the full MTL-MHOL model. 
- `Main.py` runs the full pipline. It loads the preprocessed datasets, creates rolling outer train and test fold, tunes the hyperparameters in the inner folds, and trains and tests the selected model on each fold. It evaluates the performance and summarizes the results across the folds. Moreover, it outputs the evaluation metrics per fold as well as the overall average performance summary.
