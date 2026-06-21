# Multi-Task Learning - Multi-Head Online Learning (MTL-MHOL)

This project combines a multi-head online learning model, which tackles the problem of delayed feedback, with a multi-task learning model, which handles data sparsity. Together, this model gives conversion rate predictions.

## Data
The model can be used on the (publically available) Attribution Modeling for Bidding Dataset from Criteo, as well as on the data from a private marketing company. Due to privacy reasons, the company data is not provided.

## Before running the code
- Set up the programming environment:
  - The model is coded in Python 3.12
  - The coding environment used is Databricks using Databricks runtime 17.3LTS with Apache Spark 4.0.0 and Scala 2.13 running a single CPU node rd-fleet.4xlarge with 128GB of memory and 16 cores
  - Install the required packages: `pip install -r requirements.txt`
- Set up the data:
  - The raw dataset from Criteo can be found on and downloaded from the Criteo website (https://ailab.criteo.com/ressources/).
  - This data file is pre-processed in `Data_Pre_Processing.py`, which performs the initial preprocessing of the Criteo dataset by creating temporal and user-behavior features, computing conversion delays, and generating delay-bucket labels for delayed-feedback modeling. It then filters late conversions, downsamples the dataset, encodes conversion-delay buckets as one-hot vectors, and saves the resulting preprocessed dataset as a table for use in the model pipeline.

## Usage
- `General_Data_Processing.py` processes the preprocessed data file such that it can be used for training and testing. Besides the preprocessed data file, it also returns the maximum time horizon H and the array of bucket cutoffs.
- `Time_Specific_Data_Processing.py` processes the data file returned by `General_Data_Processing.py` for time-specific training and evaluation. It creates masks that indicate which target information is available at training and testing time. In addition, it maps unseen categorical values to "unkown" values. It outputs the data file with added mask columns.
- `Evaluation.py` provides the evaluation metrics which are used in the evaluation of MTL-MHOL against several benchmarks. It computes Negative Log Loss (NLL) and Relative Cross Entropy (RCE).
- `Random_Forest.py` implements a complete Random Forest (RF) conversion-rate prediction pipeline, which is used as a benchmark in our paper.
- `Logistic_Regression.py` trains and evaluates a logistic regression, which is used as a benchmark model in our paper. The model makes CVR predictions and returns the evaluation metrics of these predictions.
- `DeepFM.py` implements the MTL-MHOL framework with a DeepFM workhorse model. This is used to evaluate whether our framework is agnostic to the choice of workhorse model.
- `Flag_Models.py` implements the main MTL-MHOL model. This MLP-based architecture supports single-head (MLP), multi-task learning (MTL), mutli-head learning (MHOL), and the full MTL-MHOL model.
- `HPTuning.py` performs hyperparameter tuning using Tree-structured Parzen Estimator (TPE) with Optuna. It uses inner fold cross validation through a rolling window and selects the configuration of hyperparameters that achieve the best aggregated RCE score.
- `HPTuner_RF.py` performs hyperparamter tuning in the same way as `HPTuning.py`, but is adjusted to work for RF so this module has no dependency on it.
- `Main.py` runs the full pipline. It loads the preprocessed datasets, creates rolling outer train and test fold, tunes the hyperparameters in the inner folds, and trains and tests the selected model on each fold. It evaluates the performance and summarizes the results across the folds. Moreover, it outputs the evaluation metrics per fold as well as the overall average performance summary.
