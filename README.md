# MTL-MHOL: Predicting Online Conversions under Delayed Feedback and Data Sparsity

This repository contains the implementation of **MTL-MHOL** (Multi-Task Learning Multi-Head Online Learning), a framework for conversion rate (CVR) prediction that jointly addresses delayed feedback and data sparsity in online advertising.
<p align="center">
  <img src="Figures/Visualisation MTL MHOL.png" width="500">
  <br>
  <em>Figure 1: Visualisation of MTL-MHOL.</em>
</p>

## Abstract
This paper proposes the model-agnostic Multi-Task Learning Multi-Head Online Learning (MTL-MHOL) framework for conversion rate (CVR) prediction. Existing approaches typically address key challenges in CVR prediction, such as delayed feedback and data sparsity, in isolation or lack flexibility and practical applicability. MTL-MHOL adopts a time bucketing approach to account for delayed feedback and combines it with multi-task learning of an auxiliary task to mitigate data sparsity. We evaluate the framework on proprietary datasets from a private company and on a public dataset from Criteo. MTL-MHOL matches or outperforms all benchmark models in terms of Negative Log-Likelihood (NLL) and Relative Cross Entropy (RCE), and it correctly captures temporal trends in the data using an MLP backbone, while maintaining strong performance with a DeepFM backbone. In particular, MTL-MHOL matches the performance of the advanced delayed-feedback method FSIW, outperforms the entire-space approach ESMM by 21.5\% in RCE, and achieves up to 81\% lift in RCE compared to the best-performing classical benchmark.

## Overview

MTL-MHOL combines two complementary components:

- **Multi-Head Online Learning (MHOL):** The conversion horizon is partitioned into mutually exclusive delay buckets. A dedicated head estimates a bucket-specific hazard probability for each bucket, with maturity masking and risk-set restriction ensuring that only eligible observations contribute to each head's training loss. Bucket-level hazards are aggregated into an overall CVR estimate via a survival-based formulation.

- **Multi-Task Learning (MTL):** An auxiliary engagement task is trained jointly with the primary CVR task via hard parameter-sharing, enriching the shared trunk with a denser supervision signal. The auxiliary target is either binary (e.g., click indicator) or continuous (e.g., log-transformed number of distinct pages visited), where the continuous case uses a multi-quantile regression head with a soft monotonicity penalty.

The framework is model-agnostic within the neural network family and is demonstrated with both MLP and DeepFM backbones.

## Repository Structure
```
├── data/               # Data loading and preprocessing pipelines
├── models/             # MLP and DeepFM backbone implementations
├── losses/             # Primary task loss, auxiliary task loss, joint loss
├── evaluation/         # Rolling window cross-validation, NLL, RCE, PR-AUC
├── baselines/          # FSIW and ESMM benchmark implementations
├── experiments/        # Training scripts and hyperparameter optimization
└── notebooks/          # Exploratory analysis and result visualization
```

## Results
Evaluated on a public Criteo attribution dataset and two proprietary session-level customer datasets under rolling window cross-validation, MTL-MHOL:

- Matches the performance of the neural delayed-feedback baseline FSIW and outperforms the entire-space baseline ESMM by 21.5% in RCE on Criteo
- Achieves up to 81% higher RCE than the best classical baseline (LR)
- Correctly captures dataset-specific temporal conversion patterns across delay buckets
- Maintains strong performance across both MLP and DeepFM backbones

## Install Instructions
- Set up the programming environment:
  - The model is coded in Python 3.12.
  - The coding environment used is Databricks using Databricks runtime 17.3LTS with Apache Spark 4.0.0 and Scala 2.13 running a single CPU node rd-fleet.4xlarge with 128GB of memory and 16 cores.
  - Install the required packages: `pip install -r requirements.txt`.
- Set up the data:
  - The unprocessed dataset from Criteo can be found on and downloaded from the Criteo website (https://ailab.criteo.com/ressources/).
  - This data file is pre-processed in `Data_Pre_Processing.py`, which performs the initial preprocessing of the Criteo dataset by creating temporal and user-behavior features, computing conversion delays, and generating delay-bucket labels for delayed-feedback modeling. It then filters late conversions, downsamples the dataset, encodes conversion-delay buckets as one-hot vectors, and saves the resulting preprocessed dataset as a table for use in the model pipeline.
 
## Citation

If you use this code in your research, please cite:

```bibtex
@article{tejeravicente2026mtlmhol,
  title     = {MTL-MHOL: Predicting Online Conversions under Delayed
               Feedback and Data Sparsity},
  author    = {Tejera Vicente, Mario and Hagen, Eva and
               van Breukelen, Emma and van de Vijver, Quinten and
               Gruber, Kathrin},
  journal   = {Transactions on Machine Learning Research},
  year      = {2026}
}
```

## Requirements

```
torch==2.7.0
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.6.1
optuna==3.6.1
lightgbm
```

## License

This project is released under the MIT License.

## References
- [Multi-head online learning for delayed feedback modeling.](https://arxiv.org/pdf/2205.12406)
- [DeepFM: a factorization-machine based neural network for CTR prediction.](https://arxiv.org/pdf/1703.04247)
- [Entire space multi-task model: An effective approach for estimating post-click conversion rate.](https://dl.acm.org/doi/pdf/10.1145/3209978.3210104)
- [A feedback shift correction in predicting conversion rates under delayed feedback.](https://dl.acm.org/doi/pdf/10.1145/3366423.3380032)
