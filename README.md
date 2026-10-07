# MTL-MHOL: Predicting Online Conversions under Delayed Feedback and Data Sparsity

This repository contains the implementation of **MTL-MHOL** (Multi-Task Learning Multi-Head Online Learning), a framework for conversion rate (CVR) prediction that jointly addresses delayed feedback and data sparsity in online advertising.
<p align="center">
  <img src="notebooks/Visualisation MTL MHOL.png" width="400">
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
├── baselines/          # LR, RF, FSIW and ESMM benchmark implementations
├── data/               # Data loading and preprocessing pipelines
├── evaluation/         # Rolling window cross-validation, NLL, RCE, PR-AUC
├── experiments/        # Training scripts and hyperparameter optimization
├── losses/             # Primary task loss, auxiliary task loss, joint loss
├── models/             # MLP and DeepFM backbone implementations
└── notebooks/          # Exploratory analysis and result visualization
```

## Results
Evaluated on a public Criteo attribution dataset and two proprietary session-level customer datasets under rolling window cross-validation, MTL-MHOL:

- Matches the performance of the neural delayed-feedback baseline FSIW and outperforms the entire-space baseline ESMM by 21.5% in RCE on Criteo
- Achieves up to 81% higher RCE than the best classical baseline (LR)
- Correctly captures dataset-specific temporal conversion patterns across delay buckets
- Maintains strong performance across both MLP and DeepFM backbones

### Predictive performance

Mean predictive performance across folds for MTL-MHOL, its ablation variants
(MHOL, MTL, MLP), and the ESMM, FSIW, LR and RF benchmarks. Standard deviations
across folds are shown in parentheses. **Best result in each column in bold.**

| Model | Criteo NLL ↓ | Criteo RCE ↑ | Customer 1 NLL ↓ | Customer 1 RCE ↑ | Customer 2 NLL ↓ | Customer 2 RCE ↑ |
|:------|-------------:|-------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| **MTL-MHOL** | **0.1215 (0.0092)** | **36.704 (0.874)** | **0.1318 (0.0059)** | **4.621 (1.804)** | **0.0937 (0.0060)** | **11.962 (1.467)** |
| MHOL | 0.1222 (0.0094) | 36.282 (0.895) | 0.1330 (0.0059) | 3.752 (1.804) | 0.0949 (0.0057) | 10.821 (0.666) |
| MTL | 0.1350 (0.0052) | 29.499 (2.874) | 0.1331 (0.0047) | 3.733 (0.525) | 0.0946 (0.0058) | 11.107 (1.689) |
| MLP | 0.1362 (0.0065) | 28.880 (2.098) | 0.1334 (0.0079) | 3.486 (1.776) | 0.0953 (0.0061) | 10.437 (1.926) |
| FSIW | 0.1221 (0.0086) | 36.357 (1.025) | — | — | — | — |
| ESMM | 0.1337 (0.0053) | 30.189 (3.017) | — | — | — | — |
| RF | 0.1684 (0.0061) | 12.049 (3.230) | 0.1332 (0.0049) | 3.602 (0.132) | 0.0979 (0.0055) | 7.981 (0.319) |
| LR | 0.1527 (0.0068) | 20.295 (2.565) | 0.1353 (0.0057) | 2.053 (0.865) | 0.0976 (0.0058) | 8.293 (1.051) |

*Table: Mean predictive performance across folds. Standard deviations across
folds are shown in parentheses. ↓ indicates lower is better; ↑ indicates higher
is better. Results for multi-head variants on the private data may be slightly
biased; see the implementation details in the paper.*
 
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
