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
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2" align="center">Criteo</th>
      <th colspan="2" align="center">Customer 1</th>
      <th colspan="2" align="center">Customer 2</th>
    </tr>
    <tr>
      <th align="center">NLL ↓</th>
      <th align="center">RCE ↑</th>
      <th align="center">NLL ↓</th>
      <th align="center">RCE ↑</th>
      <th align="center">NLL ↓</th>
      <th align="center">RCE ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MTL-MHOL</strong></td>
      <td align="center"><strong>0.1215</strong><br><sub><strong>(0.0092)</strong></sub></td>
      <td align="center"><strong>36.704</strong><br><sub><strong>(0.874)</strong></sub></td>
      <td align="center"><strong>0.1318</strong><br><sub><strong>(0.0059)</strong></sub></td>
      <td align="center"><strong>4.621</strong><br><sub><strong>(1.804)</strong></sub></td>
      <td align="center"><strong>0.0937</strong><br><sub><strong>(0.0060)</strong></sub></td>
      <td align="center"><strong>11.962</strong><br><sub><strong>(1.467)</strong></sub></td>
    </tr>

    <tr>
      <td>MHOL</td>
      <td align="center">0.1222<br><sub>(0.0094)</sub></td>
      <td align="center">36.282<br><sub>(0.895)</sub></td>
      <td align="center">0.1330<br><sub>(0.0059)</sub></td>
      <td align="center">3.752<br><sub>(1.804)</sub></td>
      <td align="center">0.0949<br><sub>(0.0057)</sub></td>
      <td align="center">10.821<br><sub>(0.666)</sub></td>
    </tr>

    <tr>
      <td>MTL</td>
      <td align="center">0.1350<br><sub>(0.0052)</sub></td>
      <td align="center">29.499<br><sub>(2.874)</sub></td>
      <td align="center">0.1331<br><sub>(0.0047)</sub></td>
      <td align="center">3.733<br><sub>(0.525)</sub></td>
      <td align="center">0.0946<br><sub>(0.0058)</sub></td>
      <td align="center">11.107<br><sub>(1.689)</sub></td>
    </tr>

    <tr>
      <td>MLP</td>
      <td align="center">0.1362<br><sub>(0.0065)</sub></td>
      <td align="center">28.880<br><sub>(2.098)</sub></td>
      <td align="center">0.1334<br><sub>(0.0079)</sub></td>
      <td align="center">3.486<br><sub>(1.776)</sub></td>
      <td align="center">0.0953<br><sub>(0.0061)</sub></td>
      <td align="center">10.437<br><sub>(1.926)</sub></td>
    </tr>

    <tr>
      <td>FSIW</td>
      <td align="center">0.1221<br><sub>(0.0086)</sub></td>
      <td align="center">36.357<br><sub>(1.025)</sub></td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
    </tr>

    <tr>
      <td>ESMM</td>
      <td align="center">0.1337<br><sub>(0.0053)</sub></td>
      <td align="center">30.189<br><sub>(3.017)</sub></td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
      <td align="center">—</td>
    </tr>

    <tr>
      <td>RF</td>
      <td align="center">0.1684<br><sub>(0.0061)</sub></td>
      <td align="center">12.049<br><sub>(3.230)</sub></td>
      <td align="center">0.1332<br><sub>(0.0049)</sub></td>
      <td align="center">3.602<br><sub>(0.132)</sub></td>
      <td align="center">0.0979<br><sub>(0.0055)</sub></td>
      <td align="center">7.981<br><sub>(0.319)</sub></td>
    </tr>

    <tr>
      <td>LR</td>
      <td align="center">0.1527<br><sub>(0.0068)</sub></td>
      <td align="center">20.295<br><sub>(2.565)</sub></td>
      <td align="center">0.1353<br><sub>(0.0057)</sub></td>
      <td align="center">2.053<br><sub>(0.865)</sub></td>
      <td align="center">0.0976<br><sub>(0.0058)</sub></td>
      <td align="center">8.293<br><sub>(1.051)</sub></td>
    </tr>
  </tbody>
</table>

<p>
  <sub>
    Mean predictive performance across folds. Standard deviations across folds
    are shown in parentheses. <strong>Best result in each column in bold.</strong>
    ↓ indicates lower is better; ↑ indicates higher is better.
  </sub>
</p>
 
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
