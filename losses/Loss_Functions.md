# Loss Functions

This document contains the loss functions defined in the paper.

## Primary Task Loss

Each bucket head is trained with a masked binary cross-entropy loss restricted to its risk set.

Let $y_{ib} \in \{0,1\}$, $m_{ib} \in \{0,1\}$, and $r_{ib} \in \{0,1\}$ denote the conversion label, maturity mask, and risk-set indicator for observation $i$ in bucket $b$, respectively. Let $\hat{h}_{ib} \in (0,1)$ be the predicted hazard.

The joint eligibility weight is

$$\omega_{ib} = m_{ib}r_{ib}.$$

The loss for bucket $b$ is

$$\mathcal{L}_{\mathrm{CE}_b} = -\frac{1}{\sum_{i=1}^{N}\omega_{ib}} \sum_{i=1}^{N} \omega_{ib} \left[ y_{ib}\log(\hat{h}_{ib}) + (1-y_{ib})\log(1-\hat{h}_{ib})n\right]. $$

This is the average negative log-likelihood over all mature, at-risk observations in bucket $b$.

The risk-set restriction is essential: without it, each head converges toward the marginal

$$P(y_{ib}=1 \mid X_i)$$

rather than the conditional hazard.

---

## Auxiliary Task Loss

### Binary Auxiliary Target

For a binary auxiliary target, the primary loss is reused with two simplifications: there is only one head ($B=1$), and the maturity mask is replaced by an auxiliary mask $a_i \in \{0,1\}$.

The mask is 1 by default and is set to 0 only when a maturity timestamp is available and the auxiliary outcome is not yet observable at cutoff $\phi$.

The resulting loss is

$$\mathcal{L}_{\mathrm{aux}}^{\mathrm{binary}}= -\frac{1}{\sum_{i=1}^{N}a_i} \sum_{i=1}^{N} a_i \left[ y_i^{\mathrm{aux}}\log(\hat{h}_i^{\mathrm{aux}}) + (1-y_i^{\mathrm{aux}}) \log(1-\hat{h}_i^{\mathrm{aux}}) \right], $$

where $\hat{h}_i^{\mathrm{aux}}$ is the predicted probability that

$$y_i^{\mathrm{aux}} = 1.$$

### Continuous Auxiliary Target

For a continuous auxiliary target, the auxiliary head predicts $Q$ conditional quantiles simultaneously using a pinball loss.

Let $\hat{y}_{\tau_q,i}^{\mathrm{aux}}$ be the predicted conditional quantile at level

$$\tau_q \in (0,1), \quad q=1,\ldots,Q.$$

The pinball loss is

$$ \mathcal{L}_{\mathrm{pinball}} = \frac{1}{\left(\sum_{i=1}^{N}a_i\right)Q} \sum_{i=1}^{N} a_i \sum_{q=1}^{Q} \max \left( \tau_q \left( y_i^{\mathrm{aux}} -\hat{y}_{\tau_q,i}^{\mathrm{aux}}\right),(\tau_q-1)\left(y_i^{\mathrm{aux}}-\hat{y}_{\tau_q,i}^{\mathrm{aux}}\right)\right).$$

The pinball loss applies an asymmetric penalty to target each conditional quantile of the auxiliary distribution.

---

## Monotonicity Penalty

To ensure that the predicted quantiles form a valid conditional distribution, a soft monotonicity penalty is added to penalize violations of

$$\hat{y}_{\tau_q,i}^{\mathrm{aux}}\leq\hat{y}_{\tau_{q+1},i}^{\mathrm{aux}}.$$

The penalty is

$$\mathcal{P} =\frac{1}{\left(\sum_{i=1}^{N}a_i\right)(Q-1)}\sum_{i=1}^{N}a_i\sum_{q=1}^{Q-1}\max\left(0,\hat{y}_{\tau_q,i}^{\mathrm{aux}}-\hat{y}_{\tau_{q+1},i}^{\mathrm{aux}}\right).$$

The combined auxiliary loss is

$$\mathcal{L}_{\mathrm{aux}}^{\mathrm{distr}}=\mathcal{L}_{\mathrm{pinball}}+\gamma\mathcal{P},$$

where $\gamma \geq 0$ controls the strength of the monotonicity penalty.

---

## Joint Loss

The overall training objective for bucket $b$ is

$$\mathcal{L}_b=\mathcal{L}_{\mathrm{CE}_b}+\lambda_{\mathrm{aux}}\mathcal{L}_{\mathrm{aux}},$$

where $\lambda_{\mathrm{aux}}$ scales the auxiliary contribution and

$$
\mathcal{L}_{\mathrm{aux}}
$$

is either

$$
\mathcal{L}_{\mathrm{aux}}^{\mathrm{binary}}
$$

or

$$
\mathcal{L}_{\mathrm{aux}}^{\mathrm{distr}}.
$$

The primary loss remains the dominant training signal, with the auxiliary term acting as a data-adaptive regularizer.
