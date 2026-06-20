import numpy as np
import pandas as pd
import torch
import gc
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from Evaluation import evaluation

PLATT_FIT_SLOPE: bool = False   # False = intercept only (recommended default)
                                # True  = fit slope + intercept

def plot_cvr_boxplot(
    y_true_bin: np.ndarray,
    p_pred: np.ndarray,
    *,
    title: str = "Test-set CVR predictions by true outcome",
    showfliers: bool = False,
):
    y_true_bin = np.asarray(y_true_bin).astype(int).ravel()
    p_pred     = np.asarray(p_pred).astype(float).ravel()
    p0 = p_pred[y_true_bin == 0]
    p1 = p_pred[y_true_bin == 1]
    fig, ax = plt.subplots()
    ax.boxplot([p0, p1], labels=["No conversion (y=0)", "Conversion (y=1)"],
               showfliers=showfliers)
    ax.set_ylabel("Predicted conversion probability")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_aux_binary_boxplot(
    aux_y_true: np.ndarray,
    aux_prob: np.ndarray,
    *,
    title: str = "Test-set AUX (binary) predictions by true AUX label",
    showfliers: bool = False,
):
    aux_y_true = np.asarray(aux_y_true).astype(int).ravel()
    aux_prob   = np.asarray(aux_prob).astype(float).ravel()
    a0 = aux_prob[aux_y_true == 0]
    a1 = aux_prob[aux_y_true == 1]
    fig, ax = plt.subplots()
    ax.boxplot([a0, a1], labels=["Aux=0", "Aux=1"], showfliers=showfliers)
    ax.set_ylabel("Predicted auxiliary probability (P(aux=1))")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def _as_ts(x):
    if isinstance(x, Column):
        return F.to_timestamp(x)
    return F.to_timestamp(F.lit(str(x)))


def _safe_log1p_nonneg(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))


# Platt calibration helpers
# (only called when calibrate=True)
def fit_platt(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    fit_slope: bool = False,
    max_iter: int = 100,
) -> tuple[float, float]:

    logits = np.asarray(logits, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    if logits.size == 0:
        return 1.0, 0.0
    y_sum = float(y_true.sum())
    if y_sum == 0.0 or y_sum == float(y_true.size):
        print("  [Platt] WARNING: calibration set has no positives or no negatives -- skipping.")
        return 1.0, 0.0

    x_t = torch.tensor(logits, dtype=torch.float64)
    y_t = torch.tensor(y_true, dtype=torch.float64)

    a = torch.tensor([1.0], dtype=torch.float64, requires_grad=bool(fit_slope))
    b = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    params = [a, b] if fit_slope else [b]

    def _closure():
        opt.zero_grad(set_to_none=True)
        z    = a.detach() * x_t + b if not fit_slope else a * x_t + b
        loss = nn.functional.binary_cross_entropy_with_logits(z, y_t)
        loss.backward()
        return loss

    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter,
                             line_search_fn="strong_wolfe")
    opt.step(_closure)

    a_val = float(a.detach().item())
    b_val = float(b.detach().item())
    print(f"  [Platt] fitted  a={a_val:.4f}  b={b_val:.4f}  "
          f"({'slope+intercept' if fit_slope else 'intercept only'})")
    return a_val, b_val


def apply_platt_to_logits(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return (a * np.asarray(logits, dtype=np.float64) + b).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float64)

def hazard_agg_np(logits: np.ndarray) -> np.ndarray:
    h = sigmoid(logits)
    return 1.0 - np.prod(1.0 - h, axis=1)


def _calibrate_single_head(
    logits_cal:  np.ndarray,   # [N_cal]
    y_cal:       np.ndarray,   # [N_cal]  binary
    logits_eval: np.ndarray,   # [N_eval]
    y_eval:      np.ndarray,   # [N_eval] binary
    *,
    fit_slope: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    a, b = fit_platt(logits_cal, y_cal, fit_slope=fit_slope)
    p    = sigmoid(apply_platt_to_logits(logits_eval, a, b))
    return p, y_eval.astype(np.float64), a, b

def _calibrate_multi_head(
    logits_cal:  np.ndarray,   # [N_cal,  K]
    y_cal:       np.ndarray,   # [N_cal,  K]  bucket_vec
    logits_eval: np.ndarray,   # [N_eval, K]
    y_eval:      np.ndarray,   # [N_eval, K]
    *,
    fit_slope: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:

    # Aggregate per-head logits -> session probability -> session logit
    p_cal     = hazard_agg_np(logits_cal)                          # [N_cal]
    agg_logit_cal = np.log(p_cal + 1e-12) - np.log1p(-p_cal + 1e-12)  # logit(p_cal)

    # Session-level binary label
    y_cal_bin = (y_cal.sum(axis=1) > 0).astype(np.float64)        # [N_cal]

    # Fit Platt on (aggregated logit, session binary label)
    a, b = fit_platt(agg_logit_cal, y_cal_bin, fit_slope=fit_slope)

    # aggregate eval logits, convert to logit space, shift, sigmoid
    p_eval_raw    = hazard_agg_np(logits_eval)                     # [N_eval]
    agg_logit_eval = np.log(p_eval_raw + 1e-12) - np.log1p(-p_eval_raw + 1e-12)
    p_eval        = sigmoid(a * agg_logit_eval + b)                # [N_eval]

    y_eval_bin = (y_eval.sum(axis=1) > 0).astype(np.float64)
    return p_eval, y_eval_bin, a, b

  
def _encode_categoricals_train_test(
    train_pdf: pd.DataFrame,
    test_pdf:  pd.DataFrame,
    *,
    CAT_INT_COLS: list[str],
):
    CAT_INT_COLS = CAT_INT_COLS or []
    mats_tr, mats_te, sizes = [], [], []

    for c in CAT_INT_COLS:
        if c not in train_pdf.columns:
            continue
        tr_series = pd.to_numeric(train_pdf[c], errors="coerce").fillna(0).astype("int64")
        te_series = pd.to_numeric(test_pdf[c],  errors="coerce").fillna(0).astype("int64")
        cats    = pd.Index(tr_series.unique())
        mapping = {v: i for i, v in enumerate(cats)}
        tr = tr_series.map(mapping).fillna(0).astype(np.int64).values
        te = te_series.map(mapping).fillna(0).astype(np.int64).values
        mats_tr.append(tr.reshape(-1, 1))
        mats_te.append(te.reshape(-1, 1))
        sizes.append(max(len(cats), 1))

    if mats_tr:
        Xc_tr = np.concatenate(mats_tr, axis=1)
        Xc_te = np.concatenate(mats_te, axis=1)
    else:
        Xc_tr = np.zeros((len(train_pdf), 0), dtype=np.int64)
        Xc_te = np.zeros((len(test_pdf),  0), dtype=np.int64)

    return Xc_tr, Xc_te, [max(1, int(s)) for s in sizes]


def _prepare_numeric_train_test(
    train_pdf: pd.DataFrame,
    test_pdf:  pd.DataFrame,
    *,
    NUM_COLS: list[str],
):
    NUM_COLS = [c for c in (NUM_COLS or []) if c in train_pdf.columns]
    Xtr, Xte = [], []

    for c in NUM_COLS:
        tr = pd.to_numeric(train_pdf[c], errors="coerce").astype(float)
        te = pd.to_numeric(test_pdf[c],  errors="coerce").astype(float)

        miss_tr = tr.isna().astype(np.float32).values.reshape(-1, 1)
        miss_te = te.isna().astype(np.float32).values.reshape(-1, 1)

        trv_t = _safe_log1p_nonneg(tr.values.astype(np.float32))
        tev_t = _safe_log1p_nonneg(te.values.astype(np.float32))

        obs = ~np.isnan(trv_t)
        mu  = float(trv_t[obs].mean()) if obs.any() else 0.0
        sd  = float(trv_t[obs].std(ddof=0)) if obs.any() else 1.0
        if sd == 0.0:
            sd = 1.0

        trv_z = np.nan_to_num((trv_t - mu) / sd, nan=0.0).astype(np.float32).reshape(-1, 1)
        tev_z = np.nan_to_num((tev_t - mu) / sd, nan=0.0).astype(np.float32).reshape(-1, 1)

        Xtr += [trv_z, miss_tr]
        Xte += [tev_z, miss_te]

    if Xtr:
        return (np.concatenate(Xtr, axis=1).astype(np.float32),
                np.concatenate(Xte, axis=1).astype(np.float32))

    return (np.zeros((len(train_pdf), 0), dtype=np.float32),
            np.zeros((len(test_pdf),  0), dtype=np.float32))


class DS_Agnostic(Dataset):
    def __init__(self, Xc, Xn, y, mask=None, y_aux=None):
        self.Xc    = torch.tensor(Xc, dtype=torch.long)
        self.Xn    = torch.tensor(Xn, dtype=torch.float32)
        self.y     = torch.tensor(y,  dtype=torch.float32)
        self.mask  = None if mask  is None else torch.tensor(mask,  dtype=torch.float32)
        self.y_aux = None if y_aux is None else torch.tensor(y_aux, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        Xc, Xn, y = self.Xc[i], self.Xn[i], self.y[i]
        if self.mask is None and self.y_aux is None:
            return Xc, Xn, y
        if self.mask is not None and self.y_aux is None:
            return Xc, Xn, y, self.mask[i]
        if self.mask is None and self.y_aux is not None:
            return Xc, Xn, y, self.y_aux[i]
        return Xc, Xn, y, self.mask[i], self.y_aux[i]


# DeepFM model
class DeepFM_Agnostic(nn.Module):
    def __init__(
        self,
        cat_sizes: list[int],
        num_dim: int,
        *,
        emb_dim: int = 8,
        deep_hidden: tuple = (512, 512),
        dropout: float = 0.1,
        n_heads: int = 1,
        use_aux: bool = False,
    ):
        super().__init__()
        self.n_cat_fields = len(cat_sizes)
        self.num_dim      = int(num_dim)
        self.emb_dim      = int(emb_dim)
        self.n_heads      = int(n_heads)
        self.use_aux      = bool(use_aux)

        self.v_cat = nn.ModuleList([nn.Embedding(int(s), self.emb_dim) for s in cat_sizes])
        self.w_cat = nn.ModuleList([nn.Embedding(int(s), 1)            for s in cat_sizes])

        if self.num_dim > 0:
            self.v_num = nn.Parameter(torch.empty(self.num_dim, self.emb_dim))
            self.w_num = nn.Linear(self.num_dim, 1, bias=True)
        else:
            self.v_num = None
            self.w_num = None

        self.bias = nn.Parameter(torch.zeros(1))

        deep_in = (self.n_cat_fields + self.num_dim) * self.emb_dim
        layers, prev = [], deep_in
        for h in deep_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(float(dropout))]
            prev = h
        self.deep      = nn.Sequential(*layers)
        self._deep_out = prev

        self.fm_to_heads   = nn.Linear(1,    self.n_heads, bias=True)
        self.deep_to_heads = nn.Linear(prev, self.n_heads, bias=True)

        if self.use_aux:
            self.fm_to_aux   = nn.Linear(1,    1, bias=True)
            self.deep_to_aux = nn.Linear(prev, 1, bias=True)
        else:
            self.fm_to_aux = self.deep_to_aux = None

        for emb in self.v_cat:
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        for w in self.w_cat:
            nn.init.zeros_(w.weight)
        if self.v_num is not None:
            nn.init.normal_(self.v_num, mean=0.0, std=0.01)
        if self.w_num is not None:
            nn.init.zeros_(self.w_num.weight)
            nn.init.zeros_(self.w_num.bias)
        for layer in [self.fm_to_heads, self.deep_to_heads]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.use_aux:
            for layer in [self.fm_to_aux, self.deep_to_aux]:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _all_embeddings(self, Xc, Xn):
        parts = []
        if self.n_cat_fields > 0:
            parts.append(
                torch.stack([emb(Xc[:, i]) for i, emb in enumerate(self.v_cat)], dim=1)
            )
        if self.v_num is not None:
            parts.append(Xn.unsqueeze(-1) * self.v_num.unsqueeze(0))
        return torch.cat(parts, dim=1) if parts else None

    def _fm_scalar(self, Xc, Xn, E):
        B     = Xc.shape[0]
        first = self.bias.expand(B, 1).clone()
        for i, w in enumerate(self.w_cat):
            first = first + w(Xc[:, i])
        if self.w_num is not None:
            first = first + self.w_num(Xn)
        if E is None:
            return torch.clamp(first, -20.0, 20.0)
        sum_e  = E.sum(dim=1)
        sum_e2 = (E * E).sum(dim=1)
        second = 0.5 * ((sum_e * sum_e) - sum_e2).sum(dim=1, keepdim=True)
        return torch.clamp(first + second, -20.0, 20.0)

    def _deep_hidden(self, E):
        if E is None:
            return None
        return self.deep(E.reshape(E.shape[0], -1))

    def forward(self, Xc, Xn):
        E      = self._all_embeddings(Xc, Xn)
        fm     = self._fm_scalar(Xc, Xn, E)
        h      = self._deep_hidden(E)
        logits = self.fm_to_heads(fm) + self.deep_to_heads(h)
        if self.n_heads == 1:
            logits = logits.squeeze(1)
        if self.use_aux:
            aux_logit = (self.fm_to_aux(fm) + self.deep_to_aux(h)).squeeze(1)
            return logits, aux_logit
        return logits



# Aggregation and loss
def hazard_agg(logits: torch.Tensor) -> torch.Tensor:
    """P(convert) = 1 - prod_k (1 - sigmoid(logit_k))"""
    h = torch.sigmoid(logits)
    return 1.0 - torch.prod(1.0 - h, dim=1)


def masked_bce_with_logits(logits, y, m):
    bce = nn.BCEWithLogitsLoss(reduction="none")
    per = bce(logits, y) * m
    return per.sum() / m.sum().clamp_min(1.0)


# Logit collection
def _collect_logits(
    model, loader, device, *,
    mh_enabled:    bool,
    aux_enabled:   bool,
    mask_col_last: bool = True,   # True  => keep only rows where m[:,-1]==1
                                  # False => keep all rows (calibration pass)
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    logits_list, ys_list = [], []

    with torch.no_grad():
        for batch in loader:
            Xc      = batch[0].to(device)
            Xn      = batch[1].to(device)
            y_batch = batch[2].to(device)

            if mh_enabled and aux_enabled:
                m_batch       = batch[3].to(device)
                raw_logits, _ = model(Xc, Xn)
            elif mh_enabled:
                m_batch    = batch[3].to(device)
                raw_logits = model(Xc, Xn)
            elif aux_enabled:
                raw_logits, _ = model(Xc, Xn)
            else:
                raw_logits = model(Xc, Xn)

            if mh_enabled:
                if mask_col_last:
                    keep = m_batch[:, -1] > 0
                    logits_list.append(raw_logits[keep].cpu().numpy())
                    ys_list.append(y_batch[keep].cpu().numpy())   # [keep, K]
                else:
                    logits_list.append(raw_logits.cpu().numpy())
                    ys_list.append(y_batch.cpu().numpy())         # [B, K]
            else:
                logits_list.append(raw_logits.cpu().numpy())
                ys_list.append(y_batch.cpu().numpy())

    logits_np = np.concatenate(logits_list).astype(np.float64)
    y_np      = np.concatenate(ys_list).astype(np.float64)
    return logits_np, y_np


def deep_fm(
    *,
    df,
    train_end,
    test_end=None,
    args=None,
    NUM_COLS,
    CAT_INT_COLS,
    TARGET="conversion",
    START_TS="timestamp_dt",
    HEAD_COL="bucket_vec",
    MASK_TRAIN_COL="mask_train",
    MASK_TEST_COL="mask_eval",
    seed: int = 0,
    hparams: dict | None = None,
    spark=None,
    outer: bool = False,
    calibrate: bool = True,
    platt_fit_slope: bool = PLATT_FIT_SLOPE,
    **kwargs,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    g = torch.Generator()
    g.manual_seed(int(seed))

    hp = dict(
        batch_size=2048,
        learning_rate=1e-3,
        l2_weight_decay=1e-5,
        dropout_rate=0.1,
        emb_dim=8,
        deep_hidden=(512, 512),
        epochs=20,
        aux_weight=1,
    )
    if hparams:
        hp.update(hparams)

    print(f"Calibration: {'enabled' if calibrate else 'disabled'}")

    # Mode flags
    n_buckets  = args.n_buckets
    AUX_TARGET = getattr(args, "aux_target", None)
    mh_enabled  = n_buckets > 1
    aux_enabled = AUX_TARGET is not None

    # Drop unused columns
    drop_cols = ("bucket", "bucket_vec", "mask_train", "mask_eval", "aux_mask")
    if mh_enabled:
        remove_set = {"bucket", "bucket_vec", "mask_train", "mask_eval"}
        drop_cols  = tuple(c for c in drop_cols if c not in remove_set)
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(c)

    # Temporal split
    ASOF_TRAIN = _as_ts(train_end)
    ASOF_TEST  = _as_ts(test_end)

    if mh_enabled:
        train_sdf = (
            df.where(F.col(START_TS) < ASOF_TRAIN)
              .where(F.aggregate(F.col(MASK_TRAIN_COL), F.lit(0),
                                 lambda acc, x: acc + x) > 0)
        )
    else:
        train_sdf = (
            df.where(F.col(START_TS) < ASOF_TRAIN)
              .where(F.element_at(F.col(MASK_TRAIN_COL), -1) == 1)
        )

    if calibrate:
        ASOF_CAL = F.to_timestamp(F.date_add(F.date_trunc("day", ASOF_TRAIN), 1))
        cal_sdf  = (df.where(F.col(START_TS) >= ASOF_TRAIN)
                      .where(F.col(START_TS) <  ASOF_CAL))
        test_sdf = (df.where(F.col(START_TS) >= ASOF_CAL)
                      .where(F.col(START_TS) <  ASOF_TEST))
    else:
        # test starts at train_end directly; no calibration set
        test_sdf = (df.where(F.col(START_TS) >= ASOF_TRAIN)
                      .where(F.col(START_TS) <  ASOF_TEST))


    train_pdf = train_sdf.toPandas()
    test_pdf  = test_sdf.toPandas()
    if calibrate:
        cal_pdf = cal_sdf.toPandas()
        print(f"Sizes -- train: {len(train_pdf)}, cal: {len(cal_pdf)}, test: {len(test_pdf)}")
    else:
        print(f"Sizes -- train: {len(train_pdf)}, test: {len(test_pdf)}")


    def stack_array_col(pdf, colname):
        if colname not in pdf.columns:
            raise ValueError(f"Missing column '{colname}'.")
        return np.stack(pdf[colname].values).astype(np.float32)

    if mh_enabled and aux_enabled:
        y_train     = stack_array_col(train_pdf, HEAD_COL)
        m_train     = stack_array_col(train_pdf, MASK_TRAIN_COL)
        y_test      = stack_array_col(test_pdf,  HEAD_COL)
        m_test      = stack_array_col(test_pdf,  MASK_TEST_COL)
        y_train_aux = np.clip(train_pdf[AUX_TARGET].astype(np.float32).values, 0.0, 1.0)
        y_test_aux  = np.clip(test_pdf[AUX_TARGET].astype(np.float32).values,  0.0, 1.0)
        if calibrate:
            y_cal     = stack_array_col(cal_pdf, HEAD_COL)
            m_cal     = stack_array_col(cal_pdf, MASK_TEST_COL)
            y_cal_aux = np.clip(cal_pdf[AUX_TARGET].astype(np.float32).values, 0.0, 1.0)

    elif mh_enabled:
        y_train = stack_array_col(train_pdf, HEAD_COL)
        m_train = stack_array_col(train_pdf, MASK_TRAIN_COL)
        y_test  = stack_array_col(test_pdf,  HEAD_COL)
        m_test  = stack_array_col(test_pdf,  MASK_TEST_COL)
        if calibrate:
            y_cal = stack_array_col(cal_pdf, HEAD_COL)
            m_cal = stack_array_col(cal_pdf, MASK_TEST_COL)

    elif aux_enabled:
        y_train     = train_pdf[TARGET].astype(np.float32).values
        y_test      = test_pdf[TARGET].astype(np.float32).values
        y_train_aux = np.clip(train_pdf[AUX_TARGET].astype(np.float32).values, 0.0, 1.0)
        y_test_aux  = np.clip(test_pdf[AUX_TARGET].astype(np.float32).values,  0.0, 1.0)
        if calibrate:
            y_cal     = cal_pdf[TARGET].astype(np.float32).values
            y_cal_aux = np.clip(cal_pdf[AUX_TARGET].astype(np.float32).values, 0.0, 1.0)

    else:
        y_train = train_pdf[TARGET].astype(np.float32).values
        y_test  = test_pdf[TARGET].astype(np.float32).values
        if calibrate:
            y_cal = cal_pdf[TARGET].astype(np.float32).values


    # Feature encoding, fit on train only, apply to all live splits
    if calibrate:
        cal_test_pdf = pd.concat([cal_pdf, test_pdf], ignore_index=True)
        n_cal        = len(cal_pdf)
        Xc_tr, Xc_calte, cat_sizes = _encode_categoricals_train_test(
            train_pdf, cal_test_pdf, CAT_INT_COLS=CAT_INT_COLS
        )
        Xn_tr, Xn_calte = _prepare_numeric_train_test(
            train_pdf, cal_test_pdf, NUM_COLS=NUM_COLS
        )
        Xc_cal, Xc_te = Xc_calte[:n_cal], Xc_calte[n_cal:]
        Xn_cal, Xn_te = Xn_calte[:n_cal], Xn_calte[n_cal:]
        del cal_test_pdf
    else:
        Xc_tr, Xc_te, cat_sizes = _encode_categoricals_train_test(
            train_pdf, test_pdf, CAT_INT_COLS=CAT_INT_COLS
        )
        Xn_tr, Xn_te = _prepare_numeric_train_test(
            train_pdf, test_pdf, NUM_COLS=NUM_COLS
        )

    del train_pdf, test_pdf
    if calibrate:
        del cal_pdf
    gc.collect()


    def _make_ds(Xc, Xn, y, m=None, aux=None):
        if mh_enabled and aux_enabled:
            return DS_Agnostic(Xc, Xn, y, mask=m, y_aux=aux)
        elif mh_enabled:
            return DS_Agnostic(Xc, Xn, y, mask=m)
        elif aux_enabled:
            return DS_Agnostic(Xc, Xn, y, y_aux=aux)
        return DS_Agnostic(Xc, Xn, y)

    bs          = int(hp["batch_size"])
    m_train_arg = m_train     if mh_enabled  else None
    aux_tr_arg  = y_train_aux if aux_enabled else None
    m_test_arg  = m_test      if mh_enabled  else None
    aux_te_arg  = y_test_aux  if aux_enabled else None

    train_loader = DataLoader(
        _make_ds(Xc_tr, Xn_tr, y_train, m_train_arg, aux_tr_arg),
        batch_size=bs, shuffle=True, generator=g,
    )
    test_loader = DataLoader(
        _make_ds(Xc_te, Xn_te, y_test, m_test_arg, aux_te_arg),
        batch_size=bs, shuffle=False,
    )

    if calibrate:
        m_cal_arg  = m_cal     if mh_enabled  else None
        aux_ca_arg = y_cal_aux if aux_enabled else None
        cal_loader = DataLoader(
            _make_ds(Xc_cal, Xn_cal, y_cal, m_cal_arg, aux_ca_arg),
            batch_size=bs, shuffle=False,
        )


    model = DeepFM_Agnostic(
        cat_sizes=cat_sizes,
        num_dim=Xn_tr.shape[1],
        emb_dim=int(hp["emb_dim"]),
        deep_hidden=tuple(hp["deep_hidden"]),
        dropout=float(hp["dropout_rate"]),
        n_heads=n_buckets if mh_enabled else 1,
        use_aux=aux_enabled,
    ).to(device)

    opt     = torch.optim.AdamW(model.parameters(),
                                lr=float(hp["learning_rate"]),
                                weight_decay=float(hp["l2_weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss()


    # Training loop
    for _epoch in range(int(hp["epochs"])):
        model.train()
        for batch in train_loader:
            Xc      = batch[0].to(device)
            Xn      = batch[1].to(device)
            y_batch = batch[2].to(device)

            if mh_enabled and aux_enabled:
                m_batch                   = batch[3].to(device)
                aux_y                     = batch[4].to(device)
                primary_logits, aux_logit = model(Xc, Xn)
                loss = (masked_bce_with_logits(primary_logits, y_batch, m_batch)
                        + hp["aux_weight"] * loss_fn(aux_logit, aux_y))

            elif mh_enabled:
                m_batch = batch[3].to(device)
                logits  = model(Xc, Xn)
                loss    = masked_bce_with_logits(logits, y_batch, m_batch)

            elif aux_enabled:
                aux_y                     = batch[3].to(device)
                primary_logits, aux_logit = model(Xc, Xn)
                loss = (loss_fn(primary_logits, y_batch)
                        + hp["aux_weight"] * loss_fn(aux_logit, aux_y))

            else:
                logits = model(Xc, Xn)
                loss   = loss_fn(logits, y_batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


    # Evaluation
    if calibrate:
        # Platt calibration
        print(">>> Fitting Platt calibration on first-day calibration set")

        logits_cal, y_cal_labels = _collect_logits(
            model, cal_loader, device,
            mh_enabled=mh_enabled,
            aux_enabled=aux_enabled,
            mask_col_last=False,    # mask_eval==1 everywhere, keep all rows
        )
        print(f"  Calibration set: {len(y_cal_labels)} rows, "
              f"positive rate: {float((y_cal_labels.sum(axis=-1) > 0).mean()):.4f}")

        logits_eval, y_eval_labels = _collect_logits(
            model, test_loader, device,
            mh_enabled=mh_enabled,
            aux_enabled=aux_enabled,
            mask_col_last=True,
        )

        if mh_enabled:
            p_all, y_all, platt_a, platt_b = _calibrate_multi_head(
                logits_cal, y_cal_labels, logits_eval, y_eval_labels,
                fit_slope=platt_fit_slope,
            )
        else:
            p_all, y_all, platt_a, platt_b = _calibrate_single_head(
                logits_cal, y_cal_labels, logits_eval, y_eval_labels,
                fit_slope=platt_fit_slope,
            )

        print(f"  Platt a={platt_a:.4f}, b={platt_b:.4f}")
        print(f"  Test set: {len(y_all)} rows, "
              f"positive rate: {float(y_all.mean()):.4f}, "
              f"mean predicted P: {float(p_all.mean()):.4f}")

    else:
        # No calibration
        logits_eval, y_eval_labels = _collect_logits(
            model, test_loader, device,
            mh_enabled=mh_enabled,
            aux_enabled=aux_enabled,
            mask_col_last=True,
        )

        if mh_enabled:
            # y_eval_labels is [N, K]; collapse to scalar binary label
            y_all = (y_eval_labels.sum(axis=1) > 0).astype(np.float64)
            p_all = hazard_agg_np(logits_eval)
        else:
            y_all = y_eval_labels
            p_all = sigmoid(logits_eval)


    y_all = y_all.astype(np.int32)
    p_all = p_all.astype(np.float64)


    if outer:
        plot_cvr_boxplot(
            y_all, p_all,
            title=(f"CVR predictions "
                   f"({'calibrated' if calibrate else 'uncalibrated'})"),
        )
        print(f"  Test set: {len(y_all)} rows, "
              f"positive rate: {float(y_all.mean()):.4f}, "
              f"mean predicted P: {float(p_all.mean()):.4f}")

        if aux_enabled:
            aux_probs_list, aux_ys_list = [], []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    Xc = batch[0].to(device)
                    Xn = batch[1].to(device)
                    aux_y_batch = batch[4 if mh_enabled else 3].to(device)
                    _, aux_logit = model(Xc, Xn)
                    aux_probs_list.append(torch.sigmoid(aux_logit).cpu().numpy())
                    aux_ys_list.append(aux_y_batch.cpu().numpy())
            plot_aux_binary_boxplot(
                np.concatenate(aux_ys_list).astype(np.int32),
                np.concatenate(aux_probs_list).astype(np.float64),
            )

    gc.collect()
    return evaluation(y_all, p_all)
