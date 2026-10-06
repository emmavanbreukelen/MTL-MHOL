# Entire Space Multi-Task Model (ESMM)
# Ma, Zhao, Huang, Wang, Hu, Zhu & Gai (2018), "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate"

import gc
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader
from pyspark.sql import functions as F

from Flag_models import (
    _as_ts,
    _build_mlp,
    _encode_categoricals_train_test,
    _prepare_numeric,
    fit_platt,
    _apply_platt,
    _sigmoid,
    plot_cvr_boxplot,
    PLATT_FIT_SLOPE,
)
from Evaluation import evaluation

class DS_ESMM(Dataset):
    def __init__(self, Xc, Xn, y_conv, y_click):
        self.Xc      = torch.tensor(Xc,      dtype=torch.long)
        self.Xn      = torch.tensor(Xn,      dtype=torch.float32)
        self.y_conv  = torch.tensor(y_conv,  dtype=torch.float32)
        self.y_click = torch.tensor(y_click, dtype=torch.float32)

    def __len__(self):
        return self.y_conv.shape[0]

    def __getitem__(self, i):
        return self.Xc[i], self.Xn[i], self.y_conv[i], self.y_click[i]
      

class ESMMNet(nn.Module):
    def __init__(
        self,
        cat_sizes: list,
        num_dim: int,
        *,
        emb_dim: int = 4,
        deep_hidden: tuple = (256, 128, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)

        # Single shared embedding table
        self.embs = nn.ModuleList([nn.Embedding(int(s), self.emb_dim) for s in cat_sizes])
        emb_out   = self.emb_dim * len(cat_sizes)
        in_dim    = emb_out + int(num_dim)

        self.ctr_trunk = _build_mlp(in_dim, deep_hidden, dropout)
        self.cvr_trunk = _build_mlp(in_dim, deep_hidden, dropout)
        self.ctr_head  = nn.Linear(deep_hidden[-1], 1)
        self.cvr_head  = nn.Linear(deep_hidden[-1], 1)

    def forward(self, Xc: torch.Tensor, Xn: torch.Tensor):
        if Xc.shape[1] > 0:
            E = torch.cat([emb(Xc[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        else:
            E = torch.zeros((Xn.shape[0], 0), device=Xn.device, dtype=Xn.dtype)

        x = torch.cat([E, Xn], dim=1)

        ctr_logit = self.ctr_head(self.ctr_trunk(x)).squeeze(-1)
        cvr_logit = self.cvr_head(self.cvr_trunk(x)).squeeze(-1)
        return ctr_logit, cvr_logit


# CTCVR - logit-space helpers
def _ctcvr_prob(ctr_logit: np.ndarray, cvr_logit: np.ndarray) -> np.ndarray:
    p_ctr = _sigmoid(ctr_logit)
    p_cvr = _sigmoid(cvr_logit)
    return (p_ctr * p_cvr).astype(np.float64)


def _prob_to_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return (np.log(p) - np.log1p(-p)).astype(np.float64)


def _collect_esmm_logits(model, loader, device):
    """Returns (ctr_logits, cvr_logits, y_conv, y_click) as np arrays."""
    model.eval()
    ctr_l, cvr_l, yc_l, yk_l = [], [], [], []
    with torch.no_grad():
        for Xc, Xn, y_conv, y_click in loader:
            Xc, Xn = Xc.to(device), Xn.to(device)
            ctr_logit, cvr_logit = model(Xc, Xn)
            ctr_l.append(ctr_logit.cpu().numpy())
            cvr_l.append(cvr_logit.cpu().numpy())
            yc_l.append(y_conv.numpy())
            yk_l.append(y_click.numpy())
    return (
        np.concatenate(ctr_l).astype(np.float64),
        np.concatenate(cvr_l).astype(np.float64),
        np.concatenate(yc_l).astype(np.float64),
        np.concatenate(yk_l).astype(np.float64),
    )


def esmm(
    *,
    df,
    train_end,
    test_end=None,
    args=None,
    NUM_COLS,
    CAT_INT_COLS,
    TARGET="conversion",
    START_TS="timestamp_dt",
    MASK_TRAIN_COL="mask_train",
    seed: int = 0,
    hparams: dict = None,
    spark=None,
    outer: bool = False,
    calibrate: bool = False,
    platt_fit_slope: bool = PLATT_FIT_SLOPE,
    max_delay_days=None,   # unused -- ESMM has no delay-aware mechanism
    **kwargs,
):
    if args is None or getattr(args, "n_buckets", 1) != 1:
        raise ValueError(
            "ESMM is a single-head method (Ma et al., 2018) -- it has no "
            "notion of delay buckets. Run with --n_buckets 1."
        )
    AUX_TARGET = getattr(args, "aux_target", None)
    if AUX_TARGET is None:
        raise ValueError(
            "ESMM requires a binary click column to be loaded via the "
            "existing auxiliary-target plumbing. Run with "
            "--aux_target click (or whatever the click column is named "
            "in the current table)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    g = torch.Generator()
    g.manual_seed(int(seed))

    hp = dict(
        batch_size      = 2048,
        learning_rate   = 1e-3,
        l2_weight_decay = 1e-5,
        deep_hidden     = (256, 128, 128),
        dropout_rate    = 0.1,
        epochs          = 15,
        emb_dim         = 4,
        # aux_weight may arrive here via the shared tuner -- unused,
        # see module docstring.
    )
    if hparams:
        hp.update(hparams)
    if isinstance(hp["deep_hidden"], str):
        hp["deep_hidden"] = tuple(
            int(x.strip()) for x in hp["deep_hidden"].strip("()").split(",") if x.strip()
        )

    ASOF_TRAIN = _as_ts(train_end)
    ASOF_TEST  = _as_ts(test_end)

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
        test_sdf = (df.where(F.col(START_TS) >= ASOF_TRAIN)
                      .where(F.col(START_TS) <  ASOF_TEST))

    drop_cols = [c for c in ("bucket", "bucket_vec", "mask_train", "mask_eval", "aux_mask")
                 if c in df.columns]
    train_sdf = train_sdf.drop(*drop_cols)
    test_sdf  = test_sdf.drop(*drop_cols)
    if calibrate:
        cal_sdf = cal_sdf.drop(*drop_cols)

    train_pdf = train_sdf.toPandas().reset_index(drop=True)
    test_pdf  = test_sdf.toPandas().reset_index(drop=True)
    if calibrate:
        cal_pdf = cal_sdf.toPandas().reset_index(drop=True)
        print(f"[ESMM] Sizes -- train: {len(train_pdf)}, cal: {len(cal_pdf)}, test: {len(test_pdf)}")
    else:
        print(f"[ESMM] Sizes -- train: {len(train_pdf)}, test: {len(test_pdf)}")

    conv_mask = train_pdf[TARGET].astype(int).values == 1
    if conv_mask.any():
        organic_share = float((train_pdf.loc[conv_mask, AUX_TARGET].astype(int).values == 0).mean())
    else:
        organic_share = 0.0
    print(f"[ESMM] Share of TRAIN conversions with click=0 (organic): {organic_share:.4%}")
    if organic_share > 0.0:
        print(
            "[ESMM] WARNING: nonzero organic-conversion share detected in this "
            "fold. pCTCVR is no longer guaranteed to equal P(conversion|"
            "impression) here -- treat this fold's ESMM-vs-TARGET comparison "
            "with caution (see module docstring)."
        )
      
    # Labels
    y_train_conv  = train_pdf[TARGET].astype(np.float32).values
    y_train_click = train_pdf[AUX_TARGET].astype(np.float32).values
    y_test_conv   = test_pdf[TARGET].astype(np.float32).values
    y_test_click  = test_pdf[AUX_TARGET].astype(np.float32).values
    if calibrate:
        y_cal_conv  = cal_pdf[TARGET].astype(np.float32).values
        y_cal_click = cal_pdf[AUX_TARGET].astype(np.float32).values

    # Feature encoding -- identical helpers used by every other neural model in this pipeline 
    if calibrate:
        *Xc_splits, cat_sizes = _encode_categoricals_train_test(
            train_pdf, cal_pdf, test_pdf, CAT_INT_COLS=CAT_INT_COLS
        )
        Xc_train, Xc_cal, Xc_test = Xc_splits
        Xn_train, Xn_cal, Xn_test = _prepare_numeric(
            train_pdf, cal_pdf, test_pdf, NUM_COLS=NUM_COLS
        )
    else:
        *Xc_splits, cat_sizes = _encode_categoricals_train_test(
            train_pdf, test_pdf, CAT_INT_COLS=CAT_INT_COLS
        )
        Xc_train, Xc_test = Xc_splits
        Xn_train, Xn_test = _prepare_numeric(
            train_pdf, test_pdf, NUM_COLS=NUM_COLS
        )

  
    bs = int(hp["batch_size"])
    train_loader = DataLoader(
        DS_ESMM(Xc_train, Xn_train, y_train_conv, y_train_click),
        batch_size=bs, shuffle=True, generator=g,
    )
    test_loader = DataLoader(
        DS_ESMM(Xc_test, Xn_test, y_test_conv, y_test_click),
        batch_size=bs, shuffle=False,
    )
    if calibrate:
        cal_loader = DataLoader(
            DS_ESMM(Xc_cal, Xn_cal, y_cal_conv, y_cal_click),
            batch_size=bs, shuffle=False,
        )


    model = ESMMNet(
        cat_sizes   = cat_sizes,
        num_dim     = Xn_train.shape[1],
        emb_dim     = int(hp["emb_dim"]),
        deep_hidden = tuple(hp["deep_hidden"]),
        dropout     = float(hp["dropout_rate"]),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp["learning_rate"]),
        weight_decay=float(hp["l2_weight_decay"]),
    )
    bce_ctr = nn.BCEWithLogitsLoss()   # stable: applied directly to ctr_logit
    eps = 1e-7


    # Training loop
    for _epoch in range(int(hp["epochs"])):
        model.train()
        for Xc, Xn, y_conv, y_click in train_loader:
            Xc, Xn = Xc.to(device), Xn.to(device)
            y_conv, y_click = y_conv.to(device), y_click.to(device)

            ctr_logit, cvr_logit = model(Xc, Xn)
            p_ctr   = torch.sigmoid(ctr_logit)
            p_cvr   = torch.sigmoid(cvr_logit)
            p_ctcvr = torch.clamp(p_ctr * p_cvr, eps, 1.0 - eps)

            loss_ctr   = bce_ctr(ctr_logit, y_click)
            loss_ctcvr = Fnn.binary_cross_entropy(p_ctcvr, y_conv)
            loss       = loss_ctr + loss_ctcvr   # UNWEIGHTED sum, Eq. (3)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    # Evaluation
    if calibrate:
        ctr_l_cal, cvr_l_cal, y_cal_labels, _ = _collect_esmm_logits(model, cal_loader, device)
        ctr_l_ev,  cvr_l_ev,  y_eval_labels, _ = _collect_esmm_logits(model, test_loader, device)

        p_ctcvr_cal = _ctcvr_prob(ctr_l_cal, cvr_l_cal)
        p_ctcvr_ev  = _ctcvr_prob(ctr_l_ev,  cvr_l_ev)

        logit_cal = _prob_to_logit(p_ctcvr_cal)
        logit_ev  = _prob_to_logit(p_ctcvr_ev)

        platt_a, platt_b = fit_platt(logit_cal, y_cal_labels, fit_slope=platt_fit_slope)
        p_all = _sigmoid(_apply_platt(logit_ev, platt_a, platt_b))
        y_all = y_eval_labels
        print(f"  [ESMM] Platt a={platt_a:.4f}, b={platt_b:.4f}")
    else:
        ctr_l_ev, cvr_l_ev, y_eval_labels, _ = _collect_esmm_logits(model, test_loader, device)
        p_all = _ctcvr_prob(ctr_l_ev, cvr_l_ev)
        y_all = y_eval_labels

    y_all = y_all.astype(np.int32)
    p_all = p_all.astype(np.float64)

    if outer:
        plot_cvr_boxplot(
            y_all, p_all,
            title=f"ESMM (pCTCVR) predictions ({'calibrated' if calibrate else 'uncalibrated'})",
        )
        print(f"  [ESMM] Eval set: {len(y_all)} rows, "
              f"positive rate: {float(y_all.mean()):.4f}, "
              f"mean predicted P: {float(p_all.mean()):.4f}")

    gc.collect()
    return evaluation(y_all, p_all)
