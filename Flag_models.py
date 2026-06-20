import gc
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from Evaluation import evaluation


PLATT_FIT_SLOPE: bool = False

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


def _as_ts(x):
    if isinstance(x, Column):
        return F.to_timestamp(x)
    return F.to_timestamp(F.lit(str(x)))


def _safe_log1p_nonneg(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))



# Platt calibration
# (only called when calibrate=True)
def fit_platt(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    fit_slope: bool = False,
    max_iter:  int  = 100,
) -> tuple[float, float]:

    logits = np.asarray(logits, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    if logits.size == 0:
        return 1.0, 0.0
    y_sum = float(y_true.sum())
    if y_sum == 0.0 or y_sum == float(y_true.size):
        print("  [Platt] WARNING: no positives or no negatives -- skipping.")
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
    print(f"  [Platt] a={a_val:.4f}  b={b_val:.4f}  "
          f"({'slope+intercept' if fit_slope else 'intercept only'})")
    return a_val, b_val


def _apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return (a * np.asarray(logits, dtype=np.float64) + b).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float64)


def _hazard_agg_np(logits: np.ndarray) -> np.ndarray:
    h = _sigmoid(logits)
    return 1.0 - np.prod(1.0 - h, axis=1)


def _calibrate_single_head(
    logits_cal:  np.ndarray,   # [N_cal]
    y_cal:       np.ndarray,   # [N_cal]  binary
    logits_eval: np.ndarray,   # [N_eval]
    y_eval:      np.ndarray,   # [N_eval] binary
    *,
    fit_slope: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (p_eval, y_eval, a, b)."""
    a, b   = fit_platt(logits_cal, y_cal, fit_slope=fit_slope)
    p_eval = _sigmoid(_apply_platt(logits_eval, a, b))
    return p_eval, y_eval.astype(np.float64), a, b


def _calibrate_multi_head(
    logits_cal:  np.ndarray,   # [N_cal,  K]
    y_cal:       np.ndarray,   # [N_cal,  K]  bucket_vec
    logits_eval: np.ndarray,   # [N_eval, K]
    y_eval:      np.ndarray,   # [N_eval, K]
    *,
    fit_slope: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:

    # 1. Aggregate per-head logits -> session probability -> session logit
    p_cal = _hazard_agg_np(logits_cal)                          # [N_cal]
    p_cal = np.clip(p_cal, 1e-12, 1 - 1e-12)
    agg_logit_cal = np.log(p_cal) - np.log1p(-p_cal)

    # 2. Session-level binary label — this is what hazard_agg is trying to predict
    y_cal_bin = (y_cal.sum(axis=1) > 0).astype(np.float64)        # [N_cal]

    # 3. Fit Platt on (aggregated logit, session binary label)
    a, b = fit_platt(agg_logit_cal, y_cal_bin, fit_slope=fit_slope)

    # 4. Apply: aggregate eval logits, convert to logit space, shift, sigmoid
    p_eval_raw    = _hazard_agg_np(logits_eval) 
    p_eval_raw    = np.clip(p_eval_raw, 1e-12, 1 - 1e-12)  # [N_eval]
    agg_logit_eval = np.log(p_eval_raw) - np.log1p(-p_eval_raw)  # [N_eval]
    p_eval        = _sigmoid(a * agg_logit_eval + b)                # [N_eval]

    y_eval_bin = (y_eval.sum(axis=1) > 0).astype(np.float64)
    return p_eval, y_eval_bin, a, b


# Feature encoding
def fit_catint_sizes_and_maps(train_pdf, CAT_INT_COLS, unknown_value=-1):
    maps, sizes = {}, {}
    for c in CAT_INT_COLS or []:
        if c not in train_pdf.columns:
            continue
        s = pd.to_numeric(train_pdf[c], errors="coerce").fillna(unknown_value).astype("int64")
        uniques = pd.Index(s.unique())
        if unknown_value not in uniques:
            uniques = uniques.insert(len(uniques), unknown_value)
        maps[c]  = {v: i for i, v in enumerate(uniques)}
        sizes[c] = max(1, len(uniques))
    return sizes, maps


def enc_catint(pdf, CAT_INT_COLS, cat_maps, unknown_value=-1):
    X = []
    for c in CAT_INT_COLS or []:
        if c not in pdf.columns:
            continue
        s   = pd.to_numeric(pdf[c], errors="coerce").fillna(unknown_value).astype("int64")
        ids = s.map(cat_maps[c])
        if ids.isna().any():
            bad = pd.Index(s[ids.isna()].unique()).tolist()[:10]
            raise ValueError(f"Unseen labels in '{c}': {bad}")
        X.append(ids.astype(np.int64).values.reshape(-1, 1))
    return np.concatenate(X, axis=1) if X else np.zeros((len(pdf), 0), np.int64)


def _encode_categoricals_train_test(train_pdf, *extra_pdfs, CAT_INT_COLS):
    cat_sizes_dict, cat_maps = fit_catint_sizes_and_maps(train_pdf, CAT_INT_COLS)
    used_cols = [c for c in (CAT_INT_COLS or []) if c in train_pdf.columns]
    cat_sizes = [int(cat_sizes_dict[c]) for c in used_cols]
    encode    = lambda pdf: enc_catint(pdf, CAT_INT_COLS, cat_maps)
    return (encode(train_pdf), *[encode(p) for p in extra_pdfs], cat_sizes)


def _prepare_numeric(train_pdf, *extra_pdfs, NUM_COLS):
    NUM_COLS = [c for c in (NUM_COLS or []) if c in train_pdf.columns]

    stats = {}
    for c in NUM_COLS:
        tr    = pd.to_numeric(train_pdf[c], errors="coerce").astype(float)
        trv_t = _safe_log1p_nonneg(tr.values.astype(np.float32))
        obs   = ~np.isnan(trv_t)
        mu    = float(trv_t[obs].mean()) if obs.any() else 0.0
        sd    = float(trv_t[obs].std(ddof=0)) if obs.any() else 1.0
        if sd == 0.0:
            sd = 1.0
        stats[c] = (mu, sd)

    def _encode_split(pdf):
        Xtr = []
        for c in NUM_COLS:
            v      = pd.to_numeric(pdf[c], errors="coerce").astype(float)
            miss   = v.isna().astype(np.float32).values.reshape(-1, 1)
            vt     = _safe_log1p_nonneg(v.values.astype(np.float32))
            mu, sd = stats[c]
            vz     = np.nan_to_num((vt - mu) / sd, nan=0.0).astype(np.float32).reshape(-1, 1)
            Xtr.append(vz)
            Xtr.append(miss)
        return (np.concatenate(Xtr, axis=1).astype(np.float32) if Xtr
                else np.zeros((len(pdf), 0), dtype=np.float32))

    return tuple(_encode_split(p) for p in (train_pdf, *extra_pdfs))



class DS_Agnostic(Dataset):
    def __init__(self, Xc, Xn, y, mask=None, y_aux=None):
        self.Xc    = torch.tensor(Xc,  dtype=torch.long)
        self.Xn    = torch.tensor(Xn,  dtype=torch.float32)
        self.y     = torch.tensor(y,   dtype=torch.float32)
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



def _build_mlp(in_dim: int, layer_sizes: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers = []
    for out_dim in layer_sizes:
        layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = out_dim
    return nn.Sequential(*layers)


class Model_Agnostic(nn.Module):
    def __init__(
        self,
        cat_sizes:    list[int],
        num_dim:      int,
        *,
        emb_dim:      int             = 4,
        deep_hidden: tuple[int, ...] = (256, 128, 128),
        dropout:      float           = 0.1,
        use_aux:      bool            = False,
        n_heads:      int             = 1,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.use_aux = bool(use_aux)
        self.emb_dim = int(emb_dim)

        self.embs     = nn.ModuleList([nn.Embedding(int(s), self.emb_dim) for s in cat_sizes])
        emb_out       = self.emb_dim * len(cat_sizes)
        in_dim        = emb_out + int(num_dim)

        self.trunk    = _build_mlp(in_dim, deep_hidden, dropout)
        trunk_out_dim = deep_hidden[-1]

        self.head     = nn.Linear(trunk_out_dim, self.n_heads)
        self.aux_head = nn.Linear(trunk_out_dim, 1) if self.use_aux else None

    def forward(self, Xc: torch.Tensor, Xn: torch.Tensor):
        if Xc.shape[1] > 0:
            E = torch.cat([emb(Xc[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        else:
            E = torch.zeros((Xn.shape[0], 0), device=Xn.device, dtype=Xn.dtype)

        h      = self.trunk(torch.cat([E, Xn], dim=1))
        logits = self.head(h)
        if self.n_heads == 1:
            logits = logits.squeeze(1)

        if self.use_aux:
            return logits, self.aux_head(h).squeeze(1)
        return logits


def hazard_agg(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.prod(1.0 - torch.sigmoid(logits), dim=1)


def masked_bce_with_logits(logits, y, m):
    bce = nn.BCEWithLogitsLoss(reduction="none")
    per = bce(logits, y) * m
    return per.sum() / m.sum().clamp_min(1.0)


# Collect logits
def _collect_logits(
    model,
    loader,
    device: str,
    *,
    mh_enabled:  bool,
    aux_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()
    logits_list, ys_list = [], []

    with torch.no_grad():
        for batch in loader:
            Xc = batch[0].to(device)
            Xn = batch[1].to(device)
            y  = batch[2].to(device)   # [B, K] or [B]

            if mh_enabled and aux_enabled:
                raw_logits, _ = model(Xc, Xn)
            elif mh_enabled:
                raw_logits    = model(Xc, Xn)
            elif aux_enabled:
                raw_logits, _ = model(Xc, Xn)
            else:
                raw_logits    = model(Xc, Xn)

            logits_list.append(raw_logits.cpu().numpy())
            ys_list.append(y.cpu().numpy())

    return (np.concatenate(logits_list).astype(np.float64),
            np.concatenate(ys_list).astype(np.float64))



def flag_model(
    *,
    df,
    train_end,
    test_end             = None,
    args                 = None,
    NUM_COLS,
    CAT_INT_COLS,
    TARGET               = "conversion",
    START_TS             = "timestamp_dt",
    HEAD_COL             = "bucket_vec",
    MASK_TRAIN_COL       = "mask_train",
    MASK_EVAL_COL        = "mask_eval",
    seed: int            = 0,
    hparams: dict | None = None,
    spark                = None,
    outer: bool          = False,
    calibrate: bool       = True,
    platt_fit_slope: bool = PLATT_FIT_SLOPE,
    **kwargs,
):

    # Settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    g = torch.Generator()
    g.manual_seed(int(seed))

    hp = dict(
        batch_size      = 2048,
        learning_rate   = 1e-3,
        l2_weight_decay = 1e-5,
        deep_hidden    = (256, 128, 128),
        dropout_rate    = 0.1,
        epochs          = 15,
        aux_weight      = 0.1,
        emb_dim         = 4,
    )
    if hparams:
        hp.update(hparams)

    # Ensure deep_hidden is always a tuple
    if isinstance(hp["deep_hidden"], str):
        hp["deep_hidden"] = tuple(
            int(x.strip()) for x in hp["deep_hidden"].strip("()").split(",") if x.strip()
        )

    # Mode flags
    n_buckets   = args.n_buckets
    AUX_TARGET  = getattr(args, "aux_target", None)
    mh_enabled  = n_buckets > 1
    aux_enabled = AUX_TARGET is not None


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
        # test starts at train_end; no calibration set
        test_sdf = (df.where(F.col(START_TS) >= ASOF_TRAIN)
                      .where(F.col(START_TS) <  ASOF_TEST))

    # Drop feature-leaking columns
    drop_cols = ("bucket", "bucket_vec", "mask_train", "mask_eval", "aux_mask")
    if mh_enabled:
        keep      = {"bucket_vec", "mask_train", "mask_eval"}
        drop_cols = tuple(c for c in drop_cols if c not in keep)

    train_sdf = train_sdf.drop(*[c for c in drop_cols if c in train_sdf.columns])
    test_sdf  = test_sdf.drop(*[c  for c in drop_cols if c in test_sdf.columns])
    if calibrate:
        cal_sdf = cal_sdf.drop(*[c for c in drop_cols if c in cal_sdf.columns])

    # Convert to Pandas
    train_pdf = train_sdf.toPandas()
    test_pdf  = test_sdf.toPandas()
    if calibrate:
        cal_pdf = cal_sdf.toPandas()
        print(f"Sizes -- train: {len(train_pdf)}, cal: {len(cal_pdf)}, test: {len(test_pdf)}")
    else:
        print(f"Sizes -- train: {len(train_pdf)}, test: {len(test_pdf)}")


    # Labels
    def stack_array_col(pdf, col):
        if col not in pdf.columns:
            raise ValueError(f"Missing column '{col}'.")
        return np.stack(pdf[col].values).astype(np.float32)

    if mh_enabled and aux_enabled:
        y_train     = stack_array_col(train_pdf, HEAD_COL)
        m_train     = stack_array_col(train_pdf, MASK_TRAIN_COL)
        y_test      = stack_array_col(test_pdf,  HEAD_COL)
        m_test      = stack_array_col(test_pdf,  MASK_EVAL_COL)
        y_train_aux = train_pdf[AUX_TARGET].astype(np.float32).values
        y_test_aux  = test_pdf[AUX_TARGET].astype(np.float32).values
        if calibrate:
            y_cal     = stack_array_col(cal_pdf, HEAD_COL)
            m_cal     = stack_array_col(cal_pdf, MASK_EVAL_COL)
            y_cal_aux = cal_pdf[AUX_TARGET].astype(np.float32).values

    elif mh_enabled:
        y_train = stack_array_col(train_pdf, HEAD_COL)
        m_train = stack_array_col(train_pdf, MASK_TRAIN_COL)
        y_test  = stack_array_col(test_pdf,  HEAD_COL)
        m_test  = stack_array_col(test_pdf,  MASK_EVAL_COL)
        if calibrate:
            y_cal = stack_array_col(cal_pdf, HEAD_COL)
            m_cal = stack_array_col(cal_pdf, MASK_EVAL_COL)

    elif aux_enabled:
        y_train     = train_pdf[TARGET].astype(np.float32).values
        y_test      = test_pdf[TARGET].astype(np.float32).values
        y_train_aux = train_pdf[AUX_TARGET].astype(np.float32).values
        y_test_aux  = test_pdf[AUX_TARGET].astype(np.float32).values
        if calibrate:
            y_cal     = cal_pdf[TARGET].astype(np.float32).values
            y_cal_aux = cal_pdf[AUX_TARGET].astype(np.float32).values

    else:
        y_train = train_pdf[TARGET].astype(np.float32).values
        y_test  = test_pdf[TARGET].astype(np.float32).values
        if calibrate:
            y_cal = cal_pdf[TARGET].astype(np.float32).values

    # Feature encoding, fit on train, apply to all live splits
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
        _make_ds(Xc_train, Xn_train, y_train, m_train_arg, aux_tr_arg),
        batch_size=bs, shuffle=True, generator=g,
    )
    test_loader = DataLoader(
        _make_ds(Xc_test, Xn_test, y_test, m_test_arg, aux_te_arg),
        batch_size=bs, shuffle=False,
    )

    if calibrate:
        m_cal_arg  = m_cal     if mh_enabled  else None
        aux_ca_arg = y_cal_aux if aux_enabled else None
        cal_loader = DataLoader(
            _make_ds(Xc_cal, Xn_cal, y_cal, m_cal_arg, aux_ca_arg),
            batch_size=bs, shuffle=False,
        )


    model = Model_Agnostic(
        cat_sizes    = cat_sizes,
        num_dim      = Xn_train.shape[1],
        emb_dim      = int(hp["emb_dim"]),
        deep_hidden = tuple(hp["deep_hidden"]),
        dropout      = float(hp["dropout_rate"]),
        n_heads      = n_buckets if mh_enabled else 1,
        use_aux      = aux_enabled,
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
        logits_cal, y_cal_labels = _collect_logits(
            model, cal_loader, device,
            mh_enabled=mh_enabled, aux_enabled=aux_enabled,
        )

        logits_eval, y_eval_labels = _collect_logits(
            model, test_loader, device,
            mh_enabled=mh_enabled, aux_enabled=aux_enabled,
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
        print(f"  Eval set: {len(y_all)} rows, "
              f"positive rate: {float(y_all.mean()):.4f}, "
              f"mean predicted P: {float(p_all.mean()):.4f}")

    else:

        logits_eval, y_eval_labels = _collect_logits(
            model, test_loader, device,
            mh_enabled=mh_enabled, aux_enabled=aux_enabled,
        )

        if mh_enabled:
            # y_eval_labels is [N, K]; collapse to scalar binary label
            y_all = (y_eval_labels.sum(axis=1) > 0).astype(np.float64)
            p_all = _hazard_agg_np(logits_eval)
        else:
            y_all = y_eval_labels
            p_all = _sigmoid(logits_eval)   
            

    y_all = y_all.astype(np.int32)
    p_all = p_all.astype(np.float64)

    if outer:
        plot_cvr_boxplot(
            y_all, p_all,
            title=(f"CVR predictions "
                   f"({'calibrated' if calibrate else 'uncalibrated'})"),
        )
        print(f"  Eval set: {len(y_all)} rows, "
              f"positive rate: {float(y_all.mean()):.4f}, "
              f"mean predicted P: {float(p_all.mean()):.4f}")

    gc.collect()
    return evaluation(y_all, p_all)
