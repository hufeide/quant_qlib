# -*- coding: utf-8 -*-
"""
Qlib 多因子综合优化框架

目标：
    综合优化
        1. RankIC
        2. HAC-ICIR
        3. Turnover
        4. Transaction Cost
        5. Net Return
        6. Net IR
        7. Max Drawdown

核心思想：
    Factor
        ↓
    Cross-sectional standardization
        ↓
    Factor IC / HAC-ICIR
        ↓
    Alpha Ensemble
        ↓
    Portfolio
        ↓
    Turnover
        ↓
    Transaction Cost
        ↓
    Net Return
        ↓
    Net IR / MaxDD
        ↓
    Walk Forward OOS

不依赖 scipy / optuna。
优化器使用随机搜索 + 坐标扰动。

注意：
    因子必须是 t 时刻可以获得的信息。
    future_return[t] 必须对应 t 建仓之后的收益。
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional, List, Tuple

from statsmodels.stats.sandwich_covariance import cov_hac


# ============================================================
# 0. CONFIG
# ============================================================

@dataclass
class Config:

    # -----------------------------
    # 数据
    # -----------------------------
    factor_file: str = "factor_data.parquet"
    return_file: str = "future_return.parquet"

    date_col: str = "date"
    instrument_col: str = "instrument"
    return_col: str = "return"

    # -----------------------------
    # 交易成本
    # -----------------------------
    cost_rate: float = 0.001       # 单边 10bp
    slippage_rate: float = 0.0001 # 单边 1bp

    # 总成本
    # 这里假设 turnover 是“单边换手”
    # 如果你的 turnover 是双边换手，需要相应调整
    @property
    def total_cost(self):
        return self.cost_rate + self.slippage_rate

    # -----------------------------
    # 投资组合
    # -----------------------------
    n_quantile: int = 5

    long_only: bool = True

    # Top quantile
    top_q: int = 5

    # 每个股票最大权重
    max_stock_weight: float = 0.05

    # -----------------------------
    # 优化
    # -----------------------------
    n_iter: int = 1000

    random_seed: int = 20260913

    # 单因子最大权重
    max_factor_weight: float = 0.15

    # L1 正则
    l1_penalty: float = 0.01

    # 换手惩罚
    turnover_penalty: float = 0.10

    # 回撤惩罚
    drawdown_penalty: float = 0.10

    # ICIR 权重
    icir_weight: float = 0.25

    # Net IR 权重
    net_ir_weight: float = 0.50

    # Net Return 权重
    net_return_weight: float = 0.15

    # Turnover penalty
    turnover_weight: float = 0.05

    # MaxDD penalty
    maxdd_weight: float = 0.05

    # -----------------------------
    # HAC
    # -----------------------------
    hac_lags: int = 5

    # -----------------------------
    # WFA
    # -----------------------------
    train_days: int = 756
    test_days: int = 252

    n_wf: int = 4


CFG = Config()


# ============================================================
# 1. LOG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger(__name__)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data(cfg: Config):

    log.info("=" * 70)
    log.info("Loading factor / return data")
    log.info("=" * 70)

    fac = pd.read_parquet(cfg.factor_file)
    ret = pd.read_parquet(cfg.return_file)

    fac[cfg.date_col] = pd.to_datetime(fac[cfg.date_col])
    ret[cfg.date_col] = pd.to_datetime(ret[cfg.date_col])

    fac = fac.sort_values(
        [cfg.date_col, cfg.instrument_col]
    )

    ret = ret.sort_values(
        [cfg.date_col, cfg.instrument_col]
    )

    log.info(
        "Factor shape = %s",
        fac.shape
    )

    log.info(
        "Return shape = %s",
        ret.shape
    )

    return fac, ret


# ============================================================
# 3. IDENTIFY FACTORS
# ============================================================

def get_factor_columns(fac: pd.DataFrame, cfg: Config):

    exclude = {
        cfg.date_col,
        cfg.instrument_col,
        cfg.return_col,
    }

    factors = [
        c for c in fac.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(fac[c])
    ]

    if len(factors) == 0:
        raise ValueError("没有找到因子列")

    log.info("Number of factors = %d", len(factors))

    return factors


# ============================================================
# 4. CROSS SECTIONAL RANK
# ============================================================

def cs_rank_zscore(
    df: pd.DataFrame,
    factors: List[str],
    cfg: Config
):

    """
    每个交易日进行截面 Rank → [-1,1] → zscore

    Rank 化可以显著降低：
        extreme value
        不同 Alpha 尺度差异
        非正态分布
    """

    result = df.copy()

    grouped = result.groupby(
        cfg.date_col,
        sort=False
    )

    for f in factors:

        rank = grouped[f].rank(
            pct=True
        )

        # [-1, 1]
        result[f] = 2.0 * rank - 1.0

    return result


# ============================================================
# 5. CROSS SECTIONAL NEUTRALIZATION
# ============================================================

def neutralize_size_industry(
    df: pd.DataFrame,
    factor_cols: List[str],
    cfg: Config,
    size_col: Optional[str] = None,
    industry_col: Optional[str] = None
):
    """
    可选：
        Size + Industry 中性化

    如果没有提供 size / industry，则原样返回。
    """

    if size_col is None and industry_col is None:
        return df

    result = df.copy()

    for date, g in result.groupby(cfg.date_col):

        idx = g.index

        X_parts = []

        if size_col is not None and size_col in g.columns:
            X_parts.append(
                g[[size_col]].astype(float)
            )

        if industry_col is not None and industry_col in g.columns:

            dummy = pd.get_dummies(
                g[industry_col],
                drop_first=True,
                dtype=float
            )

            X_parts.append(dummy)

        if not X_parts:
            continue

        X = pd.concat(
            X_parts,
            axis=1
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        valid_x = X.notna().all(axis=1)

        if valid_x.sum() < 30:
            continue

        Xv = X.loc[valid_x].values

        Xv = np.column_stack([
            np.ones(len(Xv)),
            Xv
        ])

        for f in factor_cols:

            y = g.loc[
                valid_x,
                f
            ].astype(float).values

            valid = np.isfinite(y)

            if valid.sum() < 30:
                continue

            beta = np.linalg.lstsq(
                Xv[valid],
                y[valid],
                rcond=None
            )[0]

            residual = np.full(
                len(g),
                np.nan
            )

            residual[
                np.where(valid_x)[0][valid]
            ] = (
                y[valid]
                - Xv[valid] @ beta
            )

            result.loc[idx, f] = residual

    return result


# ============================================================
# 6. DAILY RANK IC
# ============================================================

def daily_rank_ic(
    factor_df: pd.DataFrame,
    return_df: pd.DataFrame,
    factor: str,
    cfg: Config
):

    x = factor_df[
        [
            cfg.date_col,
            cfg.instrument_col,
            factor
        ]
    ]

    y = return_df[
        [
            cfg.date_col,
            cfg.instrument_col,
            cfg.return_col
        ]
    ]

    z = x.merge(
        y,
        on=[
            cfg.date_col,
            cfg.instrument_col
        ],
        how="inner"
    )

    def calc(g):

        a = g[factor]
        b = g[cfg.return_col]

        valid = (
            a.notna()
            & b.notna()
            & np.isfinite(a)
            & np.isfinite(b)
        )

        if valid.sum() < 10:
            return np.nan

        return a[valid].rank().corr(
            b[valid].rank()
        )

    ic = (
        z.groupby(
            cfg.date_col,
            sort=True
        )
        .apply(calc)
        .dropna()
    )

    return ic


# ============================================================
# 7. HAC ICIR
# ============================================================

def hac_icir(
    ic: pd.Series,
    hac_lags: int = 5
):

    """
    正确计算 HAC-ICIR。

    不是：
        mean(IC) / std(IC)

    而是：

        mean(IC)
        ----------------------
        sqrt(HAC Var(mean(IC)))

    然后乘 sqrt(T)
    得到 annualized-like ICIR。

    注意：
    这里的 IC 是每日截面 RankIC。
    """

    ic = pd.Series(ic).dropna()

    n = len(ic)

    if n < 30:
        return np.nan

    y = ic.values

    X = np.ones(
        (n, 1)
    )

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )[0][0]

    residual = y - beta

    try:

        # OLS residual
        # cov_hac 需要模型结果对象
        class OLSResult:

            pass

        model = OLSResult()

        model.model = type(
            "Model",
            (),
            {
                "exog": X
            }
        )()

        model.resid = residual

        model.normalized_cov_params = np.array(
            [[1.0 / n]]
        )

        model.scale = np.mean(
            residual ** 2
        )

        # statsmodels 的 cov_hac 对自定义对象版本差异较大
        # 因此采用等价 Newey-West 手工实现
        var_mean = newey_west_mean_variance(
            y,
            hac_lags
        )

    except Exception:

        var_mean = (
            np.var(
                y,
                ddof=1
            ) / n
        )

    if var_mean <= 0:
        return np.nan

    return beta / np.sqrt(var_mean)


# ============================================================
# 8. NEWey-West
# ============================================================

def newey_west_mean_variance(
    x,
    lags
):

    """
    HAC variance of sample mean.

    Var(mean)
      =
    [gamma0
     + 2 Σ w_l gamma_l] / T
    """

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n < 2:
        return np.nan

    x = x - x.mean()

    gamma0 = np.mean(
        x * x
    )

    var = gamma0

    lags = min(
        lags,
        n - 1
    )

    for lag in range(1, lags + 1):

        weight = (
            1.0
            - lag / (lags + 1.0)
        )

        gamma = np.mean(
            x[lag:] * x[:-lag]
        )

        var += (
            2
            * weight
            * gamma
        )

    return var / n


# ============================================================
# 9. FACTOR STATISTICS
# ============================================================

def calculate_factor_statistics(
    fac,
    ret,
    factors,
    cfg
):

    rows = []

    for i, f in enumerate(factors):

        log.info(
            "[%d/%d] Factor = %s",
            i + 1,
            len(factors),
            f
        )

        ic = daily_rank_ic(
            fac,
            ret,
            f,
            cfg
        )

        if len(ic) == 0:
            continue

        mean_ic = ic.mean()

        median_ic = ic.median()

        std_ic = ic.std(
            ddof=1
        )

        icir = hac_icir(
            ic,
            cfg.hac_lags
        )

        ic_win = (
            (ic > 0).mean()
        )

        rows.append({

            "factor": f,

            "mean_rank_ic": mean_ic,

            "median_rank_ic": median_ic,

            "std_rank_ic": std_ic,

            "icir_hac": icir,

            "ic_win_rate": ic_win,

            "n_days": len(ic)

        })

    stats = pd.DataFrame(rows)

    return stats.sort_values(
        "icir_hac",
        ascending=False
    )


# ============================================================
# 10. SIGN NORMALIZATION
# ============================================================

def normalize_factor_sign(
    fac,
    stats,
    cfg
):

    """
    把负 IC 因子反向。

    例如 KSFT2：

        IC = -0.025

    转成：

        -KSFT2

    这样所有因子统一成：
        数值越大 → 预期收益越高
    """

    result = fac.copy()

    sign_map = {}

    for _, row in stats.iterrows():

        f = row["factor"]

        ic = row["mean_rank_ic"]

        if np.isfinite(ic) and ic < 0:

            result[f] *= -1

            sign_map[f] = -1

        else:

            sign_map[f] = 1

    return result, sign_map


# ============================================================
# 11. FACTOR CORRELATION
# ============================================================

def factor_correlation(
    fac,
    factors,
    cfg
):

    """
    每个交易日先截面 Rank，
    然后计算整体因子相关性。
    """

    tmp = fac[
        [cfg.date_col] + factors
    ].copy()

    for f in factors:

        tmp[f] = (
            tmp.groupby(
                cfg.date_col
            )[f]
            .rank(pct=True)
        )

    corr = tmp[
        factors
    ].corr(
        method="spearman"
    )

    return corr


# ============================================================
# 12. FACTOR SELECTION
# ============================================================

def select_factors(
    stats,
    corr,
    min_icir=0.0,
    max_corr=0.85,
    max_factors=50
):

    """
    ICIR + 相关性双重筛选。
    """

    candidates = stats[
        stats["icir_hac"] > min_icir
    ].copy()

    selected = []

    for f in candidates["factor"]:

        if len(selected) == 0:

            selected.append(f)

        else:

            max_abs_corr = max(
                abs(
                    corr.loc[f, selected]
                )
            )

            if max_abs_corr < max_corr:

                selected.append(f)

        if len(selected) >= max_factors:
            break

    return selected


# ============================================================
# 13. FACTOR ENSEMBLE
# ============================================================

def ensemble_score(
    df,
    factors,
    weights,
    cfg
):

    X = df[factors].values

    w = np.asarray(
        weights,
        dtype=float
    )

    return np.nan_to_num(
        X,
        nan=0.0
    ) @ w


# ============================================================
# 14. PORTFOLIO CONSTRUCTION
# ============================================================

def construct_long_only_portfolio(
    df,
    score_col,
    cfg
):

    """
    每天：

        Top quantile → long

    资金等权。

    自动限制单股票权重。
    """

    result = []

    for date, g in df.groupby(
        cfg.date_col,
        sort=True
    ):

        g = g.copy()

        g = g[
            np.isfinite(
                g[score_col]
            )
        ]

        if len(g) < 10:
            continue

        threshold = g[
            score_col
        ].quantile(
            1.0 - 1.0 / cfg.n_quantile
        )

        selected = g[
            g[score_col] >= threshold
        ].copy()

        if len(selected) == 0:
            continue

        w = np.ones(
            len(selected)
        )

        w /= w.sum()

        # -------------------------
        # cap
        # -------------------------

        cap = cfg.max_stock_weight

        for _ in range(20):

            excess = np.maximum(
                w - cap,
                0
            )

            if excess.sum() <= 1e-12:
                break

            w = np.minimum(
                w,
                cap
            )

            remain = (
                1.0 - w.sum()
            )

            free = w < cap - 1e-12

            if free.sum() == 0:
                break

            w[free] += (
                remain
                / free.sum()
            )

        selected["weight"] = w

        selected["date"] = date

        result.append(
            selected[
                [
                    cfg.date_col,
                    cfg.instrument_col,
                    "weight"
                ]
            ]
        )

    if not result:
        return pd.DataFrame(
            columns=[
                cfg.date_col,
                cfg.instrument_col,
                "weight"
            ]
        )

    return pd.concat(
        result,
        ignore_index=True
    )


# ============================================================
# 15. DAILY PORTFOLIO RETURN
# ============================================================

def portfolio_return(
    weights,
    returns,
    cfg
):

    x = weights.merge(
        returns,
        on=[
            cfg.date_col,
            cfg.instrument_col
        ],
        how="left"
    )

    x["return"] = x[
        cfg.return_col
    ].fillna(0.0)

    x["pnl"] = (
        x["weight"]
        * x["return"]
    )

    daily = (
        x.groupby(
            cfg.date_col
        )["pnl"]
        .sum()
    )

    return daily


# ============================================================
# 16. TURNOVER
# ============================================================

def calculate_turnover(
    weights,
    cfg
):

    """
    单边 turnover：

        0.5 * Σ |w_t - w_{t-1}|

    如果你定义的是双边换手：
        Σ |Δw|

    那么 cost_rate 也要对应调整。

    本函数采用“单边换手”。
    """

    if len(weights) == 0:

        return pd.Series(
            dtype=float
        )

    dates = sorted(
        weights[
            cfg.date_col
        ].unique()
    )

    previous = {}

    turnovers = {}

    for date in dates:

        g = weights[
            weights[cfg.date_col] == date
        ]

        current = dict(
            zip(
                g[cfg.instrument_col],
                g["weight"]
            )
        )

        instruments = set(
            previous
        ) | set(
            current
        )

        turnover = 0.5 * sum(
            abs(
                current.get(k, 0.0)
                -
                previous.get(k, 0.0)
            )
            for k in instruments
        )

        turnovers[date] = turnover

        previous = current

    return pd.Series(
        turnovers
    )


# ============================================================
# 17. COST
# ============================================================

def calculate_net_return(
    gross_return,
    turnover,
    cfg
):

    turnover = turnover.reindex(
        gross_return.index
    ).fillna(0)

    cost = (
        turnover
        * cfg.total_cost
    )

    net = (
        gross_return
        - cost
    )

    return net, cost


# ============================================================
# 18. MAX DRAWDOWN
# ============================================================

def max_drawdown(
    returns
):

    equity = (
        1.0 + returns
    ).cumprod()

    peak = equity.cummax()

    dd = (
        equity / peak - 1.0
    )

    return dd.min()


# ============================================================
# 19. PERFORMANCE
# ============================================================

def performance_metrics(
    net_return,
    turnover
):

    r = net_return.dropna()

    if len(r) < 20:

        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "net_ir": np.nan,
            "maxdd": np.nan,
            "avg_turnover": np.nan
        }

    ann_return = (
        r.mean() * 252
    )

    ann_vol = (
        r.std(ddof=1)
        * np.sqrt(252)
    )

    net_ir = (
        ann_return / ann_vol
        if ann_vol > 0
        else np.nan
    )

    dd = max_drawdown(r)

    avg_turnover = (
        turnover
        .reindex(r.index)
        .fillna(0)
        .mean()
    )

    return {

        "ann_return": ann_return,

        "ann_vol": ann_vol,

        "net_ir": net_ir,

        "maxdd": dd,

        "avg_turnover": avg_turnover
    }


# ============================================================
# 20. OBJECTIVE
# ============================================================

def objective(
    metrics,
    icir,
    weights,
    cfg
):

    net_ir = metrics["net_ir"]

    ann_return = metrics[
        "ann_return"
    ]

    maxdd = abs(
        metrics["maxdd"]
    )

    turnover = metrics[
        "avg_turnover"
    ]

    if not np.isfinite(net_ir):
        return -1e10

    if not np.isfinite(icir):
        icir = 0.0

    if not np.isfinite(ann_return):
        ann_return = -1.0

    if not np.isfinite(maxdd):
        maxdd = 1.0

    if not np.isfinite(turnover):
        turnover = 1.0

    # -------------------------
    # 核心目标
    # -------------------------

    score = (

        cfg.net_ir_weight
        * net_ir

        +

        cfg.icir_weight
        * icir

        +

        cfg.net_return_weight
        * ann_return

        -

        cfg.turnover_weight
        * turnover

        -

        cfg.maxdd_weight
        * maxdd

    )

    # -------------------------
    # L1 regularization
    # -------------------------

    score -= (
        cfg.l1_penalty
        * np.sum(
            np.abs(weights)
        )
    )

    return score


# ============================================================
# 21. WEIGHT NORMALIZATION
# ============================================================

def normalize_weights(
    w,
    max_weight
):

    w = np.asarray(
        w,
        dtype=float
    )

    w = np.maximum(
        w,
        0
    )

    if w.sum() <= 1e-12:

        w[:] = 1.0

    # iterative cap
    for _ in range(100):

        w = w / w.sum()

        over = w > max_weight

        if not over.any():
            break

        excess = (
            w[over].sum()
            -
            max_weight * over.sum()
        )

        w[over] = max_weight

        under = ~over

        if under.sum() == 0:
            break

        w[under] += (
            excess
            * w[under]
            / w[under].sum()
        )

    return w / w.sum()


# ============================================================
# 22. FACTOR ICIR OF ENSEMBLE
# ============================================================

def ensemble_daily_ic(
    fac,
    ret,
    factors,
    weights,
    cfg
):

    x = fac[
        [
            cfg.date_col,
            cfg.instrument_col
        ] + factors
    ].copy()

    x["ensemble"] = (
        x[factors]
        .fillna(0)
        .values
        @ np.asarray(weights)
    )

    y = ret[
        [
            cfg.date_col,
            cfg.instrument_col,
            cfg.return_col
        ]
    ]

    z = x.merge(
        y,
        on=[
            cfg.date_col,
            cfg.instrument_col
        ],
        how="inner"
    )

    def calc(g):

        a = g["ensemble"]
        b = g[cfg.return_col]

        valid = (
            a.notna()
            & b.notna()
            & np.isfinite(a)
            & np.isfinite(b)
        )

        if valid.sum() < 10:
            return np.nan

        return (
            a[valid].rank()
            .corr(
                b[valid].rank()
            )
        )

    return (
        z.groupby(
            cfg.date_col
        )
        .apply(calc)
        .dropna()
    )


# ============================================================
# 23. RANDOM WEIGHT OPTIMIZATION
# ============================================================

def optimize_weights(
    fac,
    ret,
    factors,
    cfg
):

    """
    随机搜索：

        w >= 0
        sum(w)=1
        w_i <= max_factor_weight

    每次：

        Factor Ensemble
             ↓
        Portfolio
             ↓
        Turnover
             ↓
        Cost
             ↓
        Net Return
             ↓
        Net IR
             ↓
        HAC ICIR
             ↓
        Objective
    """

    rng = np.random.default_rng(
        cfg.random_seed
    )

    n = len(factors)

    # --------------------------------
    # baseline
    # --------------------------------

    best_score = -np.inf
    best_w = None
    best_result = None

    # --------------------------------
    # baseline weights
    # --------------------------------

    equal_w = np.ones(n) / n

    equal_w = normalize_weights(
        equal_w,
        cfg.max_factor_weight
    )

    candidate_weights = [
        equal_w
    ]

    # --------------------------------
    # random search
    # --------------------------------

    for iteration in range(
        cfg.n_iter
    ):

        if iteration == 0:

            w = equal_w.copy()

        else:

            # Dirichlet
            w = rng.dirichlet(
                np.ones(n)
            )

            w = normalize_weights(
                w,
                cfg.max_factor_weight
            )

        tmp = fac[
            [
                cfg.date_col,
                cfg.instrument_col
            ] + factors
        ].copy()

        tmp["ensemble"] = (
            tmp[factors]
            .fillna(0)
            .values
            @ w
        )

        weights = construct_long_only_portfolio(
            tmp,
            "ensemble",
            cfg
        )

        if len(weights) == 0:
            continue

        gross = portfolio_return(
            weights,
            ret,
            cfg
        )

        turnover = calculate_turnover(
            weights,
            cfg
        )

        net, cost = calculate_net_return(
            gross,
            turnover,
            cfg
        )

        metrics = performance_metrics(
            net,
            turnover
        )

        ic = ensemble_daily_ic(
            fac,
            ret,
            factors,
            w,
            cfg
        )

        icir = hac_icir(
            ic,
            cfg.hac_lags
        )

        score = objective(
            metrics,
            icir,
            w,
            cfg
        )

        if score > best_score:

            best_score = score

            best_w = w.copy()

            best_result = {

                "score": score,

                "weights": w.copy(),

                "metrics": metrics,

                "icir": icir,

                "mean_ic": ic.mean(),

                "ic_win_rate": (
                    (ic > 0).mean()
                ),

                "cost": cost.mean()

            }

            log.info(
                "Iter=%d | score=%.5f | "
                "NetIR=%.4f | ICIR=%.4f | "
                "Return=%.2f%% | "
                "Turnover=%.2f%% | "
                "DD=%.2f%%",
                iteration,
                score,
                metrics["net_ir"],
                icir,
                metrics["ann_return"] * 100,
                metrics["avg_turnover"] * 100,
                metrics["maxdd"] * 100
            )

    return best_w, best_result


# ============================================================
# 24. WFA SPLIT
# ============================================================

def generate_wf_splits(
    dates,
    cfg
):

    dates = sorted(
        pd.Series(dates)
        .drop_duplicates()
    )

    splits = []

    start = 0

    for i in range(
        cfg.n_wf
    ):

        train_start = start

        train_end = (
            train_start
            + cfg.train_days
        )

        test_end = (
            train_end
            + cfg.test_days
        )

        if test_end > len(dates):
            break

        train_dates = dates[
            train_start:train_end
        ]

        test_dates = dates[
            train_end:test_end
        ]

        splits.append(
            (
                train_dates,
                test_dates
            )
        )

        start = train_end

    return splits


# ============================================================
# 25. WALK FORWARD
# ============================================================

def walk_forward(
    fac,
    ret,
    factors,
    cfg
):

    dates = sorted(
        fac[
            cfg.date_col
        ].unique()
    )

    splits = generate_wf_splits(
        dates,
        cfg
    )

    all_results = []

    for fold, (
        train_dates,
        test_dates
    ) in enumerate(
        splits,
        start=1
    ):

        log.info("=" * 70)
        log.info(
            "WFA Fold %d",
            fold
        )
        log.info("=" * 70)

        train_fac = fac[
            fac[cfg.date_col].isin(
                train_dates
            )
        ].copy()

        train_ret = ret[
            ret[cfg.date_col].isin(
                train_dates
            )
        ].copy()

        test_fac = fac[
            fac[cfg.date_col].isin(
                test_dates
            )
        ].copy()

        test_ret = ret[
            ret[cfg.date_col].isin(
                test_dates
            )
        ].copy()

        # --------------------------------
        # Train
        # --------------------------------

        best_w, train_result = optimize_weights(
            train_fac,
            train_ret,
            factors,
            cfg
        )

        # --------------------------------
        # OOS
        # --------------------------------

        test_tmp = test_fac[
            [
                cfg.date_col,
                cfg.instrument_col
            ] + factors
        ].copy()

        test_tmp["ensemble"] = (
            test_tmp[factors]
            .fillna(0)
            .values
            @ best_w
        )

        weights = construct_long_only_portfolio(
            test_tmp,
            "ensemble",
            cfg
        )

        gross = portfolio_return(
            weights,
            test_ret,
            cfg
        )

        turnover = calculate_turnover(
            weights,
            cfg
        )

        net, cost = calculate_net_return(
            gross,
            turnover,
            cfg
        )

        metrics = performance_metrics(
            net,
            turnover
        )

        oos_ic = ensemble_daily_ic(
            test_fac,
            test_ret,
            factors,
            best_w,
            cfg
        )

        oos_icir = hac_icir(
            oos_ic,
            cfg.hac_lags
        )

        log.info(
            "Fold %d OOS | "
            "NetIR=%.4f | ICIR=%.4f | "
            "Return=%.2f%% | "
            "Turnover=%.2f%% | "
            "DD=%.2f%%",
            fold,
            metrics["net_ir"],
            oos_icir,
            metrics["ann_return"] * 100,
            metrics["avg_turnover"] * 100,
            metrics["maxdd"] * 100
        )

        all_results.append({

            "fold": fold,

            "train_start": train_dates[0],

            "train_end": train_dates[-1],

            "test_start": test_dates[0],

            "test_end": test_dates[-1],

            "net_ir": metrics["net_ir"],

            "ann_return": metrics["ann_return"],

            "maxdd": metrics["maxdd"],

            "turnover": metrics["avg_turnover"],

            "icir_hac": oos_icir,

            "mean_rank_ic": oos_ic.mean(),

            "ic_win_rate": (
                (oos_ic > 0).mean()
            ),

            "cost": cost.mean()

        })

    return pd.DataFrame(
        all_results
    )


# ============================================================
# 26. MAIN
# ============================================================

def main():

    cfg = CFG

    log.info("=" * 70)
    log.info("Qlib Multi-Factor Optimizer")
    log.info("=" * 70)

    # ========================================================
    # Load
    # ========================================================

    fac, ret = load_data(
        cfg
    )

    factors = get_factor_columns(
        fac,
        cfg
    )

    # ========================================================
    # Factor preprocessing
    # ========================================================

    log.info("=" * 70)
    log.info("Cross-sectional rank normalization")
    log.info("=" * 70)

    fac = cs_rank_zscore(
        fac,
        factors,
        cfg
    )

    # ========================================================
    # Factor statistics
    # ========================================================

    log.info("=" * 70)
    log.info("Factor IC / HAC-ICIR")
    log.info("=" * 70)

    stats = calculate_factor_statistics(
        fac,
        ret,
        factors,
        cfg
    )

    print(
        "\nTop Factors:\n"
    )

    print(
        stats.head(30).to_string(
            index=False
        )
    )

    # ========================================================
    # Sign normalization
    # ========================================================

    fac, sign_map = normalize_factor_sign(
        fac,
        stats,
        cfg
    )

    # ========================================================
    # Correlation
    # ========================================================

    corr = factor_correlation(
        fac,
        factors,
        cfg
    )

    # ========================================================
    # Selection
    # ========================================================

    selected = select_factors(
        stats,
        corr,
        min_icir=0.0,
        max_corr=0.85,
        max_factors=50
    )

    log.info(
        "Selected factors = %d",
        len(selected)
    )

    print(
        "\nSelected factors:"
    )

    print(
        selected
    )

    # ========================================================
    # Optimization
    # ========================================================

    log.info("=" * 70)
    log.info("Start multi-factor optimization")
    log.info("=" * 70)

    best_w, result = optimize_weights(
        fac,
        ret,
        selected,
        cfg
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Objective =",
        result["score"]
    )

    print(
        "Mean RankIC =",
        result["mean_ic"]
    )

    print(
        "HAC ICIR =",
        result["icir"]
    )

    print(
        "IC Win Rate =",
        result["ic_win_rate"]
    )

    print(
        "Net IR =",
        result["metrics"]["net_ir"]
    )

    print(
        "Annual Return =",
        result["metrics"]["ann_return"]
    )

    print(
        "Annual Vol =",
        result["metrics"]["ann_vol"]
    )

    print(
        "MaxDD =",
        result["metrics"]["maxdd"]
    )

    print(
        "Average Turnover =",
        result["metrics"]["avg_turnover"]
    )

    print(
        "Average Cost =",
        result["cost"]
    )

    # ========================================================
    # Weight table
    # ========================================================

    weight_df = pd.DataFrame({

        "factor": selected,

        "weight": best_w

    })

    weight_df = weight_df[
        weight_df["weight"] > 1e-6
    ].sort_values(
        "weight",
        ascending=False
    )

    print(
        "\nFactor Weights:"
    )

    print(
        weight_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Walk Forward
    # ========================================================

    log.info("=" * 70)
    log.info("Walk Forward OOS")
    log.info("=" * 70)

    wf = walk_forward(
        fac,
        ret,
        selected,
        cfg
    )

    print(
        "\nWFA RESULT:"
    )

    print(
        wf.to_string(
            index=False
        )
    )

    # ========================================================
    # WFA summary
    # ========================================================

    if len(wf) > 0:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "WFA SUMMARY"
        )

        print(
            "=" * 70
        )

        print(
            "Median OOS Net IR =",
            wf["net_ir"].median()
        )

        print(
            "Mean OOS Net IR =",
            wf["net_ir"].mean()
        )

        print(
            "Median OOS ICIR =",
            wf["icir_hac"].median()
        )

        print(
            "Mean OOS ICIR =",
            wf["icir_hac"].mean()
        )

        print(
            "Median Turnover =",
            wf["turnover"].median()
        )

        print(
            "Positive Net IR folds =",
            (
                wf["net_ir"] > 0
            ).mean()
        )

    # ========================================================
    # Save
    # ========================================================

    os.makedirs(
        "multi_factor_output",
        exist_ok=True
    )

    stats.to_csv(
        "multi_factor_output/factor_statistics.csv",
        index=False
    )

    weight_df.to_csv(
        "multi_factor_output/factor_weights.csv",
        index=False
    )

    wf.to_csv(
        "multi_factor_output/walk_forward.csv",
        index=False
    )

    corr.to_csv(
        "multi_factor_output/factor_correlation.csv"
    )

    log.info(
        "Results saved to multi_factor_output/"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()