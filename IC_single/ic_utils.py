"""
ic_utils.py — 单个 158 Alpha 因子研究的公共工具模块。

复用 Qlib 框架的地方：
  * qlib.init / qlib.data.D                        : 数据接入
  * qlib.contrib.data.handler.Alpha158             : 因子表（取 158 因子名称与表达式）
  * qlib.contrib.evaluate.risk_analysis            : 组合风险指标（年化、IR、最大回撤）
  * qlib.contrib.evaluate.long_short_backtest      : 含交易成本的多空组合回测
  * qlib.contrib.report.analysis_model /
    qlib.contrib.report.analysis_position          : 原生 IC / 分组 / 自相关 / 换手图（plotly）

数据格式约定：
  * LONG   : pandas MultiIndex (datetime, instrument)，1 个或多个列 —— 用于 IC、中性化
  * WIDE   : pandas DataFrame (datetime x instrument) —— 用于分组回测、衰减、换手
"""

import os
import multiprocessing

# 强制使用 fork，规避 Python3.14 下 joblib/cloudpickle 在 forkserver 下的兼容性问题
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Qlib 原生风险/回测分析（复用）
from qlib.contrib.evaluate import risk_analysis, long_short_backtest

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
PROVIDER_URI = "/home/fei/.qlib/qlib_data/cn_data"
REGION = "cn"
INSTRUMENTS = {"market": "csi300", "filter_pipe": []}   # 沪深300 股票池
BENCHMARK = "SH000300"

# 未来 1 日收益（在 t 日可知的 t -> t+1 收益）
FWD1_EXPR = "Ref($close, -1)/$close - 1"

# 中性化控制变量（尽量复用 Alpha158 因子 + 价格/成交量衍生）
#   Size     : Log(成交额) 代理（无直接市值字段时）
#   Beta     : BETA20 = Slope($close,20)/$close
#   Momentum : ROC20  = Ref($close,20)/$close - 1
#   Volatility: STD20 = Std($close,20)/$close
#   Liquidity: VMA20  = Mean($volume,20)/($volume+1e-12)
# CONTROL_EXPRS = {
#     "SIZE": "Log($close*$volume)",
#     "BETA": "Slope($close, 252)/$close",
#     "MOM": "Ref($close, 20)/$close - 1",
#     "VOL": "Std($close, 20)/$close",
#     "LIQ": "Mean($volume, 20)/($volume + 1e-12)",
# }
CONTROL_EXPRS = {
    "SIZE": "Log($close*$volume)",
    "BETA": "Slope($close, 252)/$close",
}

def qlib_init():
    qlib.init(provider_uri=PROVIDER_URI, region=REGION)


def get_alpha158_names():
    """返回 Alpha158 的 158 个因子名（不含 LABEL0）。"""
    fields, names = Alpha158DL.get_feature_config()
    return list(names), list(fields)

def cal_t(d_r):
    d_r = d_r.dropna()

    n = len(d_r)

    # IC_t = mean + error_t
    X = np.ones((n, 1))
    model = sm.OLS(d_r.values, X).fit()

    # HAC
    cov = cov_hac(model, nlags=5)

    # HAC 标准误
    se_hac = np.sqrt(cov[0, 0])

    # HAC t-stat
    ic_t = model.params[0] / se_hac
    return ic_t

def normalize_long(df: pd.DataFrame) -> pd.DataFrame:
    """将 (instrument, datetime) 的 MultiIndex 统一为 (datetime, instrument)。"""
    idx = df.index

    def is_date_level(i):
        try:
            pd.to_datetime(idx.get_level_values(i)[:3])
            return True
        except Exception:
            return False

    dt_i = [i for i in range(idx.nlevels) if is_date_level(i)][0]
    inst_i = 1 - dt_i
    df = df.reorder_levels([dt_i, inst_i])
    dts = pd.to_datetime(df.index.get_level_values(0))
    insts = df.index.get_level_values(1)
    df.index = pd.MultiIndex.from_arrays([dts, insts], names=["datetime", "instrument"])
    return df


def load_long(exprs, start, end):
    """加载 qlib 表达式为 LONG 格式 (datetime, instrument)。exprs 可为 str 或 list[str]。

    返回列名为表达式（单表达式时列名为该表达式字符串）。
    """
    if isinstance(exprs, str):
        exprs = [exprs]
    df = D.features(INSTRUMENTS, exprs, start_time=start, end_time=end, freq="day")
    df = normalize_long(df)
    return df


def to_wide(long_df: pd.DataFrame, col=None) -> pd.DataFrame:
    """将 LONG 单列转为 WIDE (datetime x instrument)。"""
    if col is not None:
        long_df = long_df[[col]]
    s = long_df.iloc[:, 0] if long_df.shape[1] == 1 else long_df
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    wide = s.reset_index().pivot(index="datetime", columns="instrument", values=s.name)
    return wide.sort_index()


def load_factor_long(expr, name, start, end):
    """加载单个因子为 LONG 格式，列名设为 name。"""
    df = load_long([expr], start, end)
    df = df.copy()
    df.columns = [name]
    return df


def load_benchmark_ret(start, end):
    """基准日收益序列（index=datetime）。"""
    df = D.features([BENCHMARK], [FWD1_EXPR], start_time=start, end_time=end, freq="day")
    df = normalize_long(df)
    s = df.iloc[:, 0].reset_index(level=1, drop=True)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.name = "bench_ret"
    return s


def load_fwd1_long(start, end):
    """个股未来 1 日收益 LONG (datetime, instrument)。"""
    return load_long([FWD1_EXPR], start, end).rename(columns={FWD1_EXPR: "fwd1"})


def forward_cumprod_ret(fwd1_wide: pd.DataFrame, h: int) -> pd.DataFrame:
    """由个股 1 日前向收益宽表，计算未来 h 日累计收益宽表（date x instrument）。

    在换仓日 t 建仓、持有 t..t+h-1 共 h 个交易日，
    收益 = product_{k=0}^{h-1}(1 + 个股日收益) - 1。

    高效实现：用对数累加替代 rolling.apply(np.prod)（rolling.apply 为逐窗口 Python
    回调，慢；rolling.sum 在 C 层向量化）。product = exp(cumsum(log(1+r)))。
    """
    grow = 1.0 + fwd1_wide.astype(float)
    # 退化情形（如极端负收益导致 1+r<=0）：回退到精确乘积，保证数值完全一致
    if (grow <= 0).any().any():
        rev = grow.iloc[::-1]
        prod = rev.rolling(h, min_periods=h).apply(np.prod, raw=True).iloc[::-1] - 1.0
        return prod
    # 正常情形（真实日收益 1+r>0）：用 log 累加替代 rolling.apply(np.prod)，
    # rolling.apply 为逐窗口 Python 回调（慢），rolling.sum 在 C 层向量化。
    rev_log = np.log(grow).iloc[::-1]
    cum_log = rev_log.rolling(h, min_periods=h).sum()
    prod = np.exp(cum_log).iloc[::-1] - 1.0
    return prod


def fwd1_wide_from_long(fwd1_long: pd.DataFrame) -> pd.DataFrame:
    return to_wide(fwd1_long, "fwd1")


# ----------------------------------------------------------------------------
# IC 计算
# ----------------------------------------------------------------------------
def calc_daily_ic(fac_long: pd.DataFrame, label_long: pd.DataFrame):
    """计算单因子的日度 Pearson IC 与 Spearman RankIC（LONG 输入）。

    返回 (daily_pearson: Series, daily_rank: Series)，按日期索引。

    高效实现：完全向量化。相关系数 = Σ(cx·cy) / sqrt(Σcx²·Σcy²)，
    其中 cx = x - group_mean，用 groupby transform 一次性算完，避免逐日 Python 循环。
    """
    fcol = fac_long.columns[0]
    lcol = label_long.columns[0]
    common = fac_long.join(label_long, how="inner").dropna()
    if common.empty:
        empty = pd.Series(dtype=float, name="pearson").sort_index()
        return empty, empty.copy()

    sidx = common.index.get_level_values(0)
    gp = common.groupby(level=0, sort=True)

    # ---- Pearson IC（向量化）----
    cf = common[fcol].values - gp[fcol].transform("mean").values
    cl = common[lcol].values - gp[lcol].transform("mean").values
    cov = pd.Series(cf * cl).groupby(sidx).sum()
    vf = pd.Series(cf * cf).groupby(sidx).sum()
    vl = pd.Series(cl * cl).groupby(sidx).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pear_v = (cov / np.sqrt(vf * vl)).values
    pear_v = np.where((vf.values > 0) & (vl.values > 0), pear_v, np.nan)
    daily_pearson = pd.Series(pear_v, index=cov.index, name="pearson")

    # ---- Spearman RankIC（先组内 rank 再做 Pearson，向量化）----
    fr = gp[fcol].rank().values
    lr = gp[lcol].rank().values
    mfr = pd.Series(fr).groupby(sidx).transform("mean").values
    mlr = pd.Series(lr).groupby(sidx).transform("mean").values
    crf = fr - mfr
    crl = lr - mlr
    covr = pd.Series(crf * crl).groupby(sidx).sum()
    vrf = pd.Series(crf * crf).groupby(sidx).sum()
    vrl = pd.Series(crl * crl).groupby(sidx).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        rk_v = (covr / np.sqrt(vrf * vrl)).values
    rk_v = np.where((vrf.values > 0) & (vrl.values > 0), rk_v, np.nan)
    daily_rank = pd.Series(rk_v, index=covr.index, name="rank")

    # 与原始实现一致：丢弃当日有效样本 < 5 的日期
    cnt = gp.size()
    mask = cnt >= 5
    return daily_pearson[mask].sort_index(), daily_rank[mask].sort_index()


def summarize_ic(d_rank: pd.Series, d_pearson: pd.Series):
    """由日度 IC 序列计算 IC 汇总指标。"""
    n = len(d_rank)
    mean_rank = d_rank.mean()
    std_rank = d_rank.std()
    median_rank = d_rank.median()
    mad = (d_rank - median_rank).abs().median()
    mean_pear = d_pearson.mean()
    std_pear = d_pearson.std()

    icir = mean_rank / std_rank if std_rank and std_rank > 0 else np.nan
    robust_icir = median_rank / (1.4826 * mad) if mad and mad > 0 else np.nan
    ic_t = mean_rank / (std_rank / np.sqrt(n)) if std_rank and std_rank > 0 else np.nan
    win_rate = float((d_rank > 0).mean())
    return {
        "RankIC": float(mean_rank),
        "RankIC_std": float(std_rank),
        "RankIC_median": float(median_rank),
        "ICIR": float(icir),
        "Robust_ICIR": float(robust_icir),
        "IC_t": float(ic_t),
        "IC_win_rate": win_rate,
        "P_IC_gt_0": win_rate,
        "IC_pos_ratio": win_rate,
        "IC_neg_ratio": float((d_rank < 0).mean()),
        "IC_zero_ratio": float((d_rank == 0).mean()),
        "Pearson_IC": float(mean_pear),
        "Pearson_IC_std": float(std_pear),
        "Pearson_IR": float(mean_pear / std_pear) if std_pear and std_pear > 0 else np.nan,
        "N_days": int(n),
    }


# ----------------------------------------------------------------------------
# 分组回测
# ----------------------------------------------------------------------------
def group_assignments(fac_wide: pd.DataFrame, horizon: int, n_groups: int = 10):
    """向量化计算分组矩阵，供 group_backtest / 换手率分析复用。

    返回 (G, reb_pos, reb_sorted)：
      G          : (n_dates x n_inst) 分组矩阵，元素为 0..n_groups-1，-1 表示无分组
      reb_pos    : 有效换仓日在 fac_wide.index 中的位置（ndarray，升序）
      reb_sorted : 有效换仓日数组（Timestamp，升序）

    高效实现：仅做一次整表 rank(axis=1)，再用“等宽分箱”把每行 rank 直接映射到分组，
    完全避免对 ~4000 个换仓日逐个调用 pd.qcut（原实现的主要耗时点）。
    注意：不构造 membership 字典——分组回测直接用矩阵切片，换手率用布尔矩阵运算。
    """
    dates = list(fac_wide.index)
    n_d = len(dates)

    reb_arr = np.array(dates[::horizon])
    reb_pos_all = fac_wide.index.get_indexer(reb_arr)   # 候选换仓日位置（O(1)）

    ranks = fac_wide.rank(axis=1, method="first")        # 整表向量化 rank
    cnt = ranks.notna().sum(axis=1).values.astype(float)
    rv = ranks.values
    chunk = cnt[:, None] / n_groups
    with np.errstate(invalid="ignore", divide="ignore"):
        # 等宽分箱：rank 1..n -> 组 0..n_groups-1（对连续整 rank 等价于 qcut labels=False）
        gv = np.where(~np.isnan(rv), np.floor((rv - 1) / chunk).astype(np.intp), -1)
    gv = np.clip(gv, -1, n_groups - 1)

    good = reb_pos_all >= 0
    cnt_reb = np.zeros(len(reb_pos_all), dtype=int)
    cnt_reb[good] = (gv[reb_pos_all[good]] >= 0).sum(axis=1)
    keep = good & (cnt_reb >= n_groups)
    reb_pos = reb_pos_all[keep]
    reb_sorted = [dates[p] for p in reb_pos]
    return gv, reb_pos, np.array(reb_sorted)


# def group_backtest(fac_wide: pd.DataFrame, ret1_wide: pd.DataFrame,
#                    horizon: int, n_groups: int = 10):
#     """按因子分 n_groups 组做分组回测，返回 (nav_df, daily_df)。

#     G1=因子最低组，Gn=因子最高组；另计算多空组合 LS = Gn - G1。
#     nav_df / daily_df 列：G1..Gn, LS。ret1_wide 为个股 1 日前向收益宽表。

#     高效实现：
#       1) group_assignments 一次算完所有换仓日分组（无逐日 qcut）；
#       2) 用切片赋值把每个换仓日的分组“覆盖”到其持有区间 [reb+1, next_reb]，
#          第 k 个换仓日的分组用于 (reb_k, reb_{k+1}]（换仓日当天用上一期，
#          与原始 searchsorted(left)-1 语义一致，避免前视）；
#       3) 用 NumPy 掩码一次性算所有日期的组分收益。
#     """
#     dates = list(ret1_wide.index)
#     insts = list(ret1_wide.columns)
#     n_d, n_i = len(dates), len(insts)

#     G, reb_pos, reb_sorted = group_assignments(fac_wide, horizon, n_groups)
#     if len(reb_pos) == 0:
#         empty = pd.DataFrame(0.0, index=dates,
#                              columns=[f"G{g}" for g in range(1, n_groups + 1)] + ["LS"])
#         return empty.cumprod(), empty

#     # 成员矩阵 mem[date, instrument] = group_index(0..n_groups-1)；-1 表示无持仓
#     mem = np.full((n_d, n_i), -1, dtype=np.int8)
#     for k in range(len(reb_pos)):
#         start = reb_pos[k] + 1
#         end = reb_pos[k + 1] + 1 if k + 1 < len(reb_pos) else n_d
#         mem[start:end] = G[reb_pos[k]].astype(np.int8)

#     ret_vals = ret1_wide.values.astype(float)
#     gcols = [f"G{g}" for g in range(1, n_groups + 1)]
#     daily = pd.DataFrame(0.0, index=dates, columns=gcols + ["LS"])
#     for gi in range(n_groups):
#         base = (mem == gi)
#         valid = base & ~np.isnan(ret_vals)          # 与原 .dropna() 一致
#         cnt = valid.sum(axis=1)
#         with np.errstate(invalid="ignore", divide="ignore"):
#             mean = (np.where(valid, ret_vals, 0.0)).sum(axis=1) / np.maximum(cnt, 1)
#         mean = np.where(cnt == 0, 0.0, mean)
#         daily[gcols[gi]] = mean
#     daily["LS"] = daily[gcols[-1]] - daily[gcols[0]]

#     nav_df = (1.0 + daily.fillna(0.0)).cumprod()
#     return nav_df, daily

def group_backtest(
    fac_wide: pd.DataFrame,
    ret1_wide: pd.DataFrame,
    horizon: int,
    n_groups: int = 10
):
    """按因子分组回测。

    G1 = 因子最低组
    Gn = 因子最高组
    LS = Gn - G1

    假设：
        ret1_wide.loc[t] = t -> t+1 的前向收益

    因子在 t 日收盘形成组合，
    因此 t 日对应的 ret1 应由当天形成的组合获得。
    """

    # 保证因子和收益的日期、股票顺序严格一致
    fac_wide = fac_wide.reindex(
        index=ret1_wide.index,
        columns=ret1_wide.columns
    )

    dates = ret1_wide.index
    n_d = len(dates)
    n_i = len(ret1_wide.columns)

    G, reb_pos, _ = group_assignments(
        fac_wide, horizon, n_groups
    )

    gcols = [f"G{g}" for g in range(1, n_groups + 1)]
    cols = gcols + ["LS"]

    if len(reb_pos) == 0:
        empty = pd.DataFrame(
            0.0,
            index=dates,
            columns=cols
        )
        return empty.cumprod(), empty

    # -1 = 无持仓
    mem = np.full(
        (n_d, n_i),
        -1,
        dtype=np.int16
    )

    # 第 k 个 reb 日形成的组合，
    # 持有到下一个 reb 日之前
    for k, reb in enumerate(reb_pos):
        start = reb
        end = (
            reb_pos[k + 1]
            if k + 1 < len(reb_pos)
            else n_d
        )

        if start < end:
            mem[start:end] = G[reb].astype(np.int16)

    ret_vals = ret1_wide.to_numpy(dtype=float)

    daily = pd.DataFrame(
        0.0,
        index=dates,
        columns=cols
    )

    for gi in range(n_groups):

        valid = (
            (mem == gi)
            & np.isfinite(ret_vals)
        )

        cnt = valid.sum(axis=1)

        sums = np.where(
            valid,
            ret_vals,
            0.0
        ).sum(axis=1)

        mean = np.divide(
            sums,
            cnt,
            out=np.zeros(n_d, dtype=float),
            where=cnt > 0
        )

        daily[gcols[gi]] = mean

    # 多空
    daily["LS"] = (
        daily[gcols[-1]]
        - daily[gcols[0]]
    )
    group_avg = daily[gcols].mean(axis=1)
    daily["LV"] = daily[gcols[-1]] - group_avg

    nav_df = (
        1.0 + daily
    ).cumprod()

    return nav_df, daily

def monotonicity(group_ret: pd.Series):
    """计算 Corr(组序号, 组收益)，衡量单调性强弱。"""
    ks = [c for c in group_ret.index if c not in ("LS", "BENCH")]
    xs = np.arange(1, len(ks) + 1)
    ys = np.asarray(group_ret[ks].values, dtype=float)
    if len(xs) < 3:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


# ----------------------------------------------------------------------------
# 风险指标
# ----------------------------------------------------------------------------
def qlib_risk_metrics(daily_ret: pd.Series, freq: str = "day"):
    """复用 Qlib 原生 risk_analysis（sum 口径：年化=mean*N）。"""
    r = daily_ret.astype(float).fillna(0.0)
    if len(r) == 0:
        return {}
    res = risk_analysis(r, freq=freq)
    return {k: float(v) for k, v in res["risk"].to_dict().items()}


def geom_metrics(daily_ret: pd.Series, bench_ret: pd.Series = None):
    """几何口径绩效指标（与文档示例一致：年化收益 / Sharpe / 最大回撤 / IR）。"""
    r = daily_ret.astype(float).fillna(0.0)
    n = len(r)
    total = float((1.0 + r).prod() - 1.0)
    ann_ret = float((1.0 + total) ** (252.0 / n) - 1.0) if n > 0 else np.nan
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    nav = (1.0 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1.0).min())
    res = {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }
    if bench_ret is not None:
        excess = (r - bench_ret.reindex(r.index).fillna(0.0))
        ir = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else np.nan
        ann_excess = float((1.0 + excess.fillna(0.0)).prod() ** (252.0 / n) - 1.0) if n > 0 else np.nan
        res.update({"excess_ann_return": ann_excess, "information_ratio": ir})
    return res


def long_short_with_cost(pred_wide, topk=50, open_cost=0.0, close_cost=0.0):
    """复用 Qlib 原生 long_short_backtest 计算含交易成本的多空收益。

    pred_wide: (datetime x instrument) 宽表，单列为因子值。
    返回 dict: long / short / long_short 三个 Series。
    """
    pred = pred_wide.stack().rename("score")
    pred.index = pred.index.reorder_levels(["datetime", "instrument"])
    pred = pred.sort_index()
    res = long_short_backtest(
        pred, topk=topk, open_cost=open_cost, close_cost=close_cost,
        shift=1, deal_price="$close",
    )
    return res


# ----------------------------------------------------------------------------
# 中性化（横截面 OLS 残差）
# ----------------------------------------------------------------------------
def neutralize_factor(fac_wide: pd.DataFrame, controls: dict):
    """横截面回归中性化：factor ~ const + controls，取残差。

    controls: dict[name -> WIDE (datetime x instrument) 宽表]
    返回残差 WIDE 宽表（datetime x instrument）。

    高效实现：把因子与控制变量预取为对齐的 ndarray，逐日做纯 NumPy OLS，
    省去原来每日构造 DataFrame / concat 的开销（原实现约 4000 次 DataFrame 拼接）。
    """
    dates = list(fac_wide.index)
    insts = list(fac_wide.columns)
    n_i = len(insts)
    n_c = len(controls)
    resid = pd.DataFrame(np.nan, index=dates, columns=insts)
    resid.columns.name = "instrument"

    # 控制变量拼成 (n_dates, n_instruments, n_controls)，与因子对齐到同一列序
    ctrl_arr = np.stack(
        [controls[c].reindex(index=dates, columns=insts).values.astype(float) for c in controls],
        axis=-1,
    )
    y_arr = fac_wide.reindex(index=dates, columns=insts).values.astype(float)
    const = np.ones((n_i, 1))

    for i, dt in enumerate(dates):
        y = y_arr[i]
        Xc = ctrl_arr[i]                      # (n_i, n_c)
        X = np.concatenate([const, Xc], axis=1)  # (n_i, n_c+1)
        valid = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if valid.sum() < X.shape[1] + 1:
            continue
        Xm = X[valid]
        ym = y[valid]
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        resid_row = np.full(n_i, np.nan)
        resid_row[valid] = ym - Xm @ beta
        resid.iloc[i] = resid_row
    return resid.sort_index()


# ----------------------------------------------------------------------------
# 行业中性化（横截面行业哑变量回归残差）
# ----------------------------------------------------------------------------
def neutralize_industry(fac_wide: pd.DataFrame, ind_wide: pd.DataFrame):
    """行业中性化：每日对因子做 factor ~ const + Σ industry_dummies，取残差。

    行业分类可随时间变化（逐日重建哑变量）。industry 缺失的个股当日不参与回归。

    ind_wide: 行业分类宽表 (datetime x instrument)，单元格为行业代码（字符串/数值）。
    返回残差 WIDE 宽表（datetime x instrument）。
    """
    dates = list(fac_wide.index)
    insts = list(fac_wide.columns)
    fac = fac_wide.reindex(index=dates, columns=insts).values.astype(float)
    ind = ind_wide.reindex(index=dates, columns=insts)
    resid = pd.DataFrame(np.nan, index=dates, columns=insts)
    resid.columns.name = "instrument"

    for i, dt in enumerate(dates):
        y = fac[i]
        codes = ind.iloc[i].values
        valid = ~np.isnan(y) & (~pd.isna(codes))
        if valid.sum() < 3:
            continue
        cats = pd.Categorical(codes[valid])
        n_l = len(cats.categories)
        Xd = np.zeros((valid.sum(), n_l))
        Xd[np.arange(valid.sum()), cats.codes] = 1.0
        X = np.concatenate([np.ones((valid.sum(), 1)), Xd], axis=1)
        ym = y[valid]
        beta, *_ = np.linalg.lstsq(X, ym, rcond=None)
        rr = np.full(len(insts), np.nan)
        rr[valid] = ym - X @ beta
        resid.iloc[i] = rr
    return resid.sort_index()


def neutralize_industry_size(fac_wide, ind_wide, controls=None):
    """行业 + Size（及可选其它连续控制变量）联合横截面中性化。

    每日截面回归：factor ~ const + Σ连续控制变量(controls) + Σ行业哑变量，取残差。
      - controls: dict[name -> WIDE]，如 {"SIZE": size_wide}；可空（仅行业）。
      - 行业哑变量由 ind_wide 当天分类 one-hot 得到（含全部类别 + const）。
        因 const 落在行业哑变量张成空间内，设计阵列秩亏，但残差（投影）唯一，
        等价于「剔除各行业均值 + 市值线性暴露」，与逐列 β 的可辨识性无关。
    返回残差 WIDE 宽表（datetime x instrument）。
    """
    dates = list(fac_wide.index)
    insts = list(fac_wide.columns)
    n_i = len(insts)
    resid = pd.DataFrame(np.nan, index=dates, columns=insts)
    resid.columns.name = "instrument"

    y_arr = fac_wide.reindex(index=dates, columns=insts).values.astype(float)
    ctrl_arr = (np.stack(
        [controls[c].reindex(index=dates, columns=insts).values.astype(float)
         for c in controls], axis=-1) if controls else None)
    ind_arr = ind_wide.reindex(index=dates, columns=insts).values

    for i, dt in enumerate(dates):
        y = y_arr[i]
        codes = ind_arr[i]
        valid = ~np.isnan(y) & ~pd.isna(codes)
        if ctrl_arr is not None:
            valid = valid & ~np.isnan(ctrl_arr[i]).any(axis=1)
        if valid.sum() < 3:
            continue
        ym = y[valid]
        X_parts = [np.ones((valid.sum(), 1))]
        if ctrl_arr is not None:
            X_parts.append(ctrl_arr[i][valid])
        cats = pd.Categorical(codes[valid])
        Xd = np.zeros((valid.sum(), len(cats.categories)))
        Xd[np.arange(valid.sum()), cats.codes] = 1.0
        X_parts.append(Xd)
        X = np.concatenate(X_parts, axis=1)
        if X.shape[0] < X.shape[1] + 1:   # 自由度不足
            continue
        beta, *_ = np.linalg.lstsq(X, ym, rcond=None)
        rr = np.full(n_i, np.nan)
        rr[valid] = ym - X @ beta
        resid.iloc[i] = rr
    return resid.sort_index()


def load_industry_wide(csv_path, start=None, end=None):
    """读取行业标签 CSV -> WIDE (datetime x instrument) 行业分类代码。

    期望列：datetime, instrument, industry（或 INDUSTRY / sector / SECTOR）。
    无文件或路径为空时返回 None。
    """
    if not csv_path or not os.path.exists(csv_path):
        return None
    ind = pd.read_csv(csv_path)
    ind_col = None
    for c in ("industry", "INDUSTRY", "sector", "SECTOR"):
        if c in ind.columns:
            ind_col = c
            break
    if ind_col is None or "datetime" not in ind.columns or "instrument" not in ind.columns:
        raise ValueError(
            f"industry.csv 需包含列 datetime, instrument, (industry/INDUSTRY)，实际: {list(ind.columns)}"
        )
    ind["datetime"] = pd.to_datetime(ind["datetime"])
    wide = ind.pivot(index="datetime", columns="instrument", values=ind_col).sort_index()

    # 单日快照（如 industry.csv 只含某一天的行业分类）：将该天的分类按工作日
    # 复制到 [start, end] 全程，使下游可跨日对齐做横截面行业中性化。
    # 注意：原 CSV 的日期可能不在分析区间内，复制时以传入的 start/end 为准生成日期轴。
    if len(wide) == 1:
        src = wide.iloc[0]
        lo = pd.to_datetime(start) if start else wide.index.min()
        hi = pd.to_datetime(end) if end else wide.index.max()
        span = pd.date_range(lo, hi, freq="B")  # 工作日（交易日 ⊆ 工作日，reindex 必有匹配）
        wide = pd.DataFrame(
            np.tile(src.values, (len(span), 1)),
            index=span, columns=wide.columns,
        )
        wide.index.name = "datetime"
        return wide

    if start:
        wide = wide.loc[wide.index >= pd.to_datetime(start)]
    if end:
        wide = wide.loc[wide.index <= pd.to_datetime(end)]
    return wide


# ----------------------------------------------------------------------------
# 画图（matplotlib Agg）
# ----------------------------------------------------------------------------
def plot_series(d, path, title="", ylabel="", legend=True):
    if not HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    if isinstance(d, pd.DataFrame):
        for c in d.columns:
            ax.plot(d.index, d[c], label=str(c))
    else:
        ax.plot(d.index, d.values, label=getattr(d, "name", "series"))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bar(d, path, title="", ylabel="", rot=45, max_ticks=20):
    if not HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    if isinstance(d, pd.DataFrame):
        d.plot(kind="bar", ax=ax)
    else:
        ax.bar(range(len(d)), d.values)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([str(x) for x in d.index], rotation=rot)
        # 长序列时只保留少量可读的刻度，避免标签挤成一团
        if len(d) > max_ticks:
            step = max(1, len(d) // max_ticks)
            ax.set_xticks(range(0, len(d), step))
            ax.set_xticklabels([str(d.index[i]) for i in range(0, len(d), step)],
                               rotation=rot, fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_hist(series, path, title="", bins=60, xlabel="", ylabel="频数", mean_std=True):
    """绘制数值分布直方图（用于 IC 分布等），x 轴为 bin 区间，标签可读。"""
    if not HAVE_MPL:
        return
    vals = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
    if mean_std and len(vals) > 1:
        m, s = vals.mean(), vals.std()
        ax.axvline(m, color="#C44E52", linestyle="--", linewidth=1.5,
                   label=f"均值={m:.4f}")
        ax.axvline(m + s, color="#55A868", linestyle=":", linewidth=1.2)
        ax.axvline(m - s, color="#55A868", linestyle=":", linewidth=1.2)
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
