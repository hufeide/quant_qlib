#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
workflow_tcn.py — 用 qlib 的「代码接口」搭一条工作流：在【中性化因子】上训练
【qlib 自带的 TCN 模型】(qlib.contrib.model.pytorch_tcn.TCN)，并回测
（交易成本口径一致）。

与 `qrun xxx.yaml` 的关系：用的是同一套积木（DataLoader -> DataHandlerLP ->
DatasetH -> Model -> Record），只是全部显式写在代码里。

一、数据
  因子：`run_ic.py` 落盘的中性化因子（行业 + Size 中性化后的 OLS 残差）
        results/alpha158_all/neutralized_factors.pkl
        长表 (datetime, instrument) × 20 个代表因子（从 158 个 Alpha158 因子中按聚类精选，float32）
  标签：qlib 表达式 LABEL_EXPR（默认 Ref($close, -1)/$close - 1，
        与 Alpha158 官方 LABEL0 同口径）
  组装：StaticDataLoader(feature/label) -> DataHandlerLP -> DatasetH
        learn 侧：Fillna(特征) + DropnaLabel（丢掉无标签样本）+ CSZScoreNorm（标签逐日截面标准化）
        infer 侧：Fillna(特征)（原样输出预测分数，供 IC 分析 / 回测使用）
        注：TCN 对特征中的 NaN 不鲁棒（前向会传播 NaN 让损失变 NaN），故在 learn/infer
            两侧的「特征」上做一次零值填充，标签仍按 DropnaLabel 丢掉空样本。

二、模型（qlib 自带「时序」TCN）
  直接调用 qlib 内部的 `qlib.contrib.model.pytorch_tcn.TCN`（下称 QlibTCN），这才是
  真正的时序 TCN：每个样本是「某只股票过去 n_days 天的 F 个中性化因子」构成的时间窗口
  (N, F * n_days)。代码里先按「因子优先、滞后在后」拼成宽表，qlib TCN 内部
  x.reshape(N, F, n_days) 后，因果卷积（dilated causal conv + 残差）作用在【时间轴】上、
  通道数 = 因子数 F，从而捕捉因子随时间的演化（纠正早期把因子维度当序列的错误用法）：
    - d_feat 必须等于因子通道数 F（自动取选中因子个数），时间窗口长度 n_days 由
      --n-days 控制（默认 20 天）；卷积在时间维，故 TCN 真正在「时序」上建模。
    - 损失：MSE(s, 标签)（标签已被 CSZScoreNorm 截面标准化，与最大化 IC 同解）。
    - 标签：优先用因子数据里的 `NFWD1`（前 1 日收益率），无则回退 D.features(label_expr)。
    - 早停：以验证集 MSE 最小为准（qlib TCN 内部实现）。

  训练区间：train = ("2017-01-01", "2023-12-31")，valid / test 见 SEGMENTS，均可用命令行覆盖。

三、回测与记账
  默认策略 BenchmarkTiltStrategy（在 hs300 基准权重基础上按 pred 做乘性倾斜的增强
  指数策略）+ SimulatorExecutor，交易成本同样是双边 0.002；
  --strategy topk 可切回原 TopkDropoutStrategy 仅作对照。
  训练结束打印「IC / 换手率代理」，回测结束打印实际换手率与成本拖累。

  实测（train 2017-01-01~2023-12-31，test 2025-01-01~2026-06-30，--n-drop 1）：
  以 qlib TCN 为 signal 喂入同一回测框架即可，IC / 含成本表现随超参变化，请自行比对。

运行（必须用 qlib_me 环境，且 cwd 不要落在 qlib 源码根目录以免遮蔽已安装的 qlib 包）：
  JOBLIB_START_METHOD=fork LOKY_START_METHOD=fork \
      ~/miniconda3/envs/qlib_me/bin/python workflow_tcn.py

常用参数：
  --train 2017-01-01,2023-12-31   训练区间
  --rounds 200 --early-stop 20     训练 epoch 数 / 早停耐心
  --d-feat 1 --n-chans 128          通道数 / 序列通道维度
  --kernel-size 5 --num-layers 5    卷积核 / 残差块数
  --dropout 0.5 --lr 1e-4 --batch-size 2000
  --gpu 0                          GPU id（无 GPU 自动回退 cpu）
  --topk 50 --n-drop 5             回测组合参数
  --check-pkl                    只检查中性化因子 pkl 的 NaN/常数/近常数/高冗余，然后退出
"""

from __future__ import annotations

import argparse
import logging
import os
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import sys

# 保证同目录下的 benchmark_tilt_strategy 模块可被 import（无论从哪个 cwd 运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无显示器也能保存 png
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel, Fillna
from qlib.contrib.evaluate import risk_analysis
# qlib 自带的 TCN 模型（pytorch 实现）
from qlib.contrib.model.pytorch_tcn import TCN as QlibTCN
from qlib.log import get_module_logger
# 在基准权重上按 pred 做乘性倾斜的增强指数策略（替换 TopkDropoutStrategy）
from benchmark_tilt_strategy import BenchmarkTiltStrategy, DEFAULT_WEIGHT_FILE
# 纯被动策略：直接按权重表（weights_day.txt）持仓，不做 pred 倾斜（用作基准对照）
from weight_hold_strategy import WeightHoldStrategy
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

QLIB_ROOT = "/home/fei/workspace/qlib"                       # 本地 qlib 源码（qlib/qlib）
PROVIDER_URI = "/home/fei/.qlib/qlib_data/cn_data"           # qlib 二进制数据
MLFLOW_URI = "sqlite:////home/fei/workspace/mlflow.db"       # 避免 mlruns 文件后端

# 中性化因子（run_ic.py 的产物）
NEUTRAL_PKL = os.path.join(HERE, "results", "alpha158_all_no", "neutralized_factors.pkl")
# 因子指标表（run_ic.py 产物：含「因子类别」「Rank IC」等列，用于按类别筛选）
METRICS_CSV = os.path.join(HERE, "results", "alpha158_all_no", "alpha158_factor_metrics.csv")
# 交给 qlib DataLoader 读取的中间文件（因子/标签按 segment 区间切片后落盘）
WORK_DIR = os.path.join(HERE, "results", "qlib_data")

MARKET = "csi300"                                            # 股票池（与 run_ic.py 一致）
BENCHMARK = "SH000300"
LABEL_EXPR = "Ref($close, -1)/$close - 1"            # Alpha158 官方 LABEL0

# 训练 / 验证 / 测试（回测）区间
SEGMENTS = {
    "train": ("2017-01-01", "2023-12-31"),
    "valid": ("2024-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2026-06-30"),
}

# 交易成本（回测口径，双边）
COST_RATE = 0.002            # 双边成本（买 0.001 + 卖 0.001）
SD_FLOOR = 1e-3              # 截面标准差下限（z-score 分母，避免除零）

# 回测
TOPK = 50
N_DROP = 5
ACCOUNT = 100_000_000

# ---- TCN 模型超参（qlib.contrib.model.pytorch_tcn.TCN）----
# 说明：这是「时序」TCN——每个样本是某只股票过去 n_days 天的 F 个因子拼成的时间窗口
# (N, F*n_days)。d_feat 自动取因子通道数 F（= 选中因子个数），时间窗口长度由 --n-days 控制；
# qlib TCN 内部 x.reshape(N, F, n_days) 后卷积作用在【时间轴】上。下面只列出与通道数无关、
# 或控制网络容量/训练过程的超参；d_feat 在 main 里按实际因子数设置，无需在此写死。
TCN_N_CHANS = 128
TCN_KERNEL_SIZE = 5
TCN_NUM_LAYERS = 5
TCN_DROPOUT = 0.5
TCN_LR = 0.0001
TCN_BATCH_SIZE = 2000
TCN_GPU = 0

log = get_module_logger("tcn_workflow", logging.INFO)


# ---------------------------------------------------------------------------
# 按日截面工具（IC / 换手率代理，与具体模型无关）
# ---------------------------------------------------------------------------
def _day_structure(index: pd.MultiIndex):
    """由 (datetime, instrument) 索引构造按日截面计算所需的三个数组。

    index 需按 datetime 排序（qlib 的 prepare 结果满足）。

    返回
      day_code : 每行所属交易日编号（0 ~ D-1，按时间递增）
      day_size : 每个交易日的样本数（长度 D）
      prev_row : 每行在「上一交易日」中同一只股票的行号，-1 表示上一日不存在该股票
    """
    dt = index.get_level_values("datetime").values
    inst = index.get_level_values("instrument")
    uniq, first, cnt = np.unique(dt, return_index=True, return_counts=True)
    day_code = np.repeat(np.arange(len(uniq)), cnt)

    prev_row = np.full(len(index), -1, dtype=np.int64)
    for k in range(1, len(uniq)):
        a0, c0 = int(first[k - 1]), int(cnt[k - 1])
        b0, c1 = int(first[k]), int(cnt[k])
        pos = pd.Series(np.arange(c0), index=inst[a0:a0 + c0])
        pos = pos[~pos.index.duplicated()]                 # 防御：同日股票代码不应重复
        m = pos.reindex(inst[b0:b0 + c1]).to_numpy()       # 当日股票在上一日的行号位置
        ok = np.flatnonzero(~pd.isna(m))
        prev_row[b0 + ok] = a0 + m[ok].astype(np.int64)
    return day_code, cnt.astype(np.int64), prev_row


def _cs_std(day_code, day_size, x, sd_floor=SD_FLOOR):
    """逐日截面 z-score：返回 (标准化后的值, 当日使用的标准差)。"""
    x = np.asarray(x, dtype=np.float64)
    n_days = len(day_size)
    mean = np.bincount(day_code, weights=x, minlength=n_days) / day_size
    c = x - mean[day_code]
    var = np.bincount(day_code, weights=c * c, minlength=n_days) / day_size
    sd = np.maximum(np.sqrt(np.maximum(var, 0.0)), sd_floor)
    return c / sd[day_code], sd


def _daily_ic(day_code, day_size, pred, label):
    """逐日截面 Pearson IC（长度 = 交易日数）。"""
    zp, _ = _cs_std(day_code, day_size, pred)
    zy, _ = _cs_std(day_code, day_size, label)
    return np.bincount(day_code, weights=zp * zy, minlength=len(day_size)) / day_size


def _turnover_proxy(day_code, day_size, prev_row, pred, sd_floor=SD_FLOOR):
    """逐日换手率代理：0.5 * Σ_i |z_ti - z_{t-1,i}| / n_t（长度 = 交易日数）。

    口径说明：z 为当日截面标准化分数，|Δz| 近似刻画组合权重的日间变化，
    0.5 的系数使之与「单边换手率」（占账户比例）同量纲。
    """
    z, _ = _cs_std(day_code, day_size, pred, sd_floor)
    cur = np.flatnonzero(prev_row >= 0)
    if cur.size == 0:
        return np.zeros(len(day_size))
    d = np.abs(z[cur] - z[prev_row[cur]])
    return 0.5 * np.bincount(day_code[cur], weights=d, minlength=len(day_size)) / day_size


def _report_segment(model, dataset, key):
    """对单个 segment 用已训练的 TCN 做预测，并报告 IC / ICIR / 换手率代理。

    与 LightGBM 的 _report_objective 等价，只是模型换成 qlib 自带 TCN：
    直接用 model.predict(dataset, segment=key) 取按 (datetime, instrument) 索引的分数。
    """
    pred = model.predict(dataset, segment=key)
    lab = dataset.prepare(key, col_set="label", data_key=DataHandlerLP.DK_L)
    # lab 是 DataFrame，必须用 columns= 重命名（Series.rename 才接受标量作 name）
    lab = lab.rename(columns={lab.columns[0]: "label"})
    df = pd.concat([pred.rename("pred"), lab], axis=1).dropna().sort_index()
    if df.empty:
        log.warning(f"  [{key}] 预测/标签对齐后为空，跳过报告")
        return None, None
    day_code, day_size, prev_row = _day_structure(df.index)
    ic = _daily_ic(day_code, day_size, df["pred"].values, df["label"].values)
    turn = _turnover_proxy(day_code, day_size, prev_row, df["pred"].values)
    cost = COST_RATE * turn
    obj = -ic + cost
    log.info(
        f"  [{key}] IC={ic.mean():+.4f}（ICIR={ic.mean() / ic.std():.3f}） "
        f"换手率代理={turn.mean():.4f} 成本={cost.mean():.6f} 目标={obj.mean():+.4f}"
    )
    R.log_metrics(
        **{
            f"{key}.daily_ic": ic.mean(),
            f"{key}.daily_icir": ic.mean() / ic.std(),
            f"{key}.turnover_proxy": turn.mean(),
            f"{key}.cost_term": cost.mean(),
            f"{key}.cost_aware_objective": obj.mean(),
        }
    )
    return ic, turn


def report_tcn_perf(model, dataset):
    """训练结束后报告：IC / 换手率代理（确认信号质量）。"""
    for key in ["train", "valid", "test"]:
        if key in dataset.segments:
            _report_segment(model, dataset, key)


# ---------------------------------------------------------------------------
# 数据：中性化因子 pkl + qlib 标签 -> DatasetH
# ---------------------------------------------------------------------------
def _as_datetime_instrument(df: pd.DataFrame) -> pd.DataFrame:
    """把索引统一成 (datetime, instrument) 且按时间有序。

    D.features 返回的是 (instrument, datetime)，run_ic.py 落盘的因子是
    (datetime, instrument)；层级顺序不一致会让 pd.concat(axis=1) 对不齐
    （索引名会变成 None），进而让 qlib 的按日处理器（CSZScoreNorm 等）报
    KeyError: 'datetime'，所以必须先统一。
    """
    names = list(df.index.names)
    if names == ["datetime", "instrument"]:
        return df if df.index.is_monotonic_increasing else df.sort_index()
    if names == ["instrument", "datetime"]:
        return df.swaplevel().sort_index()
    raise ValueError(f"意外的索引层级名称：{names}（期望 datetime/instrument）")


def _slice_by_time(df: pd.DataFrame, start, end) -> pd.DataFrame:
    dts = df.index.get_level_values("datetime")
    mask = (dts >= pd.Timestamp(start)) & (dts <= pd.Timestamp(end))
    return df[mask]


def _materialize(df: pd.DataFrame, name: str) -> str:
    """把 DataFrame 落到 WORK_DIR 并返回路径。

    StaticDataLoader 支持直接读文件，这样 Handler 内部只会留一份数据副本
    （若直接传 DataFrame，loader 的 _config 会一直持有一份引用），
    同时 Handler/Dataset 的 pickle 也会轻很多。
    """
    os.makedirs(WORK_DIR, exist_ok=True)
    path = os.path.join(WORK_DIR, name)
    df.to_pickle(path)
    log.info(f"  写出 {path}（{os.path.getsize(path) / 1e6:.1f} MB）")
    return path


# def _build_time_window(base: pd.DataFrame, n_days: int) -> pd.DataFrame:
#     """把 (datetime, instrument) × F 的因子表，按股票构建长度为 n_days 的时间窗口特征。

#     返回 (datetime, instrument) × (F * n_days)。列顺序为「因子优先、滞后在后」：
#         f0_lag0, f0_lag1, ..., f0_lag(n_days-1), f1_lag0, f1_lag1, ...
#     这样 qlib 自带 TCN 内部 x.reshape(N, F, n_days) 后，卷积作用在最后一个维度 = 时间，
#     通道数 = F（因子数），构成真正的「时序 TCN」（纠正此前把因子维度当序列的错误用法）。

#     lag=0 表示当前交易日 t，lag=n_days-1 表示 t-(n_days-1)；窗口覆盖 [t-(n_days-1) .. t]。
#     每只股票前 n_days-1 天因无足够历史会变成 NaN，这里直接 dropna 丢弃，避免把 NaN
#     当特征喂给对 NaN 不鲁棒的 TCN。
#     """
#     cols = list(base.columns)
#     # 先按股票整体平移 n_days 次（一次得到所有因子在 lag=l 的值），比逐因子平移快得多
#     shifted = [base.groupby(level="instrument", group_keys=False).shift(l) for l in range(n_days)]
#     # 重组为「因子优先」顺序：每个因子把它的 n_days 个滞后拼在一起。
#     # 注意：必须用原始列名 c 取 shifted[l][c]，再 rename 为新列名，不要先改 shifted 的列名。
#     frames = []
#     for c in cols:
#         lag_series = [shifted[l][c].rename(f"{c}_lag{l}") for l in range(n_days)]
#         frames.append(pd.concat(lag_series, axis=1))
#     windowed = pd.concat(frames, axis=1)
#     # 丢弃无法凑满窗口的行（每只股票前 n_days-1 天），避免把 NaN 当特征喂给 TCN
#     windowed = windowed.dropna()
#     return windowed
def _build_time_window(base: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """按股票构建严格时间窗口。

    输入：
        index = (datetime, instrument)
        columns = F 个特征

    输出：
        index = (datetime, instrument)
        columns = F * n_days

    列顺序：
        f0_lag0, f0_lag1, ..., f0_lag(n-1),
        f1_lag0, f1_lag1, ..., f1_lag(n-1),
        ...

    其中：
        lag=0     -> t
        lag=1     -> t-1
        ...
        lag=n-1   -> t-(n-1)

    因此最终：
        [batch, F*n_days]
            ↓
        Qlib TCN reshape
            ↓
        [batch, F, n_days]

    最后一维是时间维。
    """

    if n_days <= 0:
        raise ValueError(f"n_days 必须 > 0，当前={n_days}")

    if not isinstance(base.index, pd.MultiIndex):
        raise ValueError("base.index 必须是 MultiIndex(datetime, instrument)")

    if "datetime" not in base.index.names:
        raise ValueError("base.index 缺少 datetime level")

    if "instrument" not in base.index.names:
        raise ValueError("base.index 缺少 instrument level")

    # 确保同一股票内部时间顺序正确
    base = base.sort_index()

    cols = list(base.columns)

    # 每个 lag 按股票独立 shift，绝不会跨股票串历史
    shifted = [
        base.groupby(
            level="instrument",
            group_keys=False,
            sort=False,
        ).shift(l)
        for l in range(n_days)
    ]

    # 因子优先：
    # f0_lag0 ... f0_lagN
    # f1_lag0 ... f1_lagN
    # ...
    frames = []

    for c in cols:
        lag_series = [
            shifted[l][c].rename(f"{c}_lag{l}")
            for l in range(n_days)
        ]
        frames.append(pd.concat(lag_series, axis=1))

    windowed = pd.concat(frames, axis=1)

    # 完整窗口才保留
    windowed = windowed.dropna()

    return windowed

def select_factors_by_metrics(metrics_csv: str, rank_ic_min: float = 0.02, topk: int = 4):
    """按 metrics CSV 的「因子类别」分组，每类保留 Rank IC > rank_ic_min 且最高的 topk 个因子。

    返回筛选后的因子名列表（组内按 Rank IC 降序）。CSV 列名做了容错：自动识别
    factor / Rank IC / 因子类别 列（编码依次尝试 gbk/utf-8/latin-1）。
    """
    df = None
    for enc in ("gbk", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(metrics_csv, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise IOError(f"无法读取因子指标表：{metrics_csv}")

    factor_col = "factor" if "factor" in df.columns else df.columns[1]
    ic_col = "Rank IC" if "Rank IC" in df.columns else df.columns[3]
    cat_col = "因子类别" if "因子类别" in df.columns else df.columns[-1]

    selected = []
    selected = df.loc[
        (df[ic_col].abs() > rank_ic_min),
        factor_col
    ].tolist()
    return selected


def check_pkl_degeneracy(pkl_path: str, nan_frac_warn: float = 0.01,
                         rel_std_floor: float = 0.01, corr_sample: int = 20000):
    """检查中性化因子 pkl 是否退化：NaN 占比 / 常数 / 近常数 / 高冗余（共线）因子对。

    用法：`python workflow_tcn.py --check-pkl [--pkl 其它文件]`
    只做检查、打印退化清单后返回，不训练。

    判定规则：
      - 常数      ：整列 std == 0
      - 近常数    ：相对 std = std / 全集中位 std < rel_std_floor（消除量纲）
      - 高 NaN    ：NaN 占比 > nan_frac_warn
      - 高冗余    ：抽样计算两两 |相关系数| > 0.99 的因子对（共线）
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到 pkl：{pkl_path}")
    feat = pd.read_pickle(pkl_path)
    if not isinstance(feat, pd.DataFrame) or feat.empty:
        raise ValueError(f"{pkl_path} 内容不是非空 DataFrame")
    if list(feat.index.names) == ["instrument", "datetime"]:
        feat = feat.swaplevel().sort_index()
    elif list(feat.index.names) != ["datetime", "instrument"]:
        print(f"  [warn] 索引层级为 {feat.index.names}，按原样检查（期望 datetime/instrument）")

    n, n_col = feat.shape
    # 注意：--check-pkl 在 qlib.init() 之前运行，日志未配置，故用 print
    print("=" * 78)
    print(f"检查 pkl 退化: {pkl_path}")
    print(
        f"  形状: {feat.shape}（{n} 行 × {n_col} 列），区间 "
        f"{feat.index.get_level_values(0).min().date()} ~ "
        f"{feat.index.get_level_values(0).max().date()}"
    )

    rows = []
    for c in feat.columns:
        col = feat[c]
        nan_frac = col.isna().mean()
        s = col.dropna()
        if len(s) == 0:
            rows.append((c, int(col.isna().sum()), nan_frac, 0.0, "ALL_NAN"))
            continue
        rows.append((c, int(col.isna().sum()), nan_frac, float(s.std()), "ok"))
    df = pd.DataFrame(rows, columns=["factor", "n_nan", "nan_frac", "std", "status"])

    med_std = df["std"].median()
    df["rel_std"] = df["std"] / med_std if med_std > 0 else np.nan
    df.loc[df["std"] == 0, "status"] = "CONSTANT"
    near = (df["std"] > 0) & (df["rel_std"] < rel_std_floor)
    df.loc[near, "status"] = "NEAR_CONSTANT"
    df.loc[df["nan_frac"] > nan_frac_warn, "status"] = (
        df.loc[df["nan_frac"] > nan_frac_warn, "status"].astype(str) + f";NaN>{nan_frac_warn:.0%}"
    )

    n_const = df["status"].str.contains("CONSTANT").sum()
    n_near = df["status"].str.contains("NEAR_CONSTANT").sum()
    n_nancol = df["status"].str.contains("NaN").sum()
    print(
        f"  中位 std={med_std:.4g} | 常数={n_const} 近常数={n_near} "
        f"高NaN(>{nan_frac_warn:.0%})={n_nancol}"
    )
    bad = df[df["status"] != "ok"]
    if len(bad):
        print("  退化因子清单：")
        for _, r in bad.iterrows():
            rs = "" if pd.isna(r["rel_std"]) else f" rel_std={r['rel_std']:.3g}"
            print(f"    {r['factor']:10s} nan={r['nan_frac']:.2%} std={r['std']:.4g}{rs} -> {r['status']}")
    else:
        print("  未发现 NaN/常数/近常数列。")

    # 高冗余（共线）检查：抽样计算两两相关系数（上三角）
    try:
        samp = feat.sample(min(corr_sample, n), random_state=0)
        corr = samp.corr().abs().values
        k = corr.shape[0]
        utri = np.triu(np.ones((k, k), dtype=bool), k=1)
        hi = np.argwhere((corr > 0.99) & utri)
        if len(hi):
            names = list(samp.columns)
            print(f"  高冗余：|corr|>0.99 的因子对共 {len(hi)} 对（抽样 {len(samp)} 行）。示例：")
            for i, j in hi[:10]:
                print(f"    {names[i]} ~ {names[j]} : {corr[i, j]:.3f}")
        else:
            print(f"  高冗余：抽样 {len(samp)} 行未发现 |corr|>0.99 的因子对。")
    except Exception as e:  # noqa: BLE001
        print(f"  冗余检查跳过（{e}）")

    print("=" * 78)
    return df


def build_raw_features(pkl_path: str, segments: dict, market: str = MARKET,
                        label_expr: str = LABEL_EXPR, metrics_csv: str = METRICS_CSV,
                        rank_ic_min: float = 0.02, topk: int = 4, factor_filter: bool = True,
                        industry_mode: str = "onehot") -> tuple:
    """抽取「原始长表」特征与标签（不做时间窗口摊平），供动态窗口 Dataset 使用。

    与 build_dataset 共享同一套选因子 / 行业 onehot / 标签口径，但保持
    (T, N, F) 长表形态（index=(datetime, instrument)，columns=因子+行业 onehot），
    时间窗口由下游 Dataset.__getitem__ 按「同一只股票」切片 [L, F+K] 生成，
    **不把窗口物理复制成 (F+K)*n_days 列**，从而把内存峰值从
    N*(F+K)*n_days 降到约 N*(F+K)（~0.45 GB），n_days=100 也能在小内存机器上跑。

    返回 (feat_raw, label_raw, n_feat)：
      - feat_raw: DataFrame，index=(datetime, instrument)，columns=选中因子(+行业 onehot)
      - label_raw: DataFrame，index=(datetime, instrument)，列 "LABEL0"
      - n_feat: 通道数（= 因子数 + 行业 onehot 列数，即 TCN 输入通道 C）
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"找不到中性化因子文件：{pkl_path}\n"
            f"请先运行 run_ic.py 生成 results/alpha158_all/neutralized_factors.pkl"
        )
    feat: DataFrame | Series = pd.read_pickle(pkl_path)
    if not isinstance(feat, pd.DataFrame) or feat.empty:
        raise ValueError(f"{pkl_path} 内容不是非空 DataFrame")
    if not feat.index.is_monotonic_increasing:
        feat = feat.sort_index()
    # 先保留类别变量 INDUSTRY（字符串）与标签 NFWD1 的全量副本：它们都不在 metrics 里，
    # 下面的因子筛选(feat=feat[keep])会把它们丢掉，所以这里单独留一份。
    industry_all = feat["INDUSTRY"].copy() if "INDUSTRY" in feat.columns else None
    nfwd1_all = feat["NFWD1"].copy() if "NFWD1" in feat.columns else None

    # ---- 因子筛选：按「因子类别」分组，每类保留 Rank IC > rank_ic_min 且最高的 topk 个 ----
    if factor_filter and metrics_csv and os.path.exists(metrics_csv):
        selected = select_factors_by_metrics(metrics_csv, rank_ic_min=rank_ic_min, topk=topk)
        keep = [c for c in selected if c in feat.columns]
        missing = [c for c in selected if c not in feat.columns]
        if missing:
            log.warning(f"指标表选中但 pkl 缺失的因子（已忽略）：{missing}")
        if keep:
            dropped = [c for c in feat.columns if c not in set(keep)]
            log.info(f"应用因子筛选：保留 {len(keep)}/{feat.shape[1]} 个因子 -> {keep}")
            if dropped:
                log.info(f"  丢弃 {len(dropped)} 个因子：{dropped}")
            feat = feat[keep]
        else:
            log.warning("因子筛选结果为空，回退使用原始全部因子。")

    # 特征列 = 选中的因子，且排除 NFWD*（前向收益是标签，不能当特征，否则泄漏）
    feat_cols = [c for c in feat.columns if not str(c).startswith("NFWD")]
    if not feat_cols:
        feat_cols = list(feat.columns)
    log.info(f"特征列（已排除 NFWD*）: {len(feat_cols)} 个因子 -> {feat_cols}")

    # 只需要各 segment 覆盖到的区间（其余年份不参与训练/回测，省内存）
    data_start = min(s[0] for s in segments.values())
    data_end = max(s[1] for s in segments.values())
    feat = _slice_by_time(feat, data_start, data_end)

    # 守卫：因子必须覆盖到「测试区间终点」，否则评估期无特征 -> 预测退化为常数
    feat_max = feat.index.get_level_values(0).max()
    test_end = pd.Timestamp(segments["test"][1])
    if feat_max < test_end:
        raise ValueError(
            f"因子数据只覆盖到 {feat_max.date()}，但测试区间终点为 {test_end.date()}。"
            f"评估期没有因子特征，预测会退化为常数（valid/test 的 IC 全为 0 或 nan）。\n"
            f"请先用 run_ic.py 把因子生成到 {segments['test'][1]} 之后再跑本流程"
            f"（检查 run_ic.py 的 END 参数，应 >= {segments['test'][1]}）。"
        )

    # 标签：优先用因子数据里的 NFWD1（前 1 日收益率）；无则回退 D.features(label_expr)。
    # 注意：NFWD1 不在 metrics 里，因子筛选(feat=feat[keep])会把它丢掉，
    # 所以必须用筛选前保留的全量副本 nfwd1_all，否则会误判为“无 NFWD1”而回退到 D.features。
    if nfwd1_all is not None:
        label_raw = nfwd1_all.reindex(feat.index).to_frame("LABEL0")
        log.info("标签来源：因子数据中的 NFWD1 列（前 1 日收益率）")
    else:
        lab = D.features(
            D.instruments(market), [label_expr],
            start_time=data_start, end_time=data_end, freq="day",
        )
        lab.columns = ["LABEL0"]
        lab = _as_datetime_instrument(lab)
        label_raw = lab
        log.info(f"标签来源：D.features 表达式 {label_expr}（因子数据中无 NFWD1）")

    # 类别变量 INDUSTRY：字符串，不能直接进数值 TCN。
    # one-hot 编码成若干数值列，作为额外通道（行业在窗口内随时间不变 -> 常数上下文）。
    if industry_mode != "none" and industry_all is not None:
        ind = industry_all.reindex(feat.index).fillna("UNKNOWN")
        if industry_mode == "onehot":
            ind_dummies = pd.get_dummies(ind, prefix="IND").astype("float32")
            feat = pd.concat([feat, ind_dummies], axis=1)
            feat_cols = feat_cols + list(ind_dummies.columns)
            log.info(
                f"INDUSTRY 以 one-hot 进入模型：{ind_dummies.shape[1]} 个行业虚拟列"
                f"（动态窗口不摊平复制，仅作额外通道；d_feat=通道数={len(feat_cols)}）"
            )
        # 未来若要做 embedding 版本，可在此分支改为保留行业 id 并改造 TCN 模型。

    label_raw = label_raw.reindex(feat.index)
    return feat, label_raw, len(feat_cols)


# def build_dataset(pkl_path: str, segments: dict, market: str = MARKET,
#                   label_expr: str = LABEL_EXPR,
#                   metrics_csv: str = METRICS_CSV, rank_ic_min: float = 0.02,
#                   topk: int = 4, factor_filter: bool = True,
#                   n_days: int = 20, industry_mode: str = "onehot") -> tuple:
#     """用中性化因子 pkl + 标签构建「时序」TCN 用的 DatasetH（qlib 自带 TCN 路径）。

#     特征先经 build_raw_features 取原始长表，再 _build_time_window 摊平为 (N, F*n_days)；
#     qlib TCN 内部 reshape(N, F, n_days) 卷积在时间轴上（真正的时序 TCN）。
#     **注意**：此摊平会把窗口物理复制成 (F+K)*n_days 列，内存 ~ N*(F+K)*n_days；
#     大 n_days 在小内存机器上会 OOM，可改用动态窗口路径（tcn_dynamic，--tcn-backend dynamic）。

#     返回 (dataset, n_feat)，n_feat = 通道数（= 选中因子数 + 行业 one-hot 列数，即 TCN 的 d_feat）。
#     """
#     feat, label_raw, n_feat = build_raw_features(
#         pkl_path, segments, market, label_expr, metrics_csv,
#         rank_ic_min, topk, factor_filter, industry_mode,
#     )
#     # 把原始长表 (T,N,F) 摊平为 (N, F*n_days) 每行窗口：qlib TCN 内部 reshape(N,F,n_days)
#     windowed = _build_time_window(feat, n_days)
#     label_df = label_raw.reindex(windowed.index)
#     data_start = min(s[0] for s in segments.values())
#     data_end = max(s[1] for s in segments.values())
#     log.info(
#         f"中性化因子+行业 {feat.shape}（"
#         f"{feat.index.get_level_values(0).min().date()} ~ "
#         f"{feat.index.get_level_values(0).max().date()}，{len(feat.columns)} 个通道(因子+行业onehot)）"
#     )
#     log.info(
#         f"时序窗口特征 {windowed.shape}（每只股票回看 {n_days} 天，"
#         f"通道数(因子)={len(feat.columns)}，总宽={windowed.shape[1]}）；标签 {label_df.shape}"
#     )

#     tag = f"{data_start}_{data_end}_w{n_days}_ind{industry_mode}"
#     loader = StaticDataLoader(
#         config={
#             "feature": _materialize(windowed, f"tcn_window_{tag}.pkl"),
#             "label": _materialize(label_df, f"label_{tag}.pkl"),
#         }
#     )
#     del feat, windowed, label_df
#     handler = DataHandlerLP(
#         instruments=None,
#         start_time=None,
#         end_time=None,
#         data_loader=loader,
#         # 因子已中性化：TCN 对特征 NaN 不鲁棒，故 learn/infer 两侧对「特征」做零值填充；
#         # 标签侧：learn 侧先 Fillna 不影响（随后 DropnaLabel 丢掉空标签样本）、再 CSZScoreNorm。
#         infer_processors=[Fillna(fields_group="feature", fill_value=0.0)],
#         learn_processors=[
#             Fillna(fields_group="feature", fill_value=0.0),
#             DropnaLabel(),
#             CSZScoreNorm(fields_group="label"),
#         ],
#     )
#     dataset = DatasetH(handler=handler, segments=segments)
#     for k, seg in segments.items():
#         log.info(f"  segment {k}: {seg[0]} ~ {seg[1]}")
#     return dataset, n_feat

def build_dataset(
    pkl_path: str,
    segments: dict,
    market: str = MARKET,
    label_expr: str = LABEL_EXPR,
    metrics_csv: str = METRICS_CSV,
    rank_ic_min: float = 0.02,
    topk: int = 4,
    factor_filter: bool = True,
    n_days: int = 20,
    industry_mode: str = "onehot",
) -> tuple:
    """用中性化因子 pkl + 标签构建时序 TCN DatasetH。

    返回
    ----
    dataset : DatasetH
    n_feat : int
        TCN 输入通道数 = 因子数 + 行业 one-hot 数
    """

    # ---------------------------------------------------------
    # 1. 构建原始长表
    # ---------------------------------------------------------
    feat, label_raw, n_feat = build_raw_features(
        pkl_path=pkl_path,
        segments=segments,
        market=market,
        label_expr=label_expr,
        metrics_csv=metrics_csv,
        rank_ic_min=rank_ic_min,
        topk=topk,
        factor_filter=factor_filter,
        industry_mode=industry_mode,
    )

    # ---------------------------------------------------------
    # 2. 时间范围
    # ---------------------------------------------------------
    data_start = min(s[0] for s in segments.values())
    data_end = max(s[1] for s in segments.values())

    # ---------------------------------------------------------
    # 3. 构造时间窗口
    #
    # feat:
    #   (datetime, instrument) × C
    #
    # windowed:
    #   (datetime, instrument) × (C * n_days)
    #
    # TCN 内部：
    #   (batch, C*n_days)
    #       ↓ reshape
    #   (batch, C, n_days)
    # ---------------------------------------------------------
    windowed = _build_time_window(feat, n_days)

    label_df = label_raw.reindex(windowed.index)

    log.info(
        f"中性化因子+行业 {feat.shape}（"
        f"{feat.index.get_level_values(0).min().date()} ~ "
        f"{feat.index.get_level_values(0).max().date()}，"
        f"{len(feat.columns)} 个通道(因子+行业onehot)）"
    )

    log.info(
        f"时序窗口特征 {windowed.shape}（"
        f"每只股票回看 {n_days} 天，"
        f"通道数={n_feat}，"
        f"总宽={windowed.shape[1]}）；"
        f"标签 {label_df.shape}"
    )

    # ---------------------------------------------------------
    # 4. 构造缓存 tag
    # ---------------------------------------------------------
    tag = (
        f"{data_start}_{data_end}"
        f"_w{n_days}"
        f"_ind{industry_mode}"
    )

    # ---------------------------------------------------------
    # 5. StaticDataLoader
    # ---------------------------------------------------------
    loader = StaticDataLoader(
        config={
            "feature": _materialize(
                windowed,
                f"tcn_window_{tag}.pkl",
            ),
            "label": _materialize(
                label_df,
                f"label_{tag}.pkl",
            ),
        }
    )

    # ---------------------------------------------------------
    # 6. DataHandler
    # ---------------------------------------------------------
    handler = DataHandlerLP(
        instruments=None,
        start_time=None,
        end_time=None,
        data_loader=loader,

        # inference/test:
        # 特征 NaN → 0
        infer_processors=[
            Fillna(
                fields_group="feature",
                fill_value=0.0,
            )
        ],

        # train:
        # 特征 NaN → 0
        # 标签 NaN → 删除
        # 标签截面标准化
        learn_processors=[
            Fillna(
                fields_group="feature",
                fill_value=0.0,
            ),
            DropnaLabel(),
            # CSZScoreNorm(
            #     fields_group="label"
            # ),
        ],
    )

    # ---------------------------------------------------------
    # 7. DatasetH
    # ---------------------------------------------------------
    dataset = DatasetH(
        handler=handler,
        segments=segments,
    )

    for k, seg in segments.items():
        log.info(
            f"  segment {k}: "
            f"{seg[0]} ~ {seg[1]}"
        )

    # 重要：
    # 直接返回 build_raw_features 已经计算好的 n_feat
    return dataset, n_feat
# ---------------------------------------------------------------------------
# 回测配置
# ---------------------------------------------------------------------------
def build_port_analysis_config(model, dataset, strategy="tilt",
                               topk=TOPK, n_drop=N_DROP,
                               cost_rate=COST_RATE, bench=BENCHMARK,
                               tilt_alpha=0.1, tilt_base="auto", tilt_topk=None,
                               risk_degree=0.95, weight_file=None,
                               weight_dump_path=None,
                               start_time=SEGMENTS["test"][0], end_time=SEGMENTS["test"][1]):
    """构建回测配置。

    strategy="tilt"（默认）：用 BenchmarkTiltStrategy，在 hs300 基准权重基础上
        按 pred 做乘性倾斜（增强指数），不再使用 TopkDropoutStrategy 的固定篮子。
    strategy="topk"：保留原 TopkDropoutStrategy，仅用于与 tilt 做对照。
    strategy="hold"：用 WeightHoldStrategy，直接按权重表（weights_day.txt）被动持仓，
        不做任何 pred 倾斜，用作「复制指数」基准对照（验证权重表本身填得对、
        以及 tilt 相对纯复制到底有没有超额）。
    """
    if strategy == "tilt":
        strategy_cfg = {
            "class": "BenchmarkTiltStrategy",
            # 运行脚本所在目录在 sys.path 上，故可直接按文件名 import
            "module_path": "benchmark_tilt_strategy",
            "kwargs": {
                # PortAnaRecord 会把 <PRED> 换成 SignalRecord 落盘的 pred.pkl
                "signal": "<PRED>",
                "alpha": tilt_alpha,
                "benchmark": "csi300",
                "base": tilt_base,
                "risk_degree": risk_degree,
                "topk_limit": tilt_topk,
                "weight_file": weight_file,
            },
        }
        log.info(
            f"回测策略: BenchmarkTiltStrategy(alpha={tilt_alpha}, base={tilt_base}, "
            f"topk_limit={tilt_topk}, risk_degree={risk_degree})"
        )
    elif strategy == "hold":
        strategy_cfg = {
            "class": "WeightHoldStrategy",
            "module_path": "weight_hold_strategy",
            "kwargs": {
                # 本策略不使用 pred，但保留 <PRED> 以兼容框架信号注入接口
                "signal": "<PRED>",
                "benchmark": "csi300",
                "base": tilt_base if tilt_base != "equal" else "auto",
                "risk_degree": risk_degree,
                "weight_file": weight_file,
                # 把策略实际使用的 as-of 持仓权重落盘（诊断用）
                "dump_path": weight_dump_path,
                "rebalance": "quarterly"
            },
        }
        log.info(
            f"回测策略: WeightHoldStrategy(base={tilt_base}, "
            f"risk_degree={risk_degree}) —— 直接按权重表被动持仓"
        )
    elif strategy == "topk":
        strategy_cfg = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": topk,
                "n_drop": n_drop,
            },
        }
        log.info(f"回测策略: TopkDropoutStrategy(topk={topk}, n_drop={n_drop})")
    else:
        raise ValueError(f"未知 strategy={strategy!r}（可选 'tilt' / 'topk' / 'hold'）")

    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": strategy_cfg,
        "backtest": {
            "start_time": start_time,
            "end_time": end_time,
            "account": ACCOUNT,
            "benchmark": bench,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": cost_rate / 2,      # 双边成本对半拆到买卖两侧
                "close_cost": cost_rate / 2,
                "min_cost": 5,
            },
        },
    }


def report_turnover_cost(report: pd.DataFrame, cost_rate: float):
    """打印实际换手率与成本拖累，并给出含成本 / 不含成本的风险指标。

    口径提醒：qlib 的 risk_analysis(freq="day") 里 mean / std 是【日度】的，
    annualized_return 才是年化收益，information_ratio 已按 sqrt(252) 年化。
    """
    turnover = report["turnover"].dropna()          # 双边成交额 / 账户价值
    cost = report["cost"].dropna()
    n_days = len(turnover)

    # --- 表 1：换手 / 成本口径 ---
    df_summary = pd.DataFrame(
        [
            ("样本交易日数", n_days, "日"),
            ("双边成本", cost_rate, ""),
            ("平均双边换手率", turnover.mean(), "/日"),
            ("平均双边换手率(年化)", turnover.mean() * 252, "倍"),
            ("平均单边换手率", turnover.mean() / 2, "/日"),
            ("平均日成本", cost.mean(), "/日"),
            ("年化成本拖累", cost.mean() * 252, ""),
        ],
        columns=["指标", "数值", "单位"],
    )

    # --- 表 2：含 / 不含成本的组合表现 ---
    perf_rows = []
    for name, series in [
        ("不含成本 return", report["return"]),
        ("含成本 return-cost", report["return"] - report["cost"]),
        ("超额(不含成本)", report["return"] - report["bench"]),
        ("超额(含成本)", report["return"] - report["bench"] - report["cost"]),
    ]:
        ra = risk_analysis(series.dropna(), freq="day")["risk"]
        perf_rows.append(
            {
                "组合": name,
                "年化收益": ra["annualized_return"],
                "日波动": ra["std"],
                "年化波动": ra["std"] * np.sqrt(252),
                "信息比": ra["information_ratio"],
                "最大回撤": ra["max_drawdown"],
            }
        )
    df_perf = pd.DataFrame(perf_rows)

    # --- 打印（保持原格式便于肉眼查看）---
    print("\n" + "=" * 78)
    print(f"回测换手与成本（样本 {n_days} 个交易日，双边成本 {cost_rate:.4f}）")
    print(f"  平均双边换手率 : {turnover.mean():.2%}/日 -> 年化 {turnover.mean() * 252:.1f} 倍")
    print(f"  平均单边换手率 : {turnover.mean() / 2:.2%}/日")
    print(f"  平均日成本     : {cost.mean():.4%}/日 -> 年化成本拖累 ~{cost.mean() * 252:.2%}")
    print("  组合表现（年化收益；波动为日度口径，括号内为年化）：")
    for _, r in df_perf.iterrows():
        print(
            f"    {r['组合']:<18}: 年化收益={r['年化收益']:+.2%} "
            f"日波动={r['日波动']:.2%}(年化 {r['年化波动']:.2%}) "
            f"信息比={r['信息比']:+.2f} 最大回撤={r['最大回撤']:.2%}"
        )
    print("=" * 78 + "\n")

    return {"summary": df_summary, "performance": df_perf}


def save_result_figures(recorder, report: pd.DataFrame, importance: pd.Series,
                        args, out_dir: str):
    """把这次 run 的关键结果画成图，存到 out_dir（方便肉眼查看）。

    包含：
      1) IC / Rank IC 逐日时间序列（含均值线）
      2) 多空 / 多头平均累计收益
      3) 回测净值：组合 vs 基准，以及含/不含成本的超额净值
      4) 逐日双边换手率与成本
      5) 特征重要性 Top 20（TCN 无 gain 类重要性，importance 为 None 时跳过）

    IC / 多空等来自 SigAnaRecord 落盘的 sig_analysis/*.pkl；
    回测来自 PortAnaRecord 落盘的 portfolio_analysis/report_normal_1day.pkl。
    每个子图独立 try，单张失败不影响其余。
    """
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"开始生成结果图，输出目录：{out_dir}")

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  保存图: {path}")
        return path

    # ---- 1) IC / Rank IC 逐日时间序列 ----
    try:
        ic = recorder.load_object("sig_analysis/ic.pkl")
        ric = recorder.load_object("sig_analysis/ric.pkl")
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        axes[0].plot(ic.index, ic.values, lw=0.7, color="#1f77b4", label="IC")
        axes[0].axhline(ic.mean(), color="red", ls="--", lw=1,
                        label=f"mean={ic.mean():.4f}")
        axes[0].set_title(
            f"Daily IC  (mean={ic.mean():+.4f}, ICIR={ic.mean()/ic.std():.3f})")
        axes[0].legend(loc="best"); axes[0].grid(alpha=0.3)
        axes[1].plot(ric.index, ric.values, lw=0.7, color="#2ca02c", label="Rank IC")
        axes[1].axhline(ric.mean(), color="red", ls="--", lw=1,
                        label=f"mean={ric.mean():.4f}")
        axes[1].set_title(
            f"Daily Rank IC  (mean={ric.mean():+.4f}, RankICIR={ric.mean()/ric.std():.3f})")
        axes[1].legend(loc="best"); axes[1].grid(alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.tight_layout()
        _save(fig, "fig1_ic_rankic.png")
    except Exception as e:  # noqa: BLE001
        log.warning(f"IC 图生成失败：{e}")

    # ---- 2) 多空 / 多头平均累计收益 ----
    try:
        ls = recorder.load_object("sig_analysis/long_short_r.pkl")
        la = recorder.load_object("sig_analysis/long_avg_r.pkl")
        ls_nav = (1 + ls.fillna(0)).cumprod()
        la_nav = (1 + la.fillna(0)).cumprod()
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(ls_nav.index, ls_nav.values, lw=1.0, color="#1f77b4",
                label=f"Long-Short (ann~{ls.mean()*252:+.1%})")
        ax.plot(la_nav.index, la_nav.values, lw=1.0, color="#ff7f0e",
                label=f"Long-Avg (ann~{la.mean()*252:+.1%})")
        ax.set_title("Long-Short / Long-Avg Cumulative Return")
        ax.legend(loc="best"); ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.tight_layout()
        _save(fig, "fig2_long_short_return.png")
    except Exception as e:  # noqa: BLE001
        log.warning(f"多空收益图生成失败：{e}")

    # ---- 3) 回测净值：组合 vs 基准 + 含/不含成本超额 ----
    try:
        r = report["return"].dropna()
        b = report["bench"].reindex(r.index).fillna(0)
        c = report["cost"].reindex(r.index).fillna(0)
        nav_strat = (1 + r).cumprod()
        nav_bench = (1 + b).cumprod()
        nav_strat_cost = (1 + r - c ).cumprod()
        # qlib 回测中 strategy 首日 return 常为 0，而 bench 首日为真实基准收益，
        # 直接 cumprod 会让基准净值起点≠1；统一以首日净值为基准归一到 1 再比较
        nav_strat = nav_strat / nav_strat.iloc[0]
        nav_bench = nav_bench / nav_bench.iloc[0]
        nav_strat_cost = nav_strat_cost / nav_strat_cost.iloc[0]
        # 超额净值用归一化后的比值，起点自然为 1
        nav_excess = nav_strat / nav_bench
        nav_excess_cost = nav_strat_cost / nav_bench
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        axes[0].plot(nav_strat.index, nav_strat.values, lw=1.0, color="#1f77b4",
                     label="Strategy (o/ cost)")
        axes[0].plot(nav_strat_cost.index, nav_strat_cost.values, lw=1.0, color="#1f77b4",
                label="Strategy (w/ cost)")
        axes[0].plot(nav_bench.index, nav_bench.values, lw=1.0, color="#7f7f7f",
                     label="Benchmark")
        axes[0].set_title("Backtest Net Value: Strategy vs Benchmark")
        axes[0].legend(loc="best"); axes[0].grid(alpha=0.3)
        axes[1].plot(nav_excess.index, nav_excess.values, lw=1.0, color="#2ca02c",
                     label="Excess (ex-cost)")
        axes[1].plot(nav_excess_cost.index, nav_excess_cost.values, lw=1.0,
                     color="#d62728", label="Excess (incl cost)")
        axes[1].set_title("Excess Net Value (w/ and w/o transaction cost)")
        axes[1].legend(loc="best"); axes[1].grid(alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.tight_layout()
        _save(fig, "fig3_backtest_nav.png")
    except Exception as e:  # noqa: BLE001
        log.warning(f"回测净值图生成失败：{e}")

    # ---- 4) 逐日双边换手率与成本 ----
    try:
        turn = report["turnover"].dropna()
        cost = report["cost"].rename("cost").reindex(turn.index).fillna(0)
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        axes[0].plot(turn.index, turn.values, lw=0.7, color="#9467bd")
        axes[0].axhline(turn.mean(), color="red", ls="--", lw=1,
                        label=f"mean={turn.mean():.2%}/day")
        axes[0].set_title("Daily Turnover (two-sided)")
        axes[0].legend(loc="best"); axes[0].grid(alpha=0.3)
        axes[1].plot(cost.index, cost.values, lw=0.7, color="#8c564b")
        axes[1].axhline(cost.mean(), color="red", ls="--", lw=1,
                        label=f"mean={cost.mean():.4%}/day")
        axes[1].set_title("Daily Transaction Cost")
        axes[1].legend(loc="best"); axes[1].grid(alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.tight_layout()
        _save(fig, "fig4_turnover_cost.png")
    except Exception as e:  # noqa: BLE001
        log.warning(f"换手率/成本图生成失败：{e}")

    # ---- 5) 特征重要性 Top 20（TCN 无 gain 类重要性，跳过）----
    if importance is None:
        log.info("TCN 无可解析的 gain 特征重要性，跳过 fig5。")
    else:
        try:
            top = importance.head(20)
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(range(len(top))[::-1], top.values, color="#1f77b4")
            ax.set_yticks(range(len(top))[::-1])
            ax.set_yticklabels(top.index, fontsize=8)
            ax.set_xlabel("gain")
            ax.set_title("Feature Importance (Top 20, gain)")
            ax.grid(alpha=0.3, axis="x")
            fig.tight_layout()
            _save(fig, "fig5_feature_importance.png")
        except Exception as e:  # noqa: BLE001
            log.warning(f"特征重要性图生成失败：{e}")

    # ---- 6) 按预测 score 分位分组回测：各组累计收益 + 头尾多空曲线 ----
    try:
        pred = recorder.load_object("pred.pkl")
        label = recorder.load_object("label.pkl")
        if pred is None or label is None:
            raise ValueError("pred.pkl / label.pkl 缺失")
        score = pred.iloc[:, 0]
        lab = label.iloc[:, 0]
        df = pd.concat([score.rename("score"), lab.rename("ret")], axis=1).dropna()
        # 只取测试（回测）区间，口径与 PortAnaRecord 一致
        ts, te = [s.strip() for s in args.test.split(",")]
        dts = df.index.get_level_values("datetime")
        df = df[(dts >= pd.Timestamp(ts)) & (dts <= pd.Timestamp(te))]
        n_groups = 5
        # 逐日按 score 分位（0=最低分，n-1=最高分）分组：
        # 用 rank(pct) 再 *n_groups 取整，比 groupby.apply(qcut) 更稳，
        # 不会因 apply 把 datetime 键多叠一层索引而报 nlevels 错
        pct = df.groupby(level="datetime")["score"].rank(pct=True)
        df["grp"] = (np.floor(pct * n_groups).astype(int)).clip(upper=n_groups - 1)
        # 各组每日平均前向收益 -> 累计净值
        grp_ret = df.groupby(["datetime", "grp"])["ret"].mean().unstack("grp")
        nav = (1 + grp_ret.fillna(0)).cumprod()
        # 多空曲线：与 qlib calc_long_short_return 同口径（前20%−后20%，再 /2）
        lo, hi = nav.columns.min(), nav.columns.max()
        ls_spread = (grp_ret[hi] - grp_ret[lo]) / 2
        ls_nav = (1 + ls_spread.fillna(0)).cumprod().rename("long_short(G5-G1)")
        ls_rel_wealth = (nav[hi] / nav[lo]).rename("rel_wealth(G5 NAV / G1 NAV)")
        avg_ret = df.groupby(level="datetime")["ret"].mean()
        g5_excess_nav = (1 + (grp_ret[hi] - avg_ret).fillna(0)).cumprod().rename("G5 vs avg")

        fig, ax = plt.subplots(figsize=(15, 6))
        cmap = plt.get_cmap("viridis", len(nav.columns))
        for i, g in enumerate(sorted(nav.columns)):
            ax.plot(nav.index, nav[g].values, lw=1.0, color=cmap(i),
                    label=f"G{g+1}({'low' if g == lo else 'high' if g == hi else 'mid'})")
        ax.plot(ls_nav.index, ls_nav.values, lw=2.0, color="red",
                label="Long-Short return (G5-G1)/2")
        ax.plot(ls_rel_wealth.index, ls_rel_wealth.values, lw=1.2, color="red",
                ls="--", alpha=0.7, label="Rel. wealth G5/G1 (unbounded)")
        ax.plot(g5_excess_nav.index, g5_excess_nav.values, lw=1.5, color="black",
                label="G5 excess over avg")
        ax.set_title(
            f"Score Quantile Group Backtest (n={n_groups}, test {ts}~{te})\n"
            f"—— score 越高，累计收益应越高（单调性验证）"
        )
        ax.legend(loc="best", ncol=3, fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.tight_layout()
        _save(fig, "fig6_score_group_backtest.png")
    except Exception as e:  # noqa: BLE001
        log.warning(f"score 分组回测图生成失败：{e}")

    log.info(f"结果图已保存到：{out_dir}")


def _seg(text: str):
    s, e = [x.strip() for x in text.split(",")]
    return (s, e)


def parse_args():
    p = argparse.ArgumentParser("qlib TCN workflow on neutralized Alpha158 factors")
    p.add_argument("--pkl", default=NEUTRAL_PKL, help="中性化因子 pkl（run_ic.py 产物）")
    p.add_argument("--train", default="%s,%s" % SEGMENTS["train"], help="训练区间 start,end")
    p.add_argument("--valid", default="%s,%s" % SEGMENTS["valid"], help="验证区间 start,end")
    p.add_argument("--test", default="%s,%s" % SEGMENTS["test"], help="测试 / 回测区间 start,end")
    p.add_argument("--cost", type=float, default=COST_RATE, help="双边交易成本（默认 0.002）")
    # ---- TCN 超参 ----
    p.add_argument("--rounds", type=int, default=10, help="最大训练 epoch 数（n_epochs）")
    p.add_argument("--early-stop", type=int, default=20, help="早停耐心（验证集 MSE 多少轮不改善就停）")
    p.add_argument("--n-days", type=int, default=5,
                   help="时序 TCN 的回看窗口长度（天）。特征 = 过去 n_days 天的 F 个因子拼成 "
                        "(N, F*n_days)，qlib TCN 内部 reshape(N, F, n_days) 卷积在时间轴上；"
                        "d_feat 自动取因子通道数 F，无需手动设置。")
    p.add_argument("--industry", choices=["none", "onehot"], default="none",
                   help="类别变量 INDUSTRY 的进入方式：onehot=独热编码为额外通道"
                        "（行业在回看窗口内随时间不变，充当常数行业上下文，d_feat 自动增加；"
                        "none=不入模（默认行业信息已在因子中性化时剔除，可对比 IC 看是否有增益）。")
    p.add_argument("--tcn-backend", choices=["qlib", "dynamic"], default="qlib",
                   help="TCN 训练后端：qlib=原 qlib 自带 TCN（把窗口摊平为 (N,F*n_days)，"
                        "大 n_days 会 OOM）；dynamic=动态窗口路径（保持原始长表、按股票切片窗口，"
                        "内存仅 ~N*(F+K)，n_days=100 也能在小内存机器跑，pred.pkl/label.pkl 回测兼容）。")
    p.add_argument("--n-chans", type=int, default=TCN_N_CHANS, help="TCN 每层的通道数")
    p.add_argument("--kernel-size", type=int, default=TCN_KERNEL_SIZE, help="TCN 卷积核大小")
    p.add_argument("--num-layers", type=int, default=TCN_NUM_LAYERS, help="TCN 残差块（膨胀卷积层数）")
    p.add_argument("--dropout", type=float, default=TCN_DROPOUT, help="TCN dropout")
    p.add_argument("--lr", type=float, default=TCN_LR, help="学习率")
    p.add_argument("--batch-size", type=int, default=TCN_BATCH_SIZE, help="batch 大小")
    p.add_argument("--gpu", type=int, default=TCN_GPU, help="GPU id（>=0 且 CUDA 可用才用 GPU，否则回退 CPU）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    # ---- 回测 ----
    p.add_argument("--topk", type=int, default=TOPK)
    p.add_argument("--n-drop", type=int, default=N_DROP)
    p.add_argument("--strategy", default="topk", choices=["tilt", "topk", "hold"],
                   help="回测策略：tilt=在基准权重上按 pred 倾斜（默认，不用 TopkDropoutStrategy）；"
                        "topk=原 TopkDropoutStrategy（仅对照）；"
                        "hold=直接按权重表被动持仓（复制指数基准对照）")
    p.add_argument("--tilt-alpha", type=float, default=0.1,
                   help="基准倾斜强度 alpha：w_i = base_w_i * exp(alpha * zscore(pred_i))")
    p.add_argument("--tilt-base", default="auto", choices=["auto", "equal", "index"],
                   help="基准权重来源：auto=有真实指数权重则用否则等权；equal=强制等权；"
                        "index=强制真实指数权重 $csi300_weight")
    p.add_argument("--tilt-topk", type=int, default=None,
                   help="倾斜后最多持有前 N 只票（None=不限制，持有全部基准成分）")
    p.add_argument("--risk-degree", type=float, default=0.95,
                   help="风险预算（投到股票的比例），默认 0.95")
    p.add_argument("--hs300-weight-file", default=DEFAULT_WEIGHT_FILE,
                   help="hs300 真实权重文件路径（由 fetch_hs300_weights.py 生成）；"
                        "base=auto/index 时作为基准权重来源")
    p.add_argument("--experiment", default="workflow_neutral_tcn")
    p.add_argument("--mlflow-uri", default=MLFLOW_URI)
    p.add_argument("--metrics", default=METRICS_CSV,
                   help="因子指标表 CSV（用于按类别筛选，默认 alpha158_factor_metrics.csv）")
    p.add_argument("--rank-ic-min", type=float, default=0.015,
                   help="筛选阈值：每类因子 Rank IC 需大于该值（默认 0.015）")
    p.add_argument("--factor-topk", type=int, default=6,
                   help="每类保留 Rank IC 最高的前 N 个因子（默认 6）")
    p.add_argument("--no-factor-filter", action="store_true",
                   help="关闭按类别因子筛选（使用 pkl 中的全部因子）")
    p.add_argument("--check-pkl", action="store_true",
                   help="只检查中性化因子 pkl 的 NaN/常数/近常数列与高冗余因子对，检查后退出（不训练）")
    p.add_argument("--skip-train", action="store_true",
                   help="跳过 model.fit 与 SignalRecord：生成常数假 pred.pkl 仅用于驱动回测框架。"
                        "主要配合 --strategy hold（纯指数复制，不浪费训练算力）；"
                        "配合 tilt 时 pred 为常数 -> 退化为纯基准权重；topk 需要真实信号，不允许。")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.check_pkl:
        check_pkl_degeneracy(args.pkl)
        return
    segments = {
        "train": _seg(args.train),
        "valid": _seg(args.valid),
        "test": _seg(args.test),
    }

    import qlib.data.dataset as _qds

    log.info("=" * 78)
    log.info(f"qlib 框架源码: {os.path.dirname(os.path.dirname(os.path.dirname(_qds.__file__)))}")
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    # ---- 1) 数据 / 模型：按后端分两条路径 ----
    dynamic = (args.tcn_backend == "dynamic")
    if dynamic:
        # 动态窗口路径：保持原始长表（不摊平），训练时按股票切片窗口 -> 省内存
        from tcn_dynamic import train_dynamic
        feat_raw, label_raw, n_feat = build_raw_features(
            args.pkl, segments, MARKET, LABEL_EXPR, METRICS_CSV,
            rank_ic_min=args.rank_ic_min, topk=args.factor_topk,
            factor_filter=not args.no_factor_filter, industry_mode=args.industry,
        )
        dataset, model = None, None  # qlib DatasetH / QlibTCN 不使用
        log.info(
            f"模型: 动态窗口 CausalTCN(通道C={n_feat}, 回看 L={args.n_days} 天, "
            f"n_chans={args.n_chans}, kernel_size={args.kernel_size}, num_layers={args.num_layers}, "
            f"dropout={args.dropout}, lr={args.lr}, batch_size={args.batch_size}) | 训练 "
            f"{segments['train'][0]} ~ {segments['train'][1]}（原始长表不复制窗口，内存 ~N*C）"
        )
    else:
        dataset, n_feat = build_dataset(
            args.pkl, segments, MARKET,
            metrics_csv=args.metrics, rank_ic_min=args.rank_ic_min,
            topk=args.factor_topk, factor_filter=not args.no_factor_filter,
            n_days=args.n_days, industry_mode=args.industry,
        )

        # ---- 2) 模型：qlib 自带的「时序」TCN（pytorch_tcn.TCN）----
        # d_feat 必须等于因子通道数 F，时间窗口长度 n_days 由特征总宽 / F 得到，
        # qlib TCN 内部 x.reshape(N, F, n_days) 卷积在时间轴上（真正的时序建模）。
        model = QlibTCN(
            d_feat=n_feat,
            n_chans=args.n_chans,
            kernel_size=args.kernel_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            n_epochs=args.rounds,
            lr=args.lr,
            batch_size=args.batch_size,
            early_stop=args.early_stop,
            loss="mse",
            GPU=args.gpu,
            seed=args.seed,
        )
        log.info(
            f"模型: qlib 时序 TCN(d_feat(因子通道)={n_feat}, 时间窗口 n_days={args.n_days}, "
            f"n_chans={args.n_chans}, kernel_size={args.kernel_size}, num_layers={args.num_layers}, "
            f"dropout={args.dropout}, lr={args.lr}, batch_size={args.batch_size}) | 训练 "
            f"{segments['train'][0]} ~ {segments['train'][1]}"
        )

    port_analysis_config = build_port_analysis_config(
        model, dataset, strategy=args.strategy,
        topk=args.topk, n_drop=args.n_drop, cost_rate=args.cost,
        tilt_alpha=args.tilt_alpha, tilt_base=args.tilt_base,
        tilt_topk=args.tilt_topk, risk_degree=args.risk_degree,
        weight_file=args.hs300_weight_file,
        weight_dump_path=os.path.join(
            HERE, "results", "workflow_figs", args.experiment, "hold_weights.csv"
        ),
        start_time=segments["test"][0], end_time=segments["test"][1],
    )

    # ---- 3) 训练 -> 预测 -> 信号分析 -> 回测 ----
    with R.start(experiment_name=args.experiment, uri=args.mlflow_uri):
        R.log_params(
            **{
                "factor_source": args.pkl,
                "label": LABEL_EXPR,
                "model_type": "tcn",
                "market": MARKET,
                "factor_filter": not args.no_factor_filter,
                "metrics_csv": args.metrics,
                "rank_ic_min": args.rank_ic_min,
                "topk": args.topk,
                "strategy": args.strategy,
                "train": f"{segments['train'][0]}~{segments['train'][1]}",
                "valid": f"{segments['valid'][0]}~{segments['valid'][1]}",
                "test": f"{segments['test'][0]}~{segments['test'][1]}",
                "cost_rate": args.cost,
                "tcn.d_feat": n_feat,
                "tcn.n_days": args.n_days,
                "tcn.n_chans": args.n_chans,
                "tcn.kernel_size": args.kernel_size,
                "tcn.num_layers": args.num_layers,
                "tcn.dropout": args.dropout,
                "tcn.lr": args.lr,
                "tcn.batch_size": args.batch_size,
                "tcn.n_epochs": args.rounds,
                "tcn.early_stop": args.early_stop,
                "tcn.gpu": args.gpu,
                "tcn.seed": args.seed,
                "tilt_alpha": args.tilt_alpha,
                "tilt_base": args.tilt_base,
                "tilt_topk": args.tilt_topk,
                "risk_degree": args.risk_degree,
                "n_drop": args.n_drop,
                "skip_train": args.skip_train,
            }
        )
        recorder = R.get_recorder()

        if dynamic:
            # ---- 动态窗口路径：直接产出 pred.pkl / label.pkl（回测格式兼容）----
            from tcn_dynamic import _segment_windows
            if args.skip_train:
                if args.strategy == "topk":
                    raise SystemExit(
                        "--skip-train 仅适用于 tilt / hold 策略：topk 需要真实 pred 信号才能选股。"
                    )
                log.info(
                    "skip-train(dynamic)：用 test 段样本索引造常数假 pred.pkl 驱动回测框架"
                    "（hold 忽略信号值；tilt 退化为纯基准权重）。"
                )
                _, te_idx = _segment_windows(feat_raw, label_raw, segments["test"], args.n_days)
                dummy_pred = pd.DataFrame(0.0, index=te_idx, columns=["score"])
                dummy_label = pd.DataFrame(0.0, index=te_idx, columns=["LABEL0"])
                R.save_objects(**{"pred.pkl": dummy_pred, "label.pkl": dummy_label})
            else:
                pred_df, label_df, dyn_model = train_dynamic(
                    feat_raw, label_raw, n_feat, args, segments
                )
                model = dyn_model
                R.save_objects(**{"pred.pkl": pred_df, "label.pkl": label_df, "params.pkl": dyn_model})
                # IC / RankIC / ICIR / 多空收益
                SigAnaRecord(recorder, ana_long_short=True).generate()
            importance = None
        elif args.skip_train:
            if args.strategy == "topk":
                raise SystemExit(
                    "--skip-train 仅适用于 tilt / hold 策略：topk 需要真实 pred 信号才能选股。"
                )
            log.info(
                "skip-train：跳过 model.fit 与 SignalRecord，改用常数假 pred.pkl 驱动回测框架"
                "（hold 策略忽略信号值；tilt 退化为纯基准权重）。"
            )
            # 用数据集 test 段的索引造常数假 pred / label，覆盖回测区间，使框架依赖检查通过并正常推进
            # （PortAnaRecord.depend_cls=SignalRecord，要求 pred.pkl 与 label.pkl 都存在；hold 会忽略其值）
            feat = dataset.prepare("test", col_set="feature")
            dummy_pred = pd.DataFrame(0.0, index=feat.index, columns=["score"])
            dummy_label = pd.DataFrame(0.0, index=feat.index, columns=["LABEL0"])
            R.save_objects(**{"pred.pkl": dummy_pred, "label.pkl": dummy_label})
            importance = None
        else:
            model.fit(dataset)
            R.save_objects(**{"params.pkl": model})

            # 预测 + 标签（供 IC 分析 / 回测）
            SignalRecord(model, dataset, recorder).generate()
            # IC / RankIC / ICIR / 多空收益
            SigAnaRecord(recorder, ana_long_short=True).generate()

            # 按日截面报告：IC / ICIR / 换手率代理（TCN 无 gain 类特征重要性）
            report_tcn_perf(model, dataset)
            importance = None

        # 组合回测（同样的双边成本）—— 两种模式都跑
        PortAnaRecord(recorder, port_analysis_config, "day").generate()

        # 换手率 / 成本 / 含成本表现
        report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        tables = report_turnover_cost(report, args.cost)

        # 把汇总 / 表现表存成 CSV，方便后续对比
        tbl_dir = os.path.join(HERE, "results","workflow_figs", args.experiment)
        os.makedirs(tbl_dir, exist_ok=True)
        tables["summary"].to_csv(os.path.join(tbl_dir, "turnover_cost_summary.csv"), index=False)
        tables["performance"].to_csv(os.path.join(tbl_dir, "turnover_cost_performance.csv"), index=False)
        recorder.save_objects(**{
            "turnover_cost_summary.csv": tables["summary"].to_csv(index=False),
            "turnover_cost_performance.csv": tables["performance"].to_csv(index=False),
        })

        # 结果图：IC / 多空 / 回测净值 / 换手成本 / 特征重要性 存到文件夹
        figs_dir = os.path.join(HERE, "results", "workflow_figs", args.experiment)
        save_result_figures(recorder, report, importance, args, figs_dir)

    log.info(f"全部完成。实验：{args.experiment}（tracking uri: {args.mlflow_uri}）")


if __name__ == "__main__":
    main()
