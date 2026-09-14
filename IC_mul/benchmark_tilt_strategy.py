#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
BenchmarkTiltStrategy —— 在「基准权重」基础上，用模型预测分数 pred 做增强指数式倾斜。

与 workflow_weight.py 配套的自定义回测策略，用来替换 TopkDropoutStrategy：
不再每天固定选 top-k 个票，而是「贴近基准（默认 hs300/csi300）、按 pred 调权」。

每个调仓日：
  1) 取基准权重 base_w
       - base="auto"/"index"：优先读真实 hs300 权重文件（fetch_hs300_weights.py 落盘的
         instruments/csi300/weights_day.txt，来自 csindex 官方披露的成分股权重）；
         若文件缺失，再尝试 $<benchmark>_weight 特征（需把权重 dump 成 per-instrument 特征）；
         仍取不到则 base="auto" 回退为【等权】基准，base="index" 直接报错。
       - base="equal"：强制等权基准。
  2) 对当日 pred 做截面 z-score：z_i = (s_i - mean) / std
  3) 乘性倾斜：w_i = base_w_i * exp(alpha * z_i)，再归一化到和为 1
       —— pred 越高，相对基准超配；pred 越低，相对基准低配
  4) 返回目标权重 dict（订单生成器会按 risk_degree 自动乘以风险预算）

alpha 越大，组合越偏离基准、越像纯 alpha 组合；alpha→0 即纯复制基准。
这样既能「根据 pred 对基准权重做调整」，又不依赖 TopkDropoutStrategy 的固定篮子。
"""

import os
from typing import Any

import numpy as np
import pandas as pd

from qlib.data import D
from qlib.log import get_module_logger
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

logger = get_module_logger("benchmark_tilt", __name__)

# 真实 hs300 权重文件的默认位置（由 fetch_hs300_weights.py 生成）
DEFAULT_WEIGHT_FILE = os.path.expanduser(
    "~/.qlib/qlib_data/cn_data/instruments/csi300/weights_day.txt"
)


class BenchmarkTiltStrategy(WeightStrategyBase):
    """在基准权重上按 pred 做乘性倾斜的增强指数策略。

    参数
    ------
    signal : str / Signal
        预测分数来源，默认 "<PRED>"（SignalRecord 落盘的 pred.pkl）。
    alpha : float
        倾斜强度。w_i = base_w_i * exp(alpha * z_i)。
        典型取值 0.05~0.5；越大越偏离基准。
    benchmark : str
        基准对应的市场名（用于读 `$<benchmark>_weight` 特征），默认 "csi300"。
    base : {"auto", "index", "equal"}
        基准权重来源：
          auto  : 优先用真实 hs300 权重文件，没有再试 `$<benchmark>_weight` 特征，
                  都没有则等权（默认）
          index : 强制用真实权重（文件/特征都没有就报错）
          equal : 强制等权基准
    risk_degree : float
        风险预算（投到股票的比例），默认 0.95，与 TopkDropoutStrategy 一致。
    topk_limit : int or None
        若指定，最多持有前 topk_limit 只票（其余权重平分到入选股）；
        不指定则持有基准全部成分（仅按 pred 倾斜）。
    weight_file : str
        真实 hs300 权重文件路径（fetch_hs300_weights.py 生成），默认
        ~/.qlib/qlib_data/cn_data/instruments/csi300/weights_day.txt。
    max_dev : float
        单个标的相对基础权重的偏离上限（默认 0.5 = ±50%）。
        最终权重被约束在 [(1-max_dev)*base_i, (1+max_dev)*base_i] 内，
        即每个持仓相对基准权重最多偏离 50%。设 0 或 None 则不加约束。
    """

    def __init__(
        self,
        *,
        signal="<PRED>",
        alpha=0.1,
        benchmark="csi300",
        base="auto",
        risk_degree=0.95,
        topk_limit=None,
        weight_file=DEFAULT_WEIGHT_FILE,
        max_dev=0.5,
        **kwargs,
    ):
        super().__init__(signal=signal, risk_degree=risk_degree, **kwargs)
        self.alpha = float(alpha)
        self.benchmark = benchmark
        self.base = base
        self.topk_limit = int(topk_limit) if topk_limit else None
        self.weight_file = weight_file
        self.max_dev = float(max_dev) if max_dev is not None else None
        self._inst_all = D.instruments("all")
        self._weight_cache = None  # 缓存权重文件 (date -> Series)

    # ------------------------------------------------------------------
    def _load_weight_file(self, date):
        """从 fetch_hs300_weights.py 落盘的文件读取真实 hs300 基准权重。

        文件格式：date,instrument,weight（weight 为小数权重）。
        多期时取 <= trade date 的最近一期；没有更早的期则用最早一期（静态近似）。
        """
        if self._weight_cache is None:
            wdf = pd.read_csv(self.weight_file)
            wdf["date"] = pd.to_datetime(wdf["date"])
            self._weight_cache = {
                d: g.set_index("instrument")["weight"] for d, g in wdf.groupby("date")
            }
        dates = sorted(self._weight_cache.keys())
        cand = [d for d in dates if d <= pd.Timestamp(date)]
        pick = cand[-1] if cand else dates[0]
        w = self._weight_cache[pick]
        w = w[w > 0]
        return w if w.sum() > 0 else None

    def _base_weight(self, date):
        """返回当日基准权重（instrument 索引的 Series），取不到返回 None。

        优先级：外部权重文件（真实 hs300 权重）> $csi300_weight 特征 >
        等权（auto 时回退；index 时强制报错）。
        """
        if self.base == "equal":
            return None

        w = None
        # ① 真实 hs300 权重文件
        if self.weight_file and os.path.exists(self.weight_file):
            try:
                w = self._load_weight_file(date)
            except Exception as e:  # noqa: BLE001
                logger.warning("读取权重文件 %s 失败：%s", self.weight_file, e)
        # ② 真实指数权重特征（需先把权重 dump 成 per-instrument 特征 csi300_weight）
        if w is None and self.base in ("auto", "index"):
            try:
                wf = D.features(
                    self._inst_all, [f"${self.benchmark}_weight"],
                    start_time=date, end_time=date,
                ).squeeze()
                if isinstance(wf.index, pd.MultiIndex):
                    wf = wf.droplevel(0)
                if wf.notna().any():
                    w = wf
            except Exception as e:  # noqa: BLE001
                logger.warning("读取 $%s_weight 失败：%s", self.benchmark, e)

        if w is None:
            if self.base == "index":
                raise RuntimeError(
                    f"base='index' 但未找到真实权重（请先运行 fetch_hs300_weights.py 生成 {self.weight_file}）"
                )
            logger.warning("未取到真实基准权重，回退为等权基准。")
            return None

        w = w[w > 0]
        s = w.sum()
        return (w / s) if s > 0 else None

    # ------------------------------------------------------------------
    def _cap_deviation_from_base(self, w, base_w):
        """将权重限制在相对基础权重的 ±max_dev 内，并保证和为 1。

        采用迭代 water-filling：先把 w 截断到 [(1-max_dev)*base, (1+max_dev)*base]，
        再归一化；重复直到所有标的都落在界内。这样归一化后约束依然严格成立
        （单纯的 clip 会被后续归一化再次推破边界）。

        非成分股（base=0）被约束到 0，不会凭空持有。
        """
        max_dev = self.max_dev
        if max_dev is None or max_dev <= 0:
            return w
        base_w = base_w.reindex(w.index).fillna(0.0)
        lo = (1.0 - max_dev) * base_w
        hi = (1.0 + max_dev) * base_w
        for _ in range(50):
            w = w.clip(lower=lo, upper=hi)
            total = w.sum()
            if not (total > 0 and np.isfinite(total)):
                return base_w
            w = w / total
            inside = (w >= lo - 1e-12) & (w <= hi + 1e-12)
            if bool(inside.all()):
                break
        return w

    # ------------------------------------------------------------------
    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        # score：当日 pred。可能是 (datetime, instrument) 多索引，也可能已压成 instrument 索引
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        if isinstance(score, pd.Series) and score.index.nlevels == 2:
            dts = score.index.get_level_values(0).unique()
            score = score.xs(dts[-1], level=0)
        score = score.dropna()
        if score.empty:
            return None

        idx = score.index
        date = trade_start_time

        # ① 基准权重
        base_w = self._base_weight(date)
        if base_w is None:
            base_w = pd.Series(1.0 / len(idx), index=idx)
        else:
            base_w = base_w.reindex(idx).fillna(0.0)
            s: Any = base_w.sum()
            if s <= 0 or not np.isfinite(s):
                base_w = pd.Series(1.0 / len(idx), index=idx)
            else:
                base_w = base_w / s

        # ② 截面 z-score
        mu, sd = score.mean(), score.std()
        sd = sd if (np.isfinite(sd) and sd > 1e-8) else 1.0
        z = (score - mu) / sd

        # ③ 乘性倾斜 + 归一化 + 偏离基础权重约束
        w = base_w * np.exp(self.alpha * z)
        w = w.clip(lower=0.0)
        s = w.sum()
        if not (s > 0 and np.isfinite(s)):
            w = base_w
        else:
            w = w / s
            # 约束单个标的相对基础权重的偏离不超过 max_dev（默认 50%）：
            # 最终 w_i 落在 [(1-max_dev)*base_i, (1+max_dev)*base_i] 内
            w = self._cap_deviation_from_base(w, base_w)

        # ④ 可选：限制持仓数量
        if self.topk_limit and self.topk_limit > 0 and len(w) > self.topk_limit:
            w = w.sort_values(ascending=False)
            rest = w.iloc[self.topk_limit:].sum()
            w = w.iloc[:self.topk_limit]
            w = w + rest / len(w)

        # 返回权重（和为 1）；订单生成器会按 risk_degree 自动乘以风险预算
        return {stock: float(weight) for stock, weight in w.items() if weight > 1e-6}
