#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
WeightHoldStrategy —— 纯被动策略：直接按权重表（weights_day.txt）持仓，不做任何 pred 倾斜。

与 BenchmarkTiltStrategy 配套，用作「增强指数」的基准对照（即复制指数本身）：
每个调仓日读取权重表在当日的 as-of 快照（≤ 交易日的最近一期，向前补充），
以该快照为目标持仓，由订单生成器按 risk_degree 乘以风险预算。

典型用途
--------
- 作为 baseline：对比「按 pred 倾斜」相对「纯复制指数权重」到底赚没赚到超额。
- 验证权重表本身是否填对（持有权重表应 ≈ 沪深300 净值）。

权重来源优先级（与 BenchmarkTiltStrategy 一致）：
  base="auto"（默认）：优先读真实 hs300 权重文件，缺失则等权；
  base="index"        ：强制真实权重（文件缺失报错）；
  base="equal"        ：强制等权基准。
"""

import os

import numpy as np
import pandas as pd

from qlib.data import D
from qlib.log import get_module_logger
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

from benchmark_tilt_strategy import (
    BenchmarkTiltStrategy,
    DEFAULT_WEIGHT_FILE,
)

logger = get_module_logger("weight_hold")


class WeightHoldStrategy(BenchmarkTiltStrategy):
    """直接按权重表持仓的被动策略（不做 pred 倾斜）。

    参数
    ------
    signal : str / Signal
        与框架接口保持一致（回测时会由 <PRED> 注入），本策略不使用。
    benchmark : str
        基准市场名（用于 $<benchmark>_weight 特征回退），默认 "csi300"。
    base : {"auto", "index", "equal"}
        持仓权重来源（见模块 docstring），默认 "auto"。
    risk_degree : float
        风险预算（投到股票的比例），默认 0.95，与 TopkDropoutStrategy 一致。
    weight_file : str
        真实 hs300 权重文件路径（fetch_hs300_weights.py 生成），默认
        ~/.qlib/qlib_data/cn_data/instruments/csi300/weights_day.txt。
    dump_path : str or None
        若指定，把每个调仓日策略实际使用的 as-of 权重落盘为 CSV
        （列：trade_date,instrument,weight）。仅在权重快照相对上一期发生变化时才写，
        避免逐交易日重复刷。用于诊断「持有的到底是不是真实指数权重」。
    rebalance : {"daily", "quarterly"}
        调仓频率。默认 "quarterly"（季度调仓）：仅在进入新季度的首个交易日重算目标权重，
        其余交易日直接沿用上一次的目标权重（即持仓不动，不做再平衡）。
        "daily" 则保持原行为，每个交易日都按权重表重算并再平衡。
        季度边界以自然季度（1/4/7/10 月）为准，取每个季度的首个交易日作为调仓日。
    """

    def __init__(
        self,
        *,
        signal="<PRED>",
        benchmark="csi300",
        base="auto",
        risk_degree=0.95,
        weight_file=DEFAULT_WEIGHT_FILE,
        dump_path=None,
        rebalance="quarterly",
        **kwargs,
    ):
        # 复用父类的权重读取逻辑；tilt 专属参数（alpha/topk_limit/max_dev）在此被置为中性
        super().__init__(
            signal=signal,
            alpha=0.0,
            benchmark=benchmark,
            base=base,
            risk_degree=risk_degree,
            topk_limit=None,
            weight_file=weight_file,
            max_dev=None,
            **kwargs,
        )
        if rebalance not in ("daily", "quarterly"):
            raise ValueError(f"rebalance 仅支持 'daily'/'quarterly'，收到 {rebalance!r}")
        self.rebalance = rebalance
        self._dump_path = dump_path
        self.holding_history = {}      # date -> 权重 Series（诊断用，程序内可直接取）
        self._last_w_sig = None        # 变更检测：上一期权重指纹
        self._dump_fh = None
        self._rebalance_key = None     # 上一次调仓所属季度（如 "2024-Q1"）

    # ------------------------------------------------------------------
    def _maybe_dump_weights(self, date, w):
        """把 as-of 权重落盘（仅在快照相对上一期变化时写，避免每日重复）。"""
        if self._dump_path is None or w is None:
            return
        sig = tuple(round(float(v), 8) for v in w.reindex(sorted(w.index)).values)
        if sig == self._last_w_sig:
            return  # 权重快照未变，跳过
        self._last_w_sig = sig
        if self._dump_fh is None:
            os.makedirs(os.path.dirname(self._dump_path), exist_ok=True)
            self._dump_fh = open(self._dump_path, "w", newline="")
            self._dump_fh.write("trade_date,instrument,weight\n")
        d = pd.Timestamp(date).date()
        for inst, wt in w.items():
            self._dump_fh.write(f"{d},{inst},{wt:.10f}\n")
        self._dump_fh.flush()

    # ------------------------------------------------------------------
    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        date = trade_start_time

        # —— 季度调仓：非调仓日不交易，持仓原样持有（随行情漂移）——
        if self.rebalance == "quarterly":
            cur_key = f"{date.year}-Q{date.quarter}"
            if self._rebalance_key is not None and cur_key == self._rebalance_key:
                # 仍处于同一季度：返回 None，订单生成器会产出空订单列表，即完全不调仓
                return None

        # 权重快照（as-of）。股票池直接取「基准权重表的成分」，
        # 而不是从 pred 的索引取 —— 这样 hold 才是真正的「复制指数」，
        # 不再受某日 pred 缺失 / 覆盖不全的影响（pred 仅作兜底用）。
        w = self._base_weight(date)
        if w is None:
            # 取不到真实权重（base=equal 或权重文件缺失）：用 pred 覆盖的可交易股票池做等权兜底
            if isinstance(score, pd.DataFrame):
                score = score.iloc[:, 0]
            if isinstance(score, pd.Series) and score.index.nlevels == 2:
                score = score.xs(score.index.get_level_values(0).unique()[-1], level=0)
            idx = score.dropna().index
            if len(idx) == 0:
                return None  # 连 pred 都没有 -> 当日空仓不交易
            logger.warning("%s 未取到权重表，回退为等权基准。", date.date())
            w = pd.Series(1.0 / len(idx), index=idx)
        else:
            w = w[w > 0]
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                return None
            w = w / s

        # 记录 + 落盘实际持仓权重（诊断用）
        self.holding_history[pd.Timestamp(date)] = w.copy()
        self._maybe_dump_weights(date, w)

        # 记录本次调仓所属季度，供非调仓日判断（非调仓日返回 None = 不交易）
        self._rebalance_key = f"{pd.Timestamp(date).year}-Q{pd.Timestamp(date).quarter}"

        # 返回权重（和为 1）；订单生成器会按 risk_degree 自动乘以风险预算
        return {stock: float(weight) for stock, weight in w.items() if weight > 1e-6}
