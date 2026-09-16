#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
BenchmarkTiltStrategy —— 在「基准权重」基础上，用模型预测分数 pred 做增强指数式倾斜。

与 workflow_weight.py 配套的自定义回测策略，用来替换 TopkDropoutStrategy：
不再每天固定选 top-k 个票，而是「贴近基准（默认 hs300/csi300）、按 pred 调权」。

每个调仓日：
  1) 取基准权重 base_w（归一化到和为 1，且只保留权重 > 0 的成分）
       - base="auto"/"index"：优先读真实 hs300 权重文件（fetch_hs300_weights.py 落盘的
         instruments/csi300/weights_day.txt，来自 csindex 官方披露的成分股权重）；
         若文件缺失，再尝试 $<benchmark>_weight 特征（需把权重 dump 成 per-instrument 特征）；
         仍取不到则 base="auto" 回退为【等权】基准，base="index" 直接报错。
       - base="equal"：强制等权基准。
  2) **基准成分股才是最终 universe**：把 pred 对齐到基准成分股；
     缺失 pred 的成分股按「中性 score = 基准内均值」处理（等价于 z=0，保留基准权重），
     绝不把缺失 pred 的成分股从基准里删掉 —— 否则会引入隐性的主动暴露，
     也会让 max_dev 相对「删票后重新归一化的假基准」被伪满足。
  3) 只在最终 universe 内部做截面 z-score：z_i = (s_i - mu) / sd
     （不参与投资的股票不参与标准化）
  4) 乘性倾斜（指数平移避免 exp 溢出）：
        w_i ∝ base_w_i * exp(alpha * z_i - max_j(alpha * z_j))
     pred 越高，相对基准超配；pred 越低，相对基准低配。
  5) 可选 max_dev：把 w 欧氏投影到
        { w : sum(w)=1, (1-d)*base_i <= w_i <= (1+d)*base_i }
     （二分求 λ 的 bounded simplex projection，严格满足上下界与和为 1）
  6) 返回目标权重 dict（订单生成器会按 risk_degree 自动乘以风险预算）

alpha 越大，组合越偏离基准、越像纯 alpha 组合；alpha→0 即纯复制基准。
这样既能「根据 pred 对基准权重做调整」，又不依赖 TopkDropoutStrategy 的固定篮子。

注意
----
topk_limit 与 max_dev 在本策略中是【互斥】的：只持有 topk 只票意味着其余成分股
相对基准偏离 -100%，与 ±max_dev 的约束在数学上无可行解。指数增强建议
topk_limit=None（默认）；确需选股则设 max_dev=0/None。
"""

import os

import numpy as np
import pandas as pd

from qlib.data import D
from qlib.log import get_module_logger
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

logger = get_module_logger("benchmark_tilt", __name__)

# 真实 hs300 权重文件的默认位置（由 fetch_hs300_weights.py 生成）
DEFAULT_WEIGHT_FILE = os.path.expanduser(
    "/home/fei/workspace/qlib/me/IC_mul/data/weights_day.txt"
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
        若指定，只保留【pred 最高】的 topk_limit 只基准成分股（按 score 选，不是按
        倾斜后权重选），其余成分股权重置 0，然后在入选股上重新做 tilt。
        与 max_dev 互斥（见模块 docstring），指数增强建议保持 None。
    weight_file : str
        真实 hs300 权重文件路径（fetch_hs300_weights.py 生成），默认
        ~/.qlib/qlib_data/cn_data/instruments/csi300/weights_day.txt。
    adjust_close : bool
        是否按 close 对基准权重做漂移调整（默认 True）。权重文件给出的是
        as-of 日期的基准权重，从 as-of 到当前调仓日之间成分股价格变动会改变
        持仓权重，故按 w_i ∝ w_base_i × (close[date]/close[asof]) 重新计算
        并归一化，等价于「as-of 日买入后持有到当前日」的真实组合权重。
        设为 False 则直接使用权重文件中的原始权重（不做价格漂移）。
    max_dev : float
        单个标的相对基础权重的偏离上限（默认 0.5 = ±50%）。
        最终权重被严格约束在 [(1-max_dev)*base_i, (1+max_dev)*base_i] 内且和为 1，
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
        adjust_close=True,
        **kwargs,
    ):
        super().__init__(signal=signal, risk_degree=risk_degree, **kwargs)
        self.alpha = float(alpha)
        self.benchmark = benchmark
        self.base = base
        self.topk_limit = int(topk_limit) if topk_limit else None
        self.weight_file = weight_file
        self.max_dev = float(max_dev) if max_dev is not None else None
        self.adjust_close = bool(adjust_close)
        if self.topk_limit is not None and (self.max_dev or 0) > 0:
            # 只持有 topk 只票 => 其余成分股相对基准偏离 -100%，与 ±max_dev 无可行解
            raise ValueError(
                "BenchmarkTiltStrategy 中 topk_limit 与 max_dev 数学上冲突："
                f"topk_limit={self.topk_limit} 会把未入选成分股的权重压到 0（相对基准 -100%），"
                f"无法满足 max_dev={self.max_dev} 的 ±{self.max_dev:.0%} 偏离约束。"
                "指数增强（基准 + 连续倾斜）请设 topk_limit=None；"
                "若确实要选股，请把 max_dev 设为 0 或 None。"
            )
        self._inst_all = D.instruments("all")
        self._weight_cache = None  # 缓存权重文件 (date -> Series)
        self._close_panel = {}     # 缓存 as-of 日的 close 面板 (asof -> DataFrame[instrument, date])

    # ------------------------------------------------------------------
    def _load_weight_file(self, date):
        """从 fetch_hs300_weights.py 落盘的文件读取真实 hs300 基准权重。

        文件格式：date,instrument,weight（weight 为小数权重）。
        多期时取 <= trade date 的最近一期（as-of，避免未来数据）。
        **不向前借未来的权重**：回测日期早于权重文件第一期时返回 None
        （调用方按 base 决定回退等权还是报错），否则属于 look-ahead。
        """
        if self._weight_cache is None:
            wdf = pd.read_csv(self.weight_file)
            wdf["date"] = pd.to_datetime(wdf["date"])
            self._weight_cache = {
                d: g.set_index("instrument")["weight"] for d, g in wdf.groupby("date")
            }
        dates = sorted(self._weight_cache.keys())
        if not dates:
            return None
        cand = [d for d in dates if d <= pd.Timestamp(date)]
        if not cand:
            logger.warning(
                "权重文件 %s 最早一期为 %s，晚于交易日 %s，不使用未来权重（返回 None）。",
                self.weight_file, dates[0].date(), pd.Timestamp(date).date(),
            )
            return None
        asof = cand[-1]
        w = self._weight_cache[asof]
        w = w[w > 0]
        # 返回 (权重, 该权重对应的 as-of 日期)，供调用方按 close 做漂移调整
        return (w if w.sum() > 0 else None), asof

    # ------------------------------------------------------------------
    def _fetch_close(self, insts, start, end):
        """取 [start, end] 窗口内指定股票池的 $close，返回 DataFrame（instrument × date）。

        鲁棒性说明：
          - D.features 对单特征返回 MultiIndex(instrument, datetime)，unstack(-1)
            后得到 instrument × date 的面板（date 为列）。
          - as-of 权重日期可能落在非交易日，窗口取不到时返回空面板，由调用方按
            「≤d 最近交易日」切片兜底，不会报错。
        """
        c = D.features(list(insts), ["$close"], start_time=start, end_time=end)
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]  # -> Series，索引 (instrument, datetime)
        if isinstance(c.index, pd.MultiIndex):
            c = c.unstack(-1)  # instrument × datetime
        c = c[~c.index.duplicated(keep="last")]
        return c

    def _ensure_close_panel(self, asof, insts, date):
        """确保 asof 对应的 close 面板已覆盖到 `date`；不足则增量预取（每次多取约半年）。

        回测中 `date` 单调递增，故面板只增不删。同一 asof 窗口内只会产生极少量
        D.features 调用（约每半年一次），避免每日重复拉取 close 导致的性能问题。
        """
        asof = pd.Timestamp(asof)
        date = pd.Timestamp(date)
        panel = self._close_panel.get(asof)
        if panel is None or panel.shape[1] == 0:
            start = asof - pd.Timedelta(days=14)
            fetch_end = date + pd.Timedelta(days=180)
            self._close_panel[asof] = self._fetch_close(insts, start, fetch_end)
            return self._close_panel[asof]
        last_col = panel.columns.max()
        if date > last_col:
            fetch_end = date + pd.Timedelta(days=180)
            ext = self._fetch_close(insts, last_col + pd.Timedelta(days=1), fetch_end)
            if ext.shape[1] > 0:
                self._close_panel[asof] = panel.join(ext, how="outer")
        return self._close_panel[asof]

    @staticmethod
    def _close_at(panel, d):
        """从面板中取 <= d 的最近一个交易日的 close（instrument 索引 Series）。"""
        d = pd.Timestamp(d)
        cols = [c for c in panel.columns if c <= d]
        if not cols:
            return pd.Series(dtype="float64")
        return panel[cols[-1]]

    def _adjust_by_close(self, w, asof, date):
        """按持有期间的 close 涨跌幅对基准权重做漂移调整（buy & hold 漂移）。

        权重文件给出的是 as-of 日期的基准权重；从 asof 到当前调仓日之间，
        成分股价格变动会改变持仓权重。按
            w_i' = w_i * (close[date] / close[asof])
        重新计算后再归一化，等价于「as-of 日按基准权重买入并持有到 date」
        的实际组合权重，避免直接用过期权重当作当前权重引入隐性偏离。
        缺失 close 的标的按 ratio=1 处理（保留原权重）。
        性能：close 面板按 as-of 缓存，同一窗口内每天只做内存切片，不重复拉数据。
        """
        insts = list(w.index)
        panel = self._ensure_close_panel(asof, insts, date)
        c0 = self._close_at(panel, asof)
        c1 = self._close_at(panel, date)
        ratio = (c1 / c0.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        ratio = ratio.reindex(w.index).fillna(1.0)
        w2 = w * np.nan_to_num(ratio.to_numpy(dtype=np.float64), nan=1.0)
        s = w2.sum()
        if not np.isfinite(s) or s <= 0:
            return w
        return pd.Series(w2 / s, index=w.index)

    # ------------------------------------------------------------------
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
                w, asof = self._load_weight_file(date)
                if w is not None and self.adjust_close:
                    w = self._adjust_by_close(w, asof, date)
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
                    # D.features 返回 MultiIndex(instrument, datetime)，去掉 datetime 层
                    dts = [n for n in wf.index.names if n in ("datetime", "date")]
                    wf = wf.xs(pd.Timestamp(date), level=dts[0]) if dts else wf.droplevel(-1)
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
        """把 w 欧氏投影到「带上下界的单纯形」上，严格满足约束。

        求解
            min_w  ||w - w_raw||^2
            s.t.   sum(w) = 1,  (1-d)*base_i <= w_i <= (1+d)*base_i
        这是标准的 bounded simplex projection：解为
            w_i(λ) = clip(w_raw_i - λ, lo_i, hi_i)
        其中 sum_i w_i(λ) 关于 λ 单调不增，用二分法求 sum(w(λ)) = 1 的 λ。

        相比「clip → 归一化 → 重复 50 次」的近似 water-filling，该解是严格最优解，
        且归一化后不会再把权重推出边界。

        非成分股（base=0）的上下界都是 0，不会凭空持有。
        """
        max_dev = self.max_dev
        if max_dev is None or max_dev <= 0:
            return w

        base_w = base_w.reindex(w.index).fillna(0.0)
        base_arr = base_w.to_numpy(dtype=np.float64)
        total_base = base_arr.sum()
        if not np.isfinite(total_base) or total_base <= 0:
            return base_w if isinstance(base_w, pd.Series) else w
        base_arr = base_arr / total_base

        w_raw = np.nan_to_num(
            w.to_numpy(dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0
        )
        lo = np.maximum((1.0 - max_dev) * base_arr, 0.0)
        hi = (1.0 + max_dev) * base_arr

        # 可行性：sum(lo) <= 1 <= sum(hi)（base 已归一化且 lo>=0 时自动成立）
        if hi.sum() < 1.0 - 1e-12 or lo.sum() > 1.0 + 1e-12:
            # 上/下界本身装不下 1：退化为 clip + 归一化（尽力而为）
            logger.warning(
                "max_dev=%.3f 的上下界不可行（sum(lo)=%.4f, sum(hi)=%.4f），"
                "本日退化为 clip+归一化。",
                max_dev, lo.sum(), hi.sum(),
            )
            wc = np.clip(w_raw, lo, hi)
            s = wc.sum()
            wc = wc / s if (np.isfinite(s) and s > 0) else base_arr
            return pd.Series(wc, index=w.index)

        # 二分 λ：λ 越小 sum 越大
        lam_lo = float((w_raw - hi).min()) - 1.0   # 全部顶到上界
        lam_hi = float((w_raw - lo).max()) + 1.0   # 全部压到下界
        for _ in range(100):
            lam = 0.5 * (lam_lo + lam_hi)
            if np.clip(w_raw - lam, lo, hi).sum() > 1.0:
                lam_lo = lam
            else:
                lam_hi = lam
        wc = np.clip(w_raw - 0.5 * (lam_lo + lam_hi), lo, hi)
        s = wc.sum()
        if not np.isfinite(s) or s <= 0:
            wc = base_arr
        else:
            wc = wc / s
        return pd.Series(wc, index=w.index)

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_score(score):
        """把当日 pred 整理成 instrument 索引的 float Series（保留 NaN）。"""
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        if isinstance(score, pd.Series) and score.index.nlevels == 2:
            dts = score.index.get_level_values(0).unique()
            score = score.xs(dts[-1], level=0)
        score = pd.to_numeric(score, errors="coerce")
        score = score.replace([np.inf, -np.inf], np.nan)
        if isinstance(score.index, pd.MultiIndex):
            score = score.droplevel(list(range(score.index.nlevels - 1)))
        return score[~score.index.duplicated(keep="first")]

    # ------------------------------------------------------------------
    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        # score：当日 pred。可能是 (datetime, instrument) 多索引，也可能已压成 instrument 索引
        score = self._flatten_score(score)
        if score.empty:
            return None

        date = trade_start_time

        # ① 基准权重（已归一化、只含权重 > 0 的成分）
        base_w = self._base_weight(date)
        score= score[score.index.isin(base_w.index)]
        if base_w is None:
            # 取不到真实基准（base="equal" 或权重缺失）：用 score 覆盖的股票池等权兜底
            universe = score.dropna().index
            if len(universe) == 0:
                return None
            base_w = pd.Series(1.0 / len(universe), index=universe, dtype=np.float64)
            logger.warning("%s base=%r：未取到真实基准权重，回退为等权基准（%d 只）。",
                           pd.Timestamp(date).date(), self.base, len(universe))

        # ② 可选：按【score 排名】选股（不是按倾斜后权重排名）
        #    与 max_dev 互斥，构造时已校验；入选股之外权重直接为 0
        if self.topk_limit and self.topk_limit > 0 and len(base_w) > self.topk_limit:
            rank = score.reindex(base_w.index).dropna()
            if rank.empty:
                return None
            keep = rank.nlargest(self.topk_limit).index
            base_w = base_w.reindex(keep).dropna()
            if base_w.empty:
                return None
            base_w = base_w / base_w.sum()

        # ③ 只在最终 universe 内做截面 z-score；
        #    缺失 score 的成分股取中性值 mu（=> z=0 => 保留基准权重），绝不从基准里删除
        s = score.reindex(base_w.index)
        valid = s.dropna()
        if valid.empty:
            # 当日没有任何有效 pred：退化为纯基准
            return {stock: float(weight) for stock, weight in base_w.items() if weight > 1e-6}

        mu = float(valid.mean())
        sd = float(valid.std())
        if not np.isfinite(sd) or sd <= 1e-8:
            z = pd.Series(0.0, index=base_w.index, dtype=np.float64)
        else:
            z = (s.fillna(mu) - mu) / sd

        # ④ 乘性倾斜（减去最大值，数学结果不变但避免 exp 溢出）
        x = self.alpha * z.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=60.0, neginf=-60.0)
        x = x - np.max(x)
        tilt = np.exp(x)

        w = base_w.to_numpy(dtype=np.float64) * tilt
        ssum = w.sum()
        if not np.isfinite(ssum) or ssum <= 0:
            w = base_w.copy()
        else:
            w = pd.Series(w / ssum, index=base_w.index)
            # ⑤ 约束单个标的相对基础权重的偏离不超过 max_dev（默认 ±50%）
            w = self._cap_deviation_from_base(w, base_w)

        # 返回权重（和为 1）；订单生成器会按 risk_degree 自动乘以风险预算
        return {stock: float(weight) for stock, weight in w.items() if weight > 1e-6}
