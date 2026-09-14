"""
run_ic.py — Alpha158 全因子（158 个）批量 IC 计算与【因子筛选】

与同目录单因子脚本 run_ic_analysis.py 的关系：
  * step0_load / step1_neutral / step2_ic / step3_groups 四步的【口径与作用完全一致】，
    只是把「单个因子」推广为「一批因子」，最终汇成一张 158 行的因子指标表，用于筛选。

保留的单个因子指标（每行一个因子）：
  Rank IC / Pearson IC / ICIR / Robust_ICIR / IC t-value / IC > 0 比例
  / Q1->QH 单调性 / QH-Q1 收益(年化) / QH-AVG 收益(年化)
  / 多头换手率 / 多头净收益(年化, 扣费) / 多头净 Sharpe（cost=COST_RATES）
  / 因子类别（按日度 RankIC 相关性聚类的类别标签）

跨因子分析（Step 10）：
  * 用各因子的日度 RankIC（d_rank）对齐成 date × factor 面板，求因子×因子相关性表，
    落盘 rank_ic_corr.csv；
  * 以 1-|corr| 为距离做 average-linkage 层次聚类（CLUSTER_N 指定类别数），
    类别标签按规模降序重排为 C1/C2/... 写回指标表的「因子类别」列，
    并单独落盘 factor_cluster.csv。

落盘产物（RESULTS 目录）：
  * alpha158_factor_metrics.csv : 逐因子指标表（末列「因子类别」）
  * rank_ic_corr.csv            : 因子×因子 日度 RankIC 相关性表
  * factor_cluster.csv          : 因子 -> 聚类类别
  * neutralized_factors.pkl     : 中性化后的因子长表
                                  （index=(datetime, instrument)，columns=因子，float32）
                                  后续分析可直接 pd.read_pickle 复用，无需重算中性化。

复用 Qlib 框架（不再重复造轮子）：
  * qlib.contrib.data.loader.Alpha158DL : 取 158 因子名称与表达式
  * ic_utils.U                          : IC / 分组 / 中性化 / 绩效 / 画图

运行（必须用 qlib_me 环境，且需要 fork 启动方式）：
  JOBLIB_START_METHOD=fork LOKY_START_METHOD=fork \
      ~/miniconda3/envs/qlib_me/bin/python run_ic.py

常用参数：
  --limit 5              只跑前 N 个因子（冒烟测试，0=全部 158 个）
  --factors KMID,KLEN    只跑指定因子
  --no-neutral           跳过中性化（只算原始因子）
  --horizon 3            主分析换仓周期（交易日）
  --chunk 6              每批处理的因子数（内存紧张时调小）
  --no-save-neutral      不落盘中性化后的因子 pkl
"""

import os
import sys
import time
import argparse
import logging
import multiprocessing

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import ic_utils as U

# ----------------------------------------------------------------------------
# 配置（与 run_ic_analysis.py 保持一致）
# ----------------------------------------------------------------------------
RESULTS = os.path.join(HERE, "results", "alpha158_all")

WARM_START = "2009-01-01"        # 预热（因子计算需要历史）
START = "2017-01-01"             # 分析主区间起点
END = "2024-01-01"               # 分析主区间终点
TEST_END = "2026-09-01"    
N_GROUPS = 5                     # 分组数 Q1..Q5（QH=最高组）
PRIMARY_H = 3                    # 主分析换仓周期（日）

# 中性化控制变量
NEUTRAL_CONTROLS = ["SIZE", "BETA"]
NEUTRALIZE = True
# 中性化方式：size / all / industry / industry+size
NEUTRAL_MODE = "industry+size"
INDUSTRY_CSV = os.path.join(HERE, "data", "industry.csv")

# 交易成本（往返成本比例，与 run_ic_analysis.py 的 COST_RATES 同口径）
COST_RATES = 0.001

CHUNK = 6                        # 每批因子数（分块以控制内存峰值）

# ----- 跨因子相关性聚类（Step 10）-----
CLUSTER_N = 8                    # 目标类别数（<=0 -> 按 CLUSTER_CORR_MIN 阈值自动切分）
CLUSTER_CORR_MIN = 0.6           # 自动切分时：|corr| >= 该值视为同类（距离阈值=1-该值）
CLUSTER_USE_ABS = True           # 距离 = 1-|corr|：方向相反但高度相关的因子视为同类信息

# ----- 中性化因子落盘 -----
SAVE_NEUTRAL_PKL = True          # 是否把中性化后的因子（长表）落盘为 pkl
NEUTRAL_PKL_NAME = "neutralized_factors.pkl"

# 模式 -> (连续控制变量集合, 是否加入行业哑变量)
NEUTRAL_MODE_MAP = {
    "size":          (["SIZE"], False),
    "all":           (list(NEUTRAL_CONTROLS), False),
    "industry":      ([], True),
    "industry+size": (["SIZE"], True),
}

log = logging.getLogger("alpha158_all")


# ----------------------------------------------------------------------------
# 日志 / IO
# ----------------------------------------------------------------------------
def setup_logger():
    os.makedirs(RESULTS, exist_ok=True)
    logger = logging.getLogger("alpha158_all")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(RESULTS, "run_log.txt"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_csv(df, name):
    path = os.path.join(RESULTS, name)
    df.to_csv(path,encoding="gbk")
    log.info(f"  -> 保存 {path}")
    return path


def save_pickle(obj, name):
    path = os.path.join(RESULTS, name)
    obj.to_pickle(path)
    log.info(f"  -> 保存 {path}（{os.path.getsize(path) / 1e6:.1f} MB）")
    return path


def _wides_to_long(wides):
    """{factor: WIDE(datetime×instrument)} -> 长表（index=(datetime, instrument)，float32）。

    与 step0 加载的原始因子长表同构：仅保留因子有值的 (datetime, instrument) 行，
    便于后续直接喂给 to_wide / step2_ic 等函数复用。
    """
    cols = {}
    for nm, w in wides.items():
        s = w.stack()
        s = s[s.notna()].astype("float32")
        s.index.names = ["datetime", "instrument"]
        cols[nm] = s.rename(nm)
    return pd.concat(cols, axis=1)


def _fmt(v, spec="+.4f"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return format(float(v), spec)
    except (TypeError, ValueError):
        return "n/a"


# ----------------------------------------------------------------------------
# Step 0: 加载全部 158 个因子 + 未来收益 / 基准 / 控制变量 / 行业标签
# ----------------------------------------------------------------------------
def step0_load(names):
    log.info("=" * 70)
    log.info("Step 0 | 加载 Alpha158 全因子 + 未来收益 / 基准 / 控制变量 / 行业")

    all_names, all_fields = U.get_alpha158_names()
    exprs = dict(zip(all_names, all_fields))
    if names is None:
        names = list(all_names)
    fields = [exprs[n] for n in names]
    log.info(f"  本次因子数 = {len(names)}（Alpha158 共 {len(all_names)} 个）")

    # 一次性批量加载（1 次 DB 查询，远快于逐因子查询）
    t0 = time.time()
    factor_long = U.load_long(fields, WARM_START, END)
    factor_long.columns = names
    factor_long = factor_long.replace([np.inf, -np.inf], np.nan)
    log.info(f"  因子面板: {factor_long.shape}  耗时 {time.time() - t0:.1f}s")

    fwd1_long = U.load_fwd1_long(WARM_START, END)
    fwd1_wide = U.fwd1_wide_from_long(fwd1_long)
    bench = U.load_benchmark_ret(WARM_START, END)

    # 控制变量（Size/Beta）：一次批量查询
    ctrl_exprs = dict(U.CONTROL_EXPRS)
    ctrl_all = U.load_long(list(ctrl_exprs.values()), WARM_START, END)
    ctrl_all.columns = list(ctrl_exprs.keys())
    controls_wide = {c: U.to_wide(ctrl_all[[c]], c) for c in ctrl_exprs}

    # 行业标签（可选）
    ind_wide = U.load_industry_wide(INDUSTRY_CSV, WARM_START, END)
    log.info("  行业标签：" + ("已加载" if ind_wide is not None else "无（行业中性化 N/A）"))
    log.info(f"  主区间: {START} ~ {END}，交易日={len(fwd1_wide)}，股票={fwd1_wide.shape[1]}")

    meta = pd.DataFrame([{
        "n_factors_total": len(all_names), "n_factors_used": len(names),
        "start": START, "end": END, "warm_start": WARM_START,
        "neutralize": NEUTRALIZE, "neutral_mode": NEUTRAL_MODE if NEUTRALIZE else "none",
        "n_groups": N_GROUPS, "primary_h": PRIMARY_H,
    }])
    save_csv(meta, "meta.csv")

    # 主区间切片落盘（便于复核）
    def clip_long(df):
        d = df.index.get_level_values(0)
        mask = (d >= pd.Timestamp(START)) & (d <= pd.Timestamp(END))
        return df[mask]

    save_csv(clip_long(fwd1_long).reset_index(), "raw_fwd1.csv")
    bench.to_frame().to_csv(os.path.join(RESULTS, "raw_benchmark_ret.csv"))

    return (names, factor_long, fwd1_long, fwd1_wide, bench, controls_wide, ind_wide)


# ----------------------------------------------------------------------------
# Step 1: 中性化（批量横截面 OLS 残差）
#   同 run_ic_analysis.py，只是把「一次解一个因子」推广为「一次解一批因子」：
#   同一交易日的设计阵（const + 连续控制 + 行业哑变量）对所有因子相同，
#   故用同一个 X 一次性解出本批全部因子的残差，避免逐因子重复构建设计阵。
#   返回 {factor_name: 残差 WIDE}。
# ----------------------------------------------------------------------------
def _neutralize_batch(wides, dates, insts, ctrl_names, controls_wide, ind_wide):
    names = list(wides)
    n_d, n_i, n_f = len(dates), len(insts), len(names)

    Y = np.empty((n_d, n_i, n_f), dtype=float)
    for j, nm in enumerate(names):
        Y[:, :, j] = wides[nm].reindex(index=dates, columns=insts).values.astype(float)

    C = None
    if ctrl_names:
        C = np.stack(
            [controls_wide[c].reindex(index=dates, columns=insts).values.astype(float)
             for c in ctrl_names], axis=-1)

    IND = ind_wide.reindex(index=dates, columns=insts).values if ind_wide is not None else None

    R = np.full((n_d, n_i, n_f), np.nan)
    for i in range(n_d):
        valid = np.ones(n_i, dtype=bool)
        if IND is not None:
            valid &= ~pd.isna(IND[i])
        if C is not None:
            valid &= ~np.isnan(C[i]).any(axis=1)
        if valid.sum() < 5:
            continue

        parts = [np.ones((valid.sum(), 1))]
        if C is not None:
            parts.append(C[i][valid])
        if IND is not None:
            cats = pd.Categorical(IND[i][valid])
            Xd = np.zeros((valid.sum(), len(cats.categories)))
            Xd[np.arange(valid.sum()), cats.codes] = 1.0
            parts.append(Xd)
        Xb = np.concatenate(parts, axis=1)
        if Xb.shape[0] < Xb.shape[1] + 1:      # 自由度不足
            continue

        cols = np.flatnonzero(valid)
        Yb = Y[i][valid]
        nanm = np.isnan(Yb)
        if not nanm.any():
            # 常见情形：本批因子在当日缺失模式一致 -> 一次解多个右端项
            beta, *_ = np.linalg.lstsq(Xb, Yb, rcond=None)
            R[i, cols] = Yb - Xb @ beta
        else:
            # 缺失模式不一致 -> 逐因子各自剔除缺失行
            for j in range(n_f):
                m = ~nanm[:, j]
                if m.sum() < Xb.shape[1] + 1:
                    continue
                Xm = Xb[m]
                ym = Yb[m, j]
                beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
                R[i, cols[m], j] = ym - Xm @ beta

    return {nm: pd.DataFrame(R[:, :, j], index=dates, columns=insts)
            for j, nm in enumerate(names)}


def step1_neutral(fac_long_chunk, fwd1_wide, controls_wide, ind_wide):
    log.info("=" * 70)
    log.info(f"Step 1 | 中性化（模式={NEUTRAL_MODE if NEUTRALIZE else '关闭'}）")

    names = list(fac_long_chunk.columns)
    dates, insts = fwd1_wide.index, fwd1_wide.columns

    # LONG -> WIDE
    wides = {nm: U.to_wide(fac_long_chunk[[nm]], nm).reindex(index=dates, columns=insts)
             for nm in names}

    if not NEUTRALIZE:
        return wides

    ctrl_names, use_ind = NEUTRAL_MODE_MAP.get(NEUTRAL_MODE, (["SIZE"], True))
    # 全控制模式用配置里的全部控制变量
    if NEUTRAL_MODE == "all":
        ctrl_names = list(NEUTRAL_CONTROLS)
    ctrls = {c: controls_wide[c] for c in ctrl_names}
    ind = ind_wide if use_ind else None

    return _neutralize_batch(wides, dates, insts, list(ctrls), ctrls, ind)


# ----------------------------------------------------------------------------
# Step 2: 第一层 —— IC（Rank IC / Pearson IC / ICIR / Robust_ICIR / IC t / 胜率）
#   与 run_ic_analysis.py 的 step2_ic 口径一致。
#   额外返回日度 RankIC 序列 d_rank，供 Step 10 的跨因子相关性聚类复用。
# ----------------------------------------------------------------------------
def step2_ic(factor_long, label_long, h):
    d_pear, d_rank = U.calc_daily_ic(factor_long, label_long)
    s = U.summarize_ic(d_rank, d_pear)
    s["horizon"] = h
    return s, d_rank


# ----------------------------------------------------------------------------
# Step 3: 第二层 —— 分组收益（Q1..QH）+ 单调性 + 多空 / 多头超额年化
#   与 run_ic_analysis.py 的 step3_groups / step4_longshort 口径一致。
# ----------------------------------------------------------------------------
def step3_groups(fac_wide, fwd1_wide, h, n_groups=N_GROUPS):
    gcols = [f"G{i}" for i in range(1, n_groups + 1)]
    nav, daily = U.group_backtest(fac_wide, fwd1_wide, horizon=h, n_groups=n_groups)
    group_ret = (1.0 + daily).prod() ** (252.0 / max(len(daily), 1)) - 1.0

    # 只把真正分组的列喂给单调性（避免 LS/LV 混入）
    mono = U.monotonicity(group_ret[gcols])
    ls_ann = U.geom_metrics(daily["LS"])["ann_return"]   # QH - Q1（年化）
    lv_ann = U.geom_metrics(daily["LV"])["ann_return"]   # QH - AVG（年化）
    return group_ret, daily, mono, ls_ann, lv_ann


# ----------------------------------------------------------------------------
# Step 9: 交易成本（NetReturn = Gross - Turnover × Cost）
#   口径与 run_ic_analysis.py 的 step9_cost 一致：用 group_assignments 里【最高组
#   (=Gn/QH)】的成分变化算多头单边换手，再对多头腿【QH - AVG】的日收益扣费。
# ----------------------------------------------------------------------------
def step9_cost(fac_wide, fwd1_wide, daily, h, n_groups=N_GROUPS, cost=COST_RATES):
    """单个因子：多头腿（最高组 QH）的换手与扣费后绩效。

    返回 dict：turnover_avg_long / net_ann_return_long / net_sharpe_long
    """
    dates = list(fwd1_wide.index)
    G, reb_pos, reb_sorted = U.group_assignments(fac_wide, h, n_groups)

    turnover_daily_long = pd.Series(0.0, index=dates)
    turnover_avg_long = 0.0
    if len(reb_pos) >= 2:
        top = (G[reb_pos] == n_groups - 1)                    # 各换仓日的最高组成分
        top_cnt = top.sum(axis=1)
        ov_top = (top[1:] & top[:-1]).sum(axis=1)             # 与上一期成分的交集
        to_long = 1.0 - ov_top / np.maximum(top_cnt[1:], 1)   # 最高组单边换手率
        for k in range(1, len(reb_pos)):
            turnover_daily_long.loc[reb_sorted[k]] = float(to_long[k - 1])
        turnover_avg_long = float(to_long.mean())

    gross_lv = daily["LV"].astype(float).fillna(0.0)          # QH - AVG 毛日收益
    net_lv = gross_lv - turnover_daily_long * cost            # 换仓日按换手扣费
    m_lv = U.geom_metrics(net_lv)
    return {
        "turnover_avg_long": turnover_avg_long,
        "net_ann_return_long": m_lv["ann_return"],
        "net_sharpe_long": m_lv["sharpe"],
    }


# ----------------------------------------------------------------------------
# 单因子：跑完 step2 + step3 + step9，产出表里的一行（附带日度 RankIC）
# ----------------------------------------------------------------------------
def evaluate_factor(name, fac_wide, label_long, fwd1_wide, h):
    # WIDE -> LONG（中性化后因子）
    s = fac_wide.stack().rename(name)
    fac_long = s.to_frame()
    fac_long.index.names = ["datetime", "instrument"]
    fac_long = fac_long.sort_index()

    ic, d_rank = step2_ic(fac_long, label_long, h)
    _, daily, mono, ls_ann, lv_ann = step3_groups(fac_wide, fwd1_wide, h, N_GROUPS)
    cost = step9_cost(fac_wide, fwd1_wide, daily, h)

    row = {
        "factor": name,
        "N_days": ic["N_days"],
        "Rank IC": ic["RankIC"],
        "Pearson IC": ic["Pearson_IC"],
        "ICIR": ic["ICIR"],
        "Robust_ICIR": ic["Robust_ICIR"],
        "IC t-value": ic["IC_t"],
        "IC > 0 比例": ic["P_IC_gt_0"],
        "Q1->QH 单调性": mono,
        "QH-Q1 收益(年化)": ls_ann,
        "QH-AVG 收益(年化)": lv_ann,
        **cost,
    }
    return row, d_rank


# ----------------------------------------------------------------------------
# Step 10: 跨因子 —— d_rank 相关性表 + 基于相关性矩阵的层次聚类
#   1) 各因子日度 RankIC 对齐成 date × factor 面板 -> 因子×因子相关矩阵；
#   2) corr -> 距离（默认 1-|corr|，同向/反向的高相关因子都算同类信息）；
#   3) average-linkage 层次聚类：CLUSTER_N 指定类别数，<=0 时按距离阈值自动切分；
#   4) 类别标签按规模降序重排为 C1/C2/...，写回最终指标表。
# ----------------------------------------------------------------------------
def _hier_cluster_raw(dist, n_clusters, corr_min):
    """对因子距离矩阵做层次聚类，返回每个因子的原始类别号（np.ndarray）。"""
    n = len(dist)
    if n <= 1:
        return np.ones(n, dtype=int)

    m = dist.values.astype(float)
    np.fill_diagonal(m, 0.0)
    z = linkage(squareform((m + m.T) / 2.0, checks=False), method="average")

    if n_clusters and n_clusters > 0:
        return fcluster(z, t=int(min(n_clusters, n)), criterion="maxclust")
    return fcluster(z, t=float(1.0 - corr_min), criterion="distance")


def _relabel_by_size(raw, names):
    """把原始类别号按类别规模降序重排为 C1/C2/...，返回 Series(index=factor)。"""
    s = pd.Series(raw, index=names)
    remap = {old: f"C{i + 1}" for i, old in enumerate(s.value_counts().index)}
    return s.map(remap).rename("cluster")


def step_corr_cluster(rank_ic_map, n_clusters=CLUSTER_N, corr_min=CLUSTER_CORR_MIN,
                      use_abs=CLUSTER_USE_ABS):
    """由各因子日度 RankIC 构建相关性表并聚类。

    参数
      rank_ic_map : {factor: d_rank(Series, index=date)}，来自 step2_ic
    返回
      corr   : 因子×因子 相关性表（DataFrame，index/columns 均为 factor）
      labels : Series(index=factor) -> 类别标签 "C1"/"C2"/...
    """
    if not rank_ic_map:
        return None, pd.Series(dtype=object)

    ic_df = pd.DataFrame(rank_ic_map).sort_index()          # date × factor
    corr = ic_df.corr()                                     # 逐对完整样本相关系数
    if corr.empty:
        return None, pd.Series(dtype=object)

    # 相关性 -> 距离（|corr| 越小距离越大；缺失/退化相关性按“不相关”处理）
    c = corr.abs() if use_abs else corr
    c = c.fillna(0.0).clip(-1.0, 1.0)
    dist = pd.DataFrame(1.0 - c.values, index=c.index, columns=c.columns)

    raw = _hier_cluster_raw(dist, n_clusters, corr_min)
    return corr, _relabel_by_size(raw, list(corr.index))


# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Alpha158 batch factor screening (IC)")
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--warm-start", default=WARM_START)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 个因子（0=全部）")
    p.add_argument("--factors", default="", help="逗号分隔的因子名，只跑这些因子")
    p.add_argument("--horizon", type=int, default=PRIMARY_H, help="主分析换仓周期（交易日）")
    p.add_argument("--groups", type=int, default=N_GROUPS)
    p.add_argument("--chunk", type=int, default=CHUNK, help="每批处理的因子数")
    p.add_argument("--no-neutral", action="store_true", help="跳过中性化")
    p.add_argument("--neutral-mode", default=NEUTRAL_MODE,
                   choices=["size", "all", "industry", "industry+size"])
    p.add_argument("--clusters", type=int, default=CLUSTER_N,
                   help="RankIC 相关性聚类的类别数（<=0 时按 --cluster-corr-min 自动切分）")
    p.add_argument("--cluster-corr-min", type=float, default=CLUSTER_CORR_MIN,
                   help="自动切分类别时的 |corr| 阈值（距离阈值 = 1 - 该值）")
    p.add_argument("--no-save-neutral", action="store_true",
                   help="不落盘中性化后的因子 pkl")
    return p.parse_args()


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    global RESULTS, START, END, WARM_START, PRIMARY_H, N_GROUPS, NEUTRALIZE, NEUTRAL_MODE, log
    global CLUSTER_N, CLUSTER_CORR_MIN, SAVE_NEUTRAL_PKL

    args = parse_args()
    START, END, WARM_START = args.start, args.end, args.warm_start
    PRIMARY_H, N_GROUPS, CHUNK = args.horizon, args.groups, max(1, args.chunk)
    NEUTRALIZE = not args.no_neutral
    NEUTRAL_MODE = args.neutral_mode
    CLUSTER_N, CLUSTER_CORR_MIN = args.clusters, args.cluster_corr_min
    SAVE_NEUTRAL_PKL = SAVE_NEUTRAL_PKL and not args.no_save_neutral

    U.qlib_init()
    log = setup_logger()    # qlib.init 会重置 logging，需在其后重建
    log.info("Alpha158 全因子 IC 筛选 | " + time.strftime("%Y-%m-%d %H:%M:%S"))

    all_names, _ = U.get_alpha158_names()
    if args.factors:
        want = [x.strip() for x in args.factors.split(",") if x.strip()]
        names = [n for n in all_names if n in want]
    elif args.limit and args.limit > 0:
        names = list(all_names[:args.limit])
    else:
        names = list(all_names)

    (names, factor_long, fwd1_long, fwd1_wide, bench,
     controls_wide, ind_wide) = step0_load(names)

    # 主 horizon 的标签（一次构建，全因子复用）
    label_wide = U.forward_cumprod_ret(fwd1_wide, PRIMARY_H)
    label_long = label_wide.stack().rename("LABEL").reset_index().set_index(
        ["datetime", "instrument"])

    log.info("=" * 70)
    log.info(f"Step 1-3 | 逐批因子：中性化 -> IC -> 分组（共 {len(names)} 个，批大小 {CHUNK}）")
    t0 = time.time()
    rows = []
    rank_ic_map = {}          # {factor: 日度 RankIC 序列}，供 Step 10 聚类复用
    neu_parts = []            # 中性化因子长表分片（跑完合并落盘为 pkl）
    for lo in range(0, len(names), CHUNK):
        batch = names[lo:lo + CHUNK]
        sub = factor_long[batch]
        neu_wides = step1_neutral(sub, fwd1_wide, controls_wide, ind_wide)
        if SAVE_NEUTRAL_PKL:
            neu_parts.append(_wides_to_long(neu_wides))
        for nm in batch:
            try:
                row, d_rank = evaluate_factor(nm, neu_wides[nm], label_long, fwd1_wide, PRIMARY_H)
                rows.append(row)
                rank_ic_map[nm] = d_rank
            except Exception as e:
                log.info(f"  [跳过] {nm}: {type(e).__name__}: {e}")
                rows.append({"factor": nm})
        done = min(lo + CHUNK, len(names))
        log.info(f"  [{done}/{len(names)}] 完成，累计 {time.time() - t0:.0f}s")
        del neu_wides, sub

    table = pd.DataFrame(rows)

    # ---- 中性化因子落盘（长表 pkl，后续复用无需重算）----
    if neu_parts:
        log.info("=" * 70)
        log.info(f"落盘 | 中性化因子 pkl（中性化模式={NEUTRAL_MODE if NEUTRALIZE else '关闭'}）")
        neu_long = pd.concat(neu_parts, axis=1)
        del neu_parts
        dax = neu_long.index.get_level_values(0)
        log.info(f"  {neu_long.shape}，区间 {dax.min().date()} ~ {dax.max().date()}，"
                 f"dtype={neu_long.dtypes.iloc[0]}")
        save_pickle(neu_long, NEUTRAL_PKL_NAME)
        del neu_long

    # ---- Step 10: d_rank 相关性表 + 相关性聚类，类别写回最终表 ----
    log.info("=" * 70)
    log.info(f"Step 10 | d_rank 相关性聚类（目标类别数={CLUSTER_N}，"
             f"距离 = 1 - {'|corr|' if CLUSTER_USE_ABS else 'corr'}）")
    corr, labels = step_corr_cluster(rank_ic_map, CLUSTER_N, CLUSTER_CORR_MIN, CLUSTER_USE_ABS)
    table["因子类别"] = "NA"
    if corr is not None:
        save_csv(corr, "rank_ic_corr.csv")
        save_csv(labels.to_frame(), "factor_cluster.csv")
        table["因子类别"] = table["factor"].map(labels).fillna("NA")
        sizes = table["因子类别"].value_counts()
        sizes = sizes.reindex(sorted(sizes.index, key=lambda s: (len(s), s)))
        log.info(f"  因子相关性表: {corr.shape[0]}×{corr.shape[1]}，"
                 f"平均 |corr| = {corr.abs().values[np.triu_indices(len(corr), k=1)].mean():.3f}")
        log.info(f"  实际类别数 = {len(sizes)}：" +
                 "，".join(f"{c}={int(n)}" for c, n in sizes.items()))
        for c in sizes.index:
            members = table.loc[table["因子类别"] == c, "factor"].tolist()
            log.info(f"    {c}: " + ", ".join(members[:12]) +
                     (f" ... (+{len(members) - 12})" if len(members) > 12 else ""))

    # 按 ICIR 降序，方便筛选（RankIC 为负的因子同样有效，方向在筛选时再统一）
    table["abs_ICIR"] = table["ICIR"].abs()
    table = table.sort_values("abs_ICIR", ascending=False).drop(columns=["abs_ICIR"])
    table = table.reset_index(drop=True)
    save_csv(table, "alpha158_factor_metrics.csv")

    # 汇总日志
    log.info("=" * 70)
    log.info(f"Step 4 | 结果汇总：共 {len(table)} 个因子")
    log.info(f"  Rank IC 均值 = {table['Rank IC'].mean():+.4f}，"
             f"|ICIR| 均值 = {table['ICIR'].abs().mean():.3f}，"
             f"IC t 绝对值 > 2 的因子数 = {int((table['IC t-value'].abs() > 2).sum())}")
    log.info("  按 |ICIR| 排名 Top 20：")
    for _, r in table.head(20).iterrows():
        log.info(f"    {r['factor']:<10} RankIC={_fmt(r['Rank IC'])} "
                 f"ICIR={_fmt(r['ICIR'], '+.3f')} "
                 f"RobICIR={_fmt(r['Robust_ICIR'], '+.3f')} "
                 f"t={_fmt(r['IC t-value'], '+.2f')} "
                 f"P(IC>0)={_fmt(r['IC > 0 比例'], '.2%')} "
                 f"mono={_fmt(r['Q1->QH 单调性'], '+.2f')} "
                 f"QH-Q1={_fmt(r['QH-Q1 收益(年化)'], '+.2%')} "
                 f"QH-AVG={_fmt(r['QH-AVG 收益(年化)'], '+.2%')} "
                 f"to_L={_fmt(r['turnover_avg_long'], '.2%')} "
                 f"netAnn_L={_fmt(r['net_ann_return_long'], '+.2%')} "
                 f"netSharpe_L={_fmt(r['net_sharpe_long'], '.2f')} "
                 f"cat={r['因子类别']}")

    log.info("=" * 70)
    log.info(f"全部完成。结果目录：{RESULTS}")




if __name__ == "__main__":
    main()
