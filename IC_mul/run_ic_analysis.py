"""
run_ic_analysis.py — 对【单个 158 Alpha 因子】按 ic_research.md 文档做 6 层 + 衰减 / 条件依赖 /
交易成本 / Alpha Scorecard 的完整验证，每一步的中间结果都落盘保存。

复用 Qlib 框架：
  * qlib.contrib.data.handler.Alpha158 / Alpha158DL : 取 158 因子名称与表达式
  * qlib.contrib.evaluate.risk_analysis             : 组合风险指标
  * qlib.contrib.evaluate.long_short_backtest       : 含交易成本的多空回测
  * qlib.contrib.report.analysis_model /
    qlib.contrib.report.analysis_position           : 原生 IC / 分组 / 自相关 / 换手图

运行（必须用 qlib_me 环境，且需要 fork 启动方式）：
  JOBLIB_START_METHOD=fork LOKY_START_METHOD=fork \
      ~/miniconda3/envs/qlib_me/bin/python run_ic_analysis.py
"""

import os
import sys
import time
import logging
import multiprocessing

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

import numpy as np
import pandas as pd

import ic_utils as U

# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FACTOR_OVERRIDE = "VSUMN10"
RESULTS = os.path.join(HERE, "results", FACTOR_OVERRIDE)
os.makedirs(RESULTS, exist_ok=True)

WARM_START = "2009-01-01"        # 预热（因子计算需要历史）
START = "2010-01-01"             # 分析主区间起点
END = "2026-06-30"               # 分析主区间终点

# Walk-forward 年份区间（由主区间自动推导，避免写死）
#   WF 训练窗口为 test 年前 6 年（y-6 ~ y-1），故起点取 START+6，
#   终点取 END 所在年份。
WF_START_YEAR = int(pd.Timestamp(START).year) + 6
WF_END_YEAR = int(pd.Timestamp(END).year)

N_GROUPS = 5                    # 分组数 Q1..Q10（文档要求 10 组）
PRIMARY_H = 3                    # 主分析换仓周期（日）
DECAY_HS = [1, 2, 3, 5, 10, 20] # 衰减分析 horizons

# 中性化控制变量（见 ic_utils.CONTROL_EXPRS）
NEUTRAL_CONTROLS = ["SIZE", "BETA"]

# 是否在 Step0 之后对因子做中性化（在 Step1 IC 分析之前执行）
#   True  -> 下游 Step2~Step11 全部使用中性化后的因子
#   False -> 下游使用原始因子（中性化仅作为对照，不影响主流程）
NEUTRALIZE = True
# 当 NEUTRALIZE=True 时选择的中性化方式：
#   "size"     -> 仅剔除 Size
#   "all"      -> 剔除 NEUTRAL_CONTROLS（SIZE + BETA）
#   "industry" -> 行业中性化（需 me/data/industry.csv，缺失则回退到原始因子）
#   "industry+size" -> 行业 + Size 联合横截面回归残差（标准做法，需 industry.csv）
NEUTRAL_MODE = "industry+size"

# OOS 三段切分（日期区间）
OOS_SPLITS = {
    "Train(IS)": ("2010-01-01", "2018-12-31"),
    "Valid":      ("2019-01-01", "2021-12-31"),
    "Test(OOS)":  ("2022-01-01", "2026-06-30"),
}

# 交易成本情景（round-trip 成本比例）
COST_RATES = [0.001, 0.002, 0.003, 0.004]

# 可选行业标签文件（无则行业中性化标记为 N/A）
INDUSTRY_CSV = os.path.join(HERE, "data", "industry.csv")


# ----------------------------------------------------------------------------
# 日志：同时写文件 + 控制台
# ----------------------------------------------------------------------------
def setup_logger():
    logger = logging.getLogger("ic_single")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(RESULTS, "run_log.txt"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = setup_logger()


def save_csv(df, name):
    path = os.path.join(RESULTS, name)
    df.to_csv(path)
    log.info(f"  -> 保存 {path}")
    return path


def save_pkl(obj, name):
    import pickle
    path = os.path.join(RESULTS, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info(f"  -> 保存 {path}")
    return path


# ----------------------------------------------------------------------------
# Step 0: 选择“中间一个”因子 + 加载原始数据
# ----------------------------------------------------------------------------
def step0_load(factor_override=None):
    log.info("=" * 70)
    log.info("Step 0 | 选择中间因子并加载原始数据")
    names, fields = U.get_alpha158_names()
    exprs = dict(zip(names, fields))
    log.info(f"  Alpha158 共 {len(names)} 个因子；取中间一个 idx={len(names)//2}")
    factor_name = factor_override or names[len(names) // 2]
    factor_expr = exprs[factor_name]
    log.info(f"  选中因子: {factor_name}  |  表达式: {factor_expr}")

    # 加载因子 LONG 与未来 1 日收益 LONG
    factor_long = U.load_factor_long(factor_expr, "FACTOR", WARM_START, END)
    fwd1_long = U.load_fwd1_long(WARM_START, END)
    fwd1_wide = U.fwd1_wide_from_long(fwd1_long)
    bench = U.load_benchmark_ret(WARM_START, END)

    # 控制变量（Size/Beta/Mom/Vol/Liq）+ 成交量：一次性批量拉取（减少 D.features 调用）
    ctrl_exprs = dict(U.CONTROL_EXPRS)
    ctrl_cols = list(ctrl_exprs.values()) #+ ["$volume"]
    ctrl_names = list(ctrl_exprs.keys()) #+ ["$volume"]
    ctrl_all = U.load_long(ctrl_cols, WARM_START, END)   # 1 次 DB 查询
    ctrl_all.columns = ctrl_names
    controls_long = {c: ctrl_all[[c]] for c in ctrl_exprs}
    # controls_long["$volume"] = ctrl_all[["$volume"]]
    # controls_long["MKT"] = bench.to_frame("MKT")

    # # 市场收益（基准）作为宽表控制变量
    # mkt_wide = pd.DataFrame(
    #     np.tile(bench.values, (len(fwd1_wide.columns), 1)).T,
    #     index=fwd1_wide.index, columns=fwd1_wide.columns,
    # ).rename(columns={c: c for c in fwd1_wide.columns})
    # mkt_wide.columns = fwd1_wide.columns
    controls_wide = {c: U.to_wide(ctrl_all[[c]], c) for c in ctrl_exprs}
    # controls_wide["MKT"] = mkt_wide

    # 落盘原始数据（仅主区间切片，便于复核）
    def clip_long(df):
        mask = (df.index.get_level_values(0) >= pd.Timestamp(START)) & \
               (df.index.get_level_values(0) <= pd.Timestamp(END))
        return df[mask]

    save_csv(clip_long(factor_long).reset_index(), "raw_factor.csv")
    save_csv(clip_long(fwd1_long).reset_index(), "raw_fwd1.csv")
    bench.to_frame().to_csv(os.path.join(RESULTS, "raw_benchmark_ret.csv"))
    # 控制变量合并落盘（仅 MultiIndex 的 5 个风格因子；MKT 已体现在 benchmark 中）
    ctrl_all = pd.concat([controls_long[c] for c in U.CONTROL_EXPRS], axis=1)
    save_csv(clip_long(ctrl_all).reset_index(), "raw_controls.csv")

    meta = {
        "factor_name": factor_name,
        "factor_expr": factor_expr,
        "n_factors_total": len(names),
        "selected_index": names.index(factor_name),
        "start": START, "end": END, "warm_start": WARM_START,
    }
    pd.DataFrame([meta]).to_csv(os.path.join(RESULTS, "meta.csv"), index=False)
    log.info(f"  主区间: {START} ~ {END}，股票池 csi300，基准 SH000300")
    return factor_name, factor_expr, factor_long, fwd1_long, fwd1_wide, bench, controls_wide, controls_long


# ----------------------------------------------------------------------------
# Step 1: 中性化（Size/Beta + 行业）—— 在 Step0 之后、IC 分析之前执行
#   若 NEUTRALIZE=False，下游仍使用原始因子；中性化结果仅作为对照落盘。
#   返回 (res, neu_factor_long)：neu_factor_long 为下游实际使用的因子 LONG。
# ----------------------------------------------------------------------------
def step1_neutral(factor_long, fwd1_wide, controls_wide, hs):
    log.info("=" * 70)
    log.info("Step 1 | 中性化（剔除常见风险因子）")
    fac_wide = U.to_wide(factor_long)
    factor_name = factor_long.columns[0]
    label_wide = U.forward_cumprod_ret(fwd1_wide, hs[0])
    label_long = label_wide.stack().rename("LABEL").reset_index().set_index(["datetime", "instrument"])

    def ic_of(wide):
        st = wide.stack().rename("FACTOR")        # Series, MultiIndex (datetime, instrument)
        fl = st.to_frame()
        fl.index.names = ["datetime", "instrument"]
        _, d_rank = U.calc_daily_ic(fl, label_long)
        return float(d_rank.mean())

    def wide_to_long(wide):
        s = wide.stack().rename(factor_name)
        long = s.to_frame()
        long.index.names = ["datetime", "instrument"]
        return long.sort_index()

    raw_ic = ic_of(fac_wide)
    log.info(f"  原始 RankIC (h={hs[0]}d) = {raw_ic:+.4f}")

    # 行业中性化（可选）：若提供 me/data/industry.csv，则做行业哑变量回归取残差
    ind_ic = np.nan
    neu_ind = None
    ind_wide = U.load_industry_wide(INDUSTRY_CSV, WARM_START, END)
    if ind_wide is not None and len(ind_wide) > 0:
        log.info("  检测到行业标签，执行行业中性化（横截面行业哑变量回归残差）")
        neu_ind = U.neutralize_industry(fac_wide, ind_wide)
        ind_ic = ic_of(neu_ind)
        log.info(f"  行业中性后 RankIC  = {ind_ic:+.4f}")
    else:
        log.info("  未提供行业标签（me/data/industry.csv），行业中性化标记为 N/A")

    # Size 单独中性化
    size_wide = {k: controls_wide[k] for k in ["SIZE"]}
    neu_size = U.neutralize_factor(fac_wide, size_wide)
    size_ic = ic_of(neu_size)

    # 全部控制变量中性化
    neu_all = U.neutralize_factor(fac_wide, {k: controls_wide[k] for k in NEUTRAL_CONTROLS})
    all_ic = ic_of(neu_all)

    # 行业 + Size 联合中性化（同一回归同时剔除两者，标准做法）
    ind_size_ic = np.nan
    neu_ind_size = None
    if ind_wide is not None and len(ind_wide) > 0:
        neu_ind_size = U.neutralize_industry_size(fac_wide, ind_wide, size_wide)
        ind_size_ic = ic_of(neu_ind_size)
        log.info(f"  行业+Size 中性后 RankIC = {ind_size_ic:+.4f}")

    log.info(f"  Size 中性后 RankIC   = {size_ic:+.4f}")
    log.info(f"  全控制中性后 RankIC = {all_ic:+.4f}")

    res = pd.DataFrame([
        {"neutralization": "raw", "RankIC": raw_ic},
        {"neutralization": "size", "RankIC": size_ic},
        {"neutralization": "size+beta", "RankIC": all_ic},
        {"neutralization": "industry", "RankIC": ind_ic},
        {"neutralization": "industry+size", "RankIC": ind_size_ic},
    ])
    save_csv(res, "step1_neutral_ic.csv")
    save_pkl({"raw": fac_wide, "size_neu": neu_size, "all_neu": neu_all,
              "ind_neu": neu_ind, "ind_size_neu": neu_ind_size}, "step1_neutral_factor.pkl")

    # 决定下游实际使用的因子
    if not NEUTRALIZE:
        log.info(f"  中性化开关=关闭（NEUTRALIZE=False），下游使用【原始】因子")
        neu_factor_long = factor_long
    elif NEUTRAL_MODE == "size":
        log.info(f"  中性化开关=开启，模式=size，下游使用【Size 中性】因子")
        neu_factor_long = wide_to_long(neu_size)
    elif NEUTRAL_MODE == "industry":
        if neu_ind is not None:
            log.info(f"  中性化开关=开启，模式=industry，下游使用【行业中性】因子")
            neu_factor_long = wide_to_long(neu_ind)
        else:
            log.info(f"  中性化开关=开启，模式=industry，但缺失行业标签，回退使用【原始】因子")
            neu_factor_long = factor_long
    elif NEUTRAL_MODE == "industry+size":
        if neu_ind_size is not None:
            log.info(f"  中性化开关=开启，模式=industry+size，下游使用【行业+Size 联合中性】因子")
            neu_factor_long = wide_to_long(neu_ind_size)
        else:
            log.info(f"  中性化开关=开启，模式=industry+size，但缺失行业标签，回退使用【原始】因子")
            neu_factor_long = factor_long
    else:  # "all"
        log.info(f"  中性化开关=开启，模式=all，下游使用【全控制中性】因子")
        neu_factor_long = wide_to_long(neu_all)

    return res, neu_factor_long


# ----------------------------------------------------------------------------
# Step 2: 第一层 —— Rank IC / Pearson IC / ICIR / IC t / 胜率
# ----------------------------------------------------------------------------
def step2_ic(factor_long, fwd1_wide, hs):
    log.info("=" * 70)
    log.info("Step 2 | 第一层：IC（Rank IC / Pearson IC / ICIR / t / 胜率）")
    ic_store = {}
    rows = []
    for h in hs:
        label_wide = U.forward_cumprod_ret(fwd1_wide, h)
        label_long = label_wide.stack().rename("LABEL").reset_index().set_index(["datetime", "instrument"])
        d_pear, d_rank = U.calc_daily_ic(factor_long, label_long)
        ic_store[h] = {"pearson": d_pear, "rank": d_rank}
        s = U.summarize_ic(d_rank, d_pear)
        s.update({"horizon": h})
        rows.append(s)
        log.info(f"  h={h:>2}d | RankIC={s['RankIC']:+.4f}  ICIR={s['ICIR']:+.3f}  "
                 f"PearsonIC={s['Pearson_IC']:+.4f}  IC_t={s['IC_t']:.2f}  "
                 f"P(IC>0)={s['P_IC_gt_0']:.2%}  N={s['N_days']}")

    ic_summary = pd.DataFrame(rows)
    save_csv(ic_summary, "step2_ic_summary.csv")
    save_pkl(ic_store, "step2_daily_ic.pkl")

    # 主 horizon 的日度 IC 曲线
    h0 = hs[0]
    daily = pd.DataFrame({"RankIC": ic_store[h0]["rank"], "PearsonIC": ic_store[h0]["pearson"]})
    save_csv(daily, "step2_daily_ic_h%d.csv" % h0)
    U.plot_series(daily.cumsum(), os.path.join(RESULTS, "step2_ic_cumsum.png"),
                  title=f"{factor_long.columns[0]} 日度 IC 累计 (h={h0}d)")
    # 分布图用直方图（4246 个日度值不适合画柱状图，否则 x 轴标签挤成一团）
    U.plot_hist(daily["RankIC"].dropna(), os.path.join(RESULTS, "step2_rankic_hist.png"),
                title=f"Rank IC 分布 (h={h0}d)", xlabel="Rank IC", ylabel="天数(频数)")
    return ic_store, ic_summary


# ----------------------------------------------------------------------------
# Step 3: 第二层 —— 分组收益（Q1..Q10）+ 单调性
# ----------------------------------------------------------------------------
def step3_groups(factor_long, fwd1_wide, h, n_groups=N_GROUPS):
    log.info("=" * 70)
    log.info(f"Step 3 | 第二层：分组收益 Q1..Q{n_groups}（换仓 h={h}d）+ 单调性")
    fac_wide = U.to_wide(factor_long)
    group_cols = [f"G{i}" for i in range(1, n_groups + 1)]
    nav, daily = U.group_backtest(fac_wide, fwd1_wide, horizon=h, n_groups=n_groups)
    group_ret = (1 + daily).prod() ** (252 / len(daily)) - 1      # 年化组收益（算术）
    mono = U.monotonicity(group_ret)
    log.info("  各组年化收益：")
    for g in [f"G{i}" for i in range(1, n_groups + 1)] + ["LS"]:
        log.info(f"    {g:>4}: {group_ret[g]:+.4%}")
    log.info(f"  单调性 Corr(组序号, 组收益) = {mono:+.4f}")

    save_csv(daily, f"step3_group_daily_ret_h{h}.csv")
    save_csv(nav, f"step3_group_nav_h{h}.csv")
    save_csv(group_ret.rename("ann_return").to_frame(), f"step3_group_ann_return_h{h}.csv")
    pd.Series({"monotonicity": mono}).to_csv(os.path.join(RESULTS, f"step3_monotonicity_h{h}.csv"))
    U.plot_series(nav, os.path.join(RESULTS, f"step3_group_nav_h{h}.png"),
                  title=f"{factor_long.columns[0]} 分组净值 (h={h}d)")
    U.plot_bar(group_ret.drop("LS"), os.path.join(RESULTS, f"step3_group_ann_return_h{h}.png"),
               title=f"各组年化收益 (h={h}d)", rot=0)
    return nav, daily, group_ret, mono


# ----------------------------------------------------------------------------
# Step 4: 第三层 —— 多空组合（Long Q10 - Short Q1）风险指标
# ----------------------------------------------------------------------------
def step4_longshort(daily, bench, hs=1):
    log.info("=" * 70)
    log.info("Step 4 | 第三层：多空组合 Q10 - Q1 绩效（复用 qlib risk_analysis + 几何口径）")
    ls_daily = daily["LS"]
    bench_reidx = bench.reindex(ls_daily.index).fillna(0.0)
    geo = U.geom_metrics(ls_daily)
    geo_lv = {
        f"{k}_lv": v
        for k, v in U.geom_metrics(daily["LV"]).items()
    }
    log.info("  几何口径: " + ", ".join(f"{k}={v:.4f}" for k, v in geo.items()))

    out = {"horizon": hs}
    out.update(geo)
    out.update(geo_lv)
    save_csv(pd.DataFrame([out]), f"step4_longshort_metrics_h{hs}.csv")
    save_csv(daily[["LS","LV"]],
             f"step4_longshort_daily_ret_h{hs}.csv")
    nav = (1.0 + daily[["LS","LV"]].fillna(0.0)).cumprod()
    nav.to_csv(os.path.join(RESULTS, f"step4_longshort_nav_h{hs}.csv"))
    U.plot_series(nav, os.path.join(RESULTS, f"step4_longshort_nav_h{hs}.png"),
                  title=f"{daily.columns.name or 'LS'} 多空净值 (h={hs}d)")
    return out


# ----------------------------------------------------------------------------
# Step 5: 第五层 —— 样本外（Train/Valid/Test）+ Walk-forward
# ----------------------------------------------------------------------------
def step5_oos(factor_long, fwd1_wide, ic_store, h):
    log.info("=" * 70)
    log.info("Step 5 | 第五层：样本外 OOS（三段切分 + Walk-forward）")
    d_rank_full = ic_store[h]["rank"]
    d_pear_full = ic_store[h]["pearson"]

    # 三段切分
    rows = []
    for split, (s, e) in OOS_SPLITS.items():
        mask = (d_rank_full.index >= pd.Timestamp(s)) & (d_rank_full.index <= pd.Timestamp(e))
        sub_r = d_rank_full[mask]
        sub_p = d_pear_full[mask]
        if len(sub_r) == 0:
            continue
        n = len(sub_r)
        icir = sub_r.mean() / sub_r.std() if sub_r.std() > 0 else np.nan
        ic_t = sub_r.mean() / (sub_r.std() / np.sqrt(n)) if sub_r.std() > 0 else np.nan
        rows.append({
            "split": split, "start": s, "end": e, "N_days": n,
            "RankIC": float(sub_r.mean()), "PearsonIC": float(sub_p.mean()),
            "ICIR": float(icir), "IC_t": float(ic_t),
            "P_IC_gt_0": float((sub_r > 0).mean()),
        })
        tag = "IS" if split.startswith("Train") else "OOS"
        log.info(f"  {split:<10} {s}~{e}: RankIC={sub_r.mean():+.4f}  ICIR={icir:+.3f}  "
                 f"IC_t={ic_t:.2f}  P(IC>0)={float((sub_r>0).mean()):.2%}  [{tag}]")
    oos_df = pd.DataFrame(rows)
    save_csv(oos_df, "step5_oos_splits.csv")

    # Walk-forward：train 5y -> test 1y
    wf_rows = []
    label_wide = U.forward_cumprod_ret(fwd1_wide, h)
    label_long = label_wide.stack().rename("LABEL").reset_index().set_index(["datetime", "instrument"])
    fac_wide = U.to_wide(factor_long)
    for y in range(WF_START_YEAR, WF_END_YEAR + 1):
        tr_s, tr_e = f"{y-6}-01-01", f"{y-1}-12-31"
        te_s, te_e = f"{y}-01-01", f"{y}-12-31"
        if pd.Timestamp(te_e) > pd.Timestamp(END):
            te_e = END
        mask = (label_long.index.get_level_values(0) >= pd.Timestamp(te_s)) & \
               (label_long.index.get_level_values(0) <= pd.Timestamp(te_e))
        sub_lab = label_long[mask]
        fac_sub = factor_long.reindex(sub_lab.index)
        _, d_r = U.calc_daily_ic(fac_sub, sub_lab)
        if len(d_r) == 0:
            continue
        n = len(d_r)
        icir = d_r.mean() / d_r.std() if d_r.std() > 0 else np.nan
        # ic_t = d_r.mean() / (d_r.std() / np.sqrt(n)) if d_r.std() > 0 else np.nan
        ic_t = U.cal_t(d_r) if d_r.std() > 0 else np.nan
        wf_rows.append({"train": f"{tr_s}~{tr_e}", "test": f"{te_s}~{te_e}",
                        "N_days": n, "RankIC": float(d_r.mean()),
                        "ICIR": float(icir), "IC_t": float(ic_t)})
        log.info(f"  WF test {te_s}~{te_e}: RankIC={d_r.mean():+.4f}  ICIR={icir:+.3f}  IC_t={ic_t:.2f}")
    wf_df = pd.DataFrame(wf_rows)
    save_csv(wf_df, "step5_walkforward.csv")

    # 绘制 walk-forward IC
    if len(wf_df):
        U.plot_bar(wf_df.set_index("test")["RankIC"], os.path.join(RESULTS, "step5_walkforward_ic.png"),
                   title=f"Walk-forward 样本外 Rank IC (h={h}d)", rot=45)
    return oos_df, wf_df


# ----------------------------------------------------------------------------
# Step 6: 第六层 —— 稳定性（逐年 IC / IC>0 比例 / t 值）
# ----------------------------------------------------------------------------
def step6_stability(ic_store, h):
    log.info("=" * 70)
    log.info("Step 6 | 第六层：稳定性（逐年 IC）")
    d_rank = ic_store[h]["rank"]
    d_pear = ic_store[h]["pearson"]
    yr = d_rank.index.year
    rows = []
    for y, g in d_rank.groupby(yr):
        sub = g.dropna()
        if len(sub) == 0:
            continue
        n = len(sub)
        icir = sub.mean() / sub.std() if sub.std() > 0 else np.nan
        ic_t = sub.mean() / (sub.std() / np.sqrt(n)) if sub.std() > 0 else np.nan
        rows.append({"year": int(y), "N_days": n, "RankIC": float(sub.mean()),
                     "PearsonIC": float(d_pear.loc[sub.index].mean()),
                     "ICIR": float(icir), "IC_t": float(ic_t),
                     "P_IC_gt_0": float((sub > 0).mean())})
        log.info(f"  {y}: RankIC={sub.mean():+.4f}  ICIR={icir:+.3f}  "
                 f"P(IC>0)={float((sub>0).mean()):.2%}")
    yearly = pd.DataFrame(rows)
    save_csv(yearly, "step6_yearly_ic.csv")
    if len(yearly):
        U.plot_bar(yearly.set_index("year")["RankIC"], os.path.join(RESULTS, "step6_yearly_ic.png"),
                   title=f"逐年 Rank IC (h={h}d)", rot=0)
    return yearly


# ----------------------------------------------------------------------------
# Step 7: 衰减分析（horizon = 1,2,3,5,10,20）
# ----------------------------------------------------------------------------
def step7_decay(factor_long, fwd1_wide, hs=DECAY_HS):
    log.info("=" * 70)
    log.info("Step 7 | 衰减分析（IC 随预测周期 h 的衰减）")
    rows = []
    for h in hs:
        label_wide = U.forward_cumprod_ret(fwd1_wide, h)
        label_long = label_wide.stack().rename("LABEL").reset_index().set_index(["datetime", "instrument"])
        d_pear, d_rank = U.calc_daily_ic(factor_long, label_long)
        s = U.summarize_ic(d_rank, d_pear)
        s.update({"horizon": h})
        rows.append(s)
        log.info(f"  h={h:>2}d | RankIC={s['RankIC']:+.4f}  ICIR={s['ICIR']:+.3f}  "
                 f"PearsonIC={s['Pearson_IC']:+.4f}  IC_t={s['IC_t']:.2f}")
    decay = pd.DataFrame(rows)
    save_csv(decay[["horizon", "RankIC", "Pearson_IC", "ICIR", "IC_t", "P_IC_gt_0", "N_days"]],
              "step7_decay_ic.csv")
    U.plot_series(decay.set_index("horizon")[["RankIC", "Pearson_IC"]],
                  os.path.join(RESULTS, "step7_decay_ic.png"),
                  title="IC 随预测周期衰减", ylabel="IC", legend=True)
    return decay


# ----------------------------------------------------------------------------
# Step 8: IC 的条件依赖（市场环境 / 波动 / 成交量 状态）
# ----------------------------------------------------------------------------
def step8_regime(factor_long, fwd1_wide, bench, controls_long, hs):
    log.info("=" * 70)
    log.info("Step 8 | IC 的条件依赖（按市场状态分样本）")
    h = hs[0]
    label_wide = U.forward_cumprod_ret(fwd1_wide, h)
    label_long = label_wide.stack().rename("LABEL").reset_index().set_index(["datetime", "instrument"])
    d_pear, d_rank = U.calc_daily_ic(factor_long, label_long)
    dates = d_rank.index

    # 趋势状态：基准 60 日收益
    bench_price = (1.0 + bench.reindex(dates).fillna(0.0)).cumprod()
    bench_60 = bench_price / bench_price.shift(60) - 1
    trend = pd.Series(index=dates, dtype=object)
    trend[bench_60 > 0.08] = "bull"
    trend[bench_60 < -0.08] = "bear"
    trend[bench_60.isna()] = np.nan
    trend = trend.fillna("oscillation")

    # 波动状态：基准 20 日已实现波动中位数切分
    bench_vol = bench.reindex(dates).fillna(0.0).rolling(20).std()
    vol_med = bench_vol.median()
    vol_state = np.where(bench_vol >= vol_med, "high_vol", "low_vol")

    # # 成交量状态：市场日均成交量 20 日均值中位数切分（复用 step0 预取的 $volume）
    # mkt_vol = controls_long["LIQ"].copy()  # 已含量价信息；改用原始成交量更直观
    # vol_panel = U.to_wide(controls_long["$volume"], "$volume")
    # vol_panel = vol_panel.reindex(dates)
    # mvol = vol_panel.mean(axis=1).rolling(20).mean()
    # mvol_med = mvol.median()
    # vol_state2 = np.where(mvol >= mvol_med, "high_vol", "low_vol")

    regimes = {
        "trend(bull/bear/osc)": trend.values,
        "vol(high/low)": vol_state,
    }
    rows = []
    for rname, rvals in regimes.items():
        rvals = pd.Series(rvals, index=dates)
        for state in sorted(set(rvals.dropna())):
            m = rvals == state
            sub_r = d_rank[m]
            sub_p = d_pear[m]
            if len(sub_r) < 5:
                continue
            rows.append({
                "regime_type": rname, "state": state, "N_days": len(sub_r),
                "RankIC": float(sub_r.mean()), "PearsonIC": float(sub_p.mean()),
                "ICIR": float(sub_r.mean() / sub_r.std()) if sub_r.std() > 0 else np.nan,
            })
            log.info(f"  {rname:<22} {state:<10}: RankIC={sub_r.mean():+.4f}  "
                     f"PearsonIC={sub_p.mean():+.4f}  N={len(sub_r)}")
    regime_df = pd.DataFrame(rows)
    save_csv(regime_df, "step8_regime_ic.csv")

    # 趋势三态可视化
    piv = regime_df[regime_df.regime_type == "trend(bull/bear/osc)"].pivot(
        index="state", columns="regime_type", values="RankIC")
    U.plot_bar(piv.iloc[:, 0] if piv.shape[1] else piv, os.path.join(RESULTS, "step8_regime_ic.png"),
               title="不同市场趋势下的 Rank IC", rot=0)
    return regime_df


# ----------------------------------------------------------------------------
# Step 9: 交易成本（NetReturn = Gross - Turnover * Cost）
# ----------------------------------------------------------------------------
def step9_cost(factor_long, fwd1_wide, daily_ls, h, bench):
    log.info("=" * 70)
    log.info("Step 9 | 交易成本（NetReturn = Gross - Turnover × Cost）")
    fac_wide = U.to_wide(factor_long)
    dates = list(fwd1_wide.index)
    # 复用矢量化的分组矩阵（避免逐日 qcut 与构造 membership 字典）
    G, reb_pos, reb_sorted = U.group_assignments(fac_wide, h, N_GROUPS)
    n_reb = len(reb_pos)
    turnover_daily = pd.Series(0.0, index=dates)
    turnover_daily_long = pd.Series(0.0, index=dates)
    avg_to = 0.0
    to_long = np.array([], dtype=float)
    to_short = np.array([], dtype=float)
    to_total = np.array([], dtype=float)
    if n_reb >= 2:
        # 布尔矩阵：每个换仓日的最高组(=Gn)/最低组(=G1)成分
        top = (G[reb_pos] == N_GROUPS - 1)      # (n_reb, n_inst)
        bot = (G[reb_pos] == 0)
        top_cnt = top.sum(axis=1)
        bot_cnt = bot.sum(axis=1)
        ov_top = (top[1:] & top[:-1]).sum(axis=1)
        ov_bot = (bot[1:] & bot[:-1]).sum(axis=1)
        to_long = 1 - ov_top / np.maximum(top_cnt[1:], 1)
        to_short = 1 - ov_bot / np.maximum(bot_cnt[1:], 1)
        # 多空双腿各自单边换手；扣费按双腿合计成交金额计
        # （合计单边换手 = to_long + to_short，往返成本 c 对每条腿各收一次）
        to_total = to_long + to_short
        for k in range(1, n_reb):
            turnover_daily.loc[reb_sorted[k]] = float(to_total[k - 1])
            turnover_daily_long.loc[reb_sorted[k]] = float(to_long[k - 1])
        avg_to = float(to_total.mean())
        avg_to_long = float(to_long.mean())

    log.info(f"  平均单次换仓多空合计单边换手率 = {avg_to:.2%}（多头 {to_long.mean():.2%} / 空头 {to_short.mean():.2%}）")

    gross = daily_ls["LS"].astype(float).fillna(0.0)
    gross_ann = U.geom_metrics(gross)["ann_return"]     # 只算一次
    gross_lv = daily_ls["LV"].astype(float).fillna(0.0)
    gross_ann_lv = U.geom_metrics(gross_lv)["ann_return"]     # 只算一次
    rows = []
    for c in COST_RATES:
        net = gross - turnover_daily * c
        net_lv = gross_lv - turnover_daily_long * c
        m = U.geom_metrics(net)
        m_lv = U.geom_metrics(net_lv)
        rows.append({"cost_roundtrip": c, "turnover_avg": avg_to,
                     "gross_ann_return": gross_ann,
                     "net_ann_return": m["ann_return"], "net_sharpe": m["sharpe"],
                     "net_max_drawdown": m["max_drawdown"]})
        rows.append({"cost_roundtrip_long": c, "turnover_avg_long": avg_to_long,
                "gross_ann_return_long": gross_ann_lv,
                "net_ann_return_long": m_lv["ann_return"], "net_sharpe_long": m_lv["sharpe"],
                "net_max_drawdown_long": m_lv["max_drawdown"]})
        log.info(f"  cost={c:.2%} | 毛年化={gross_ann:+.2%}  "
                 f"净年化={m['ann_return']:+.2%}  Sharpe={m['sharpe']:.2f}  "
                 f"MDD={m['max_drawdown']:+.2%}")
        log.info(f"  cost={c:.2%} | 毛年化long={gross_ann_lv:+.2%}  "
            f"净年化_long={m_lv['ann_return']:+.2%}  Sharpe_long={m_lv['sharpe']:.2f}  "
            f"MDD_long={m_lv['max_drawdown']:+.2%}")
    cost_df = pd.DataFrame(rows)
    save_csv(cost_df, "step9_transaction_cost.csv")

    # 复用 qlib long_short_backtest 做含成本多空回测交叉验证（单一成本档）
    # 注：当前 qlib 版本的 long_short_backtest 与 Exchange 签名不兼容，作为可选交叉验证，
    #     失败时不中断主流程（主结论已由上方 turnover×cost 法给出）。
    # c0 = COST_RATES[1]
    # try:
    #     ls_res = U.long_short_with_cost(fac_wide, topk=50, open_cost=c0 / 2, close_cost=c0 / 2)
    #     ls_net = ls_res["long_short"]
    #     m = U.geom_metrics(ls_net)
    #     log.info(f"  [交叉验证] qlib long_short_backtest(cost={c0:.2%}): 年化={m['ann_return']:+.2%}  "
    #              f"Sharpe={m['sharpe']:.2f}  MDD={m['max_drawdown']:+.2%}")
    #     ls_net.to_frame("long_short_net").to_csv(os.path.join(RESULTS, "step9_qlib_ls_net.csv"))
    # except Exception as e:
    #     log.info(f"  [交叉验证] qlib long_short_backtest 因环境兼容性问题跳过：{type(e).__name__}: {e}")
    return cost_df, avg_to


# ----------------------------------------------------------------------------
# Step 10: Alpha Scorecard + 等级评定
# ----------------------------------------------------------------------------
def step10_scorecard(factor_name, ic_summary, mono, neutral_df, oos_df, wf_df,
                     yearly, decay, regime_df, geo, cost_df, avg_to):
    log.info("=" * 70)
    log.info("Step 10 | Alpha Scorecard 汇总与等级评定")
    h0 = ic_summary.iloc[0]["horizon"]
    ic0 = ic_summary.iloc[0]

    def oos_val(col, split="Test(OOS)"):
        r = oos_df[oos_df.split == split]
        return float(r[col].values[0]) if len(r) else np.nan

    decay_h1 = decay[decay.horizon == 1].iloc[0]
    raw_neu = neutral_df[neutral_df.neutralization == "raw"]["RankIC"].values[0]
    all_neu = neutral_df[neutral_df.neutralization == "size+beta"]["RankIC"].values[0]
    size_neu = neutral_df[neutral_df.neutralization == "size"]["RankIC"].values[0]
    ind_neu = neutral_df[neutral_df.neutralization == "industry"]["RankIC"].values[0]
    ind_size_neu = neutral_df[neutral_df.neutralization == "industry+size"]["RankIC"].values[0]

    # 市场状态 IC（取趋势三态平均）
    trend = regime_df[regime_df.regime_type == "trend(bull/bear/osc)"]
    regime_mean = float(trend["RankIC"].mean()) if len(trend) else np.nan

    cost_row = cost_df.iloc[1]  # 0.3% 档
    net_ann = float(cost_row["net_ann_return"])

    card = [
        ("Rank IC", round(ic0["RankIC"], 4), "核心"),
        ("Pearson IC", round(ic0["Pearson_IC"], 4), "辅助"),
        ("ICIR", round(ic0["ICIR"], 3), "核心"),
        ("Robust_ICIR", round(ic0["Robust_ICIR"], 3), "核心"),
        ("IC t-value", round(ic0["IC_t"], 2), "核心"),
        ("IC > 0 比例", round(ic0["P_IC_gt_0"], 3), "核心"),
        ("Q1->QH 单调性", round(float(mono), 4), "核心"),
        ("QH-Q1 收益(年化)", round(float(geo["ann_return"]), 4), "核心"),
        ("QH-AVG 收益(年化)", round(float(geo["ann_return_lv"]), 4), "核心"),
        ("OOS RankIC", round(oos_val("RankIC"), 4), "**最重要**"),
        ("OOS ICIR", round(oos_val("ICIR"), 3), "**最重要**"),
        ("IC 衰减(h=1->20)", f"{decay_h1['RankIC']:.4f}->{decay[decay.horizon==20].iloc[0]['RankIC']:.4f}", "核心"),
        ("不同市场状态 IC(均值)", round(regime_mean, 4), "稳定性"),
        ("行业中性后 IC", "N/A(无标签)" if np.isnan(ind_neu) else round(ind_neu, 4), "独立性"),
        ("Indestry+Size 中性后 IC", round(ind_size_neu, ndigits=4), "独立性"),
        ("Winsorize 后 IC", "同上(原始已去极值)", "鲁棒性"),
        ("多空合计单边换手率", round(avg_to, 4), "交易性"),
        ("成本后净年化(0.3%)", round(net_ann, 4), "**最终**"),
    ]
    card_df = pd.DataFrame(card, columns=["指标", "评价", "类别"])
    save_csv(card_df, "step10_alpha_scorecard.csv")

    # 等级评定
    grade = "D"
    reasons = []
    if ic0["RankIC"] > 0 and ic0["Pearson_IC"] > 0 and ic0["ICIR"] > 0.3 and mono > 0.8 \
       and oos_val("RankIC") > 0 and net_ann > 0 and all_neu > 0:
        grade = "A"
        reasons.append("RankIC>0,PearsonIC>0,ICIR>0.3,分组单调,OOS>0,成本后仍盈利,中性化后仍有效")
    elif ic0["RankIC"] > 0 and oos_val("RankIC") > 0:
        if ic0["Pearson_IC"] <= 0:
            grade = "B"
            reasons.append("RankIC>0 但 PearsonIC<=0（非线性/极端值，需进一步研究）；OOS 有效")
        else:
            grade = "B"
            reasons.append("RankIC>0,PearsonIC 不稳定,OOS 有效")
    elif ic0["RankIC"] > 0 and net_ann <= 0:
        grade = "C"
        reasons.append("IC 统计有效但成本后收益消失（换手过高）")
    else:
        grade = "D"
        reasons.append("IS 高 / OOS≈0 或 IC 主要来自少数年份（伪 Alpha）")

    log.info("  Alpha Scorecard:")
    for _, r in card_df.iterrows():
        log.info(f"    {r['指标']:<22} {r['评价']:<18} [{r['类别']}]")
    log.info(f"  >>> 综合等级：{grade}  |  依据：{'; '.join(reasons)}")
    with open(os.path.join(RESULTS, "step10_alpha_scorecard.md"), "w", encoding="utf-8") as f:
        f.write(f"# Alpha Scorecard — {factor_name}\n\n")
        f.write(f"综合等级：**{grade}**\n\n依据：{'; '.join(reasons)}\n\n")
        f.write("| 指标 | 评价 | 类别 |\n")
        f.write("| --- | --- | --- |\n")
        for _, r in card_df.iterrows():
            f.write(f"| {r['指标']} | {r['评价']} | {r['类别']} |\n")
    return card_df, grade


# ----------------------------------------------------------------------------
# Step 11: 复用 qlib.report 原生图（IC / 分组 / 自相关 / 换手）
# ----------------------------------------------------------------------------
def step11_qlib_report(factor_long, fwd1_long):
    log.info("=" * 70)
    log.info("Step 11 | 复用 qlib.contrib.report 生成原生分析图（plotly HTML）")
    try:
        from qlib.contrib.report.analysis_model import model_performance_graph
    except Exception as e:
        log.info(f"  跳过：无法导入 analysis_model ({e})")
        return
    score = factor_long.copy()
    score.columns = ["score"]
    label = fwd1_long.copy()
    label.columns = ["label"]
    pred_label = pd.concat([score, label], axis=1)
    pred_label.index = pred_label.index.reorder_levels(["instrument", "datetime"])
    pred_label = pred_label.sort_index()
    # try:
    #     figs = model_performance_graph(pred_label, N=10, lag=1, reverse=False,
    #                                    graph_names=["group_return", "pred_ic", "pred_autocorr", "pred_turnover"],
    #                                    show_notebook=False)
    #     for i, fig in enumerate(figs):
    #         fig.write_html(os.path.join(RESULTS, f"step11_qlib_report_{i}.html"))
    #     log.info(f"  生成 {len(figs)} 张原生图（group_return/pred_ic/autocorr/turnover）")
    # except Exception as e:
    #     log.info(f"  生成图失败：{e}")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def _timed(label, fn, *args, **kwargs):
    """包裹一步，记录耗时，便于观察各函数回测效率。"""
    t0 = time.time()
    out = fn(*args, **kwargs)
    log.info(f"  [用时] {label}: {time.time() - t0:.2f}s")
    return out


def main():
    U.qlib_init()
    factor_name, factor_expr, factor_long, fwd1_long, fwd1_wide, bench, controls_wide, controls_long = step0_load(factor_override = FACTOR_OVERRIDE)

    # Step 1: 中性化（在 IC 分析之前），返回下游实际使用的因子
    neutral_df, factor_long = _timed("Step1 中性化",
        step1_neutral, factor_long, fwd1_wide, controls_wide, [PRIMARY_H])

    ic_store, ic_summary = _timed("Step2 IC",
        step2_ic, factor_long, fwd1_wide, [PRIMARY_H])
    nav_h1, daily_h1, group_ret_h1, mono_h1 = _timed("Step3 分组",
        step3_groups, factor_long, fwd1_wide, PRIMARY_H)
    geo = _timed("Step4 多空",
        step4_longshort, daily_h1, bench, PRIMARY_H)
    oos_df, wf_df = _timed("Step5 OOS",
        step5_oos, factor_long, fwd1_wide, ic_store, PRIMARY_H)
    yearly = _timed("Step6 稳定性",
        step6_stability, ic_store, PRIMARY_H)
    decay = _timed("Step7 衰减",
        step7_decay, factor_long, fwd1_wide)
    regime_df = _timed("Step8 市场状态",
        step8_regime, factor_long, fwd1_wide, bench, controls_long, [PRIMARY_H])
    cost_df, avg_to = _timed("Step9 成本",
        step9_cost, factor_long, fwd1_wide, daily_h1, PRIMARY_H, bench)

    # 重新读取中性化结果用于 scorecard
    neutral_df = pd.read_csv(os.path.join(RESULTS, "step1_neutral_ic.csv"))
    card, grade = _timed("Step10 Scorecard",
        step10_scorecard,
        factor_name, ic_summary, mono_h1, neutral_df, oos_df, wf_df,
        yearly, decay, regime_df, geo, cost_df, avg_to,
    )
    _timed("Step11 qlib报告",
        step11_qlib_report, factor_long, fwd1_long)

    log.info("=" * 70)
    log.info(f"全部完成。因子={factor_name}  等级={grade}  结果目录：{RESULTS}")


if __name__ == "__main__":
    main()
