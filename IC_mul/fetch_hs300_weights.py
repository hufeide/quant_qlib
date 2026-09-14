#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
抓取沪深300（csi300）成分股权重，落盘为 qlib 可读的 weights 文件。

数据来源与限制
--------------
- 真实【最新一期】成分股权重：akshare.index_stock_cons_weight_csindex(symbol="000300")
  （底层 csindex.com.cn 官方披露，免费、无需 token）。akshare 只返回最新一期，
  拿不到 2017 年以来的历史权重。
- 真实【历史成分（成员）】：baostock.query_hs300_stocks(date=...)，
  可免费按任意历史日期取沪深300成分（解决“哪些股票当时在指数里”）。

因此本脚本产出的是“按季度采样 + 向后(前)补充(as-of)”的权重面板：
  - 成员：每个季度用 baostock 取当时的真实沪深300成分（时间维度真实）。
  - 权重值：用 csindex 最新一期的真实权重做锚；对“当时在指数、但已被调出”的
    股票用均值回退（weight_mode="current”），或简单地等权（weight_mode="equal"）。
    —— 历史权重值本环境无法免费获得，这是已知的近似；若需要真实历史权重，
       可装 tushare 并配置 token 后调用 pro.index_weight()。

落盘格式（CSV）：date,instrument,weight
    weight 为【小数权重】（= 披露百分比 / 100），如 0.00433 表示 0.433%。
    每个季度一行快照；策略 BenchmarkTiltStrategy 在回测时会取“≤交易日的最近一期”
    做前滚填充（as-of），即用户所说的“向后补充”。

用法
----
    # 默认：2017-01-01 起，按季度抓取，权重用最新真实权重锚定
    python fetch_hs300_weights.py

    # 自定义区间 / 等权模式
    python fetch_hs300_weights.py --start 2017-01-01 --end 2026-09-14 --mode equal

    # 仅更新最新一期（单快照，不按季度展开）
    python fetch_hs300_weights.py --single
"""

import argparse
import os
import sys
import time

import pandas as pd

# qlib 代码前缀映射（与 qlib 一致）：沪市 SH、深市 SZ
EXCHANGE_MAP = {"sh": "SH", "sz": "SZ", "bj": "BJ","上海证券交易所":"SH", "深圳证券交易所": "SZ", "北京证券交易所": "BJ"}


# ---------------------------------------------------------------------------
# 1) 最新一期真实权重（csindex / akshare）
# ---------------------------------------------------------------------------
def fetch_current_weights():
    """返回 (date_str, Series[instrument -> 小数权重])，最新一期的真实 hs300 权重。"""
    import akshare as ak

    df = ak.index_stock_cons_weight_csindex(symbol="000300")
    date = pd.to_datetime(df["日期"].iloc[0]).strftime("%Y-%m-%d")
    code = df["成分券代码"].astype(str).str.zfill(6)
    exch = df["交易所"].map(EXCHANGE_MAP).fillna("SH")
    instrument = exch + code
    weight = pd.to_numeric(df["权重"], errors="coerce") / 100.0
    s = pd.Series(weight.values, index=instrument.values).dropna()
    s = s[s > 0]
    return date, (s / s.sum())


# ---------------------------------------------------------------------------
# 2) 历史成分（baostock，按日期）
# ---------------------------------------------------------------------------
def fetch_hs300_members(date, bs_mod):
    """返回某历史日期的沪深300成分（qlib 代码列表）。date 可为 'YYYY-MM-DD'。"""
    rs = bs_mod.query_hs300_stocks(date=date)
    codes = []
    while rs.next():
        row = rs.get_row_data()
        # row: [snap_date, 'sh.600000', '浦发银行', ...]
        bcode = row[1]  # e.g. sh.600000
        prefix, num = bcode.split(".", 1)
        qcode = EXCHANGE_MAP.get(prefix.lower(), "SH") + num
        codes.append(qcode)
    return codes


# ---------------------------------------------------------------------------
# 3) 构建季度面板
# ---------------------------------------------------------------------------
def build_quarterly_panel(start_date, end_date, weight_mode="current",
                          out_path=None, quiet=False):
    import baostock as bs

    if out_path is None:
        out_path = os.path.expanduser(
            "~/.qlib/qlib_data/cn_data/instruments/csi300/weights_day.txt"
        )

    # 最新真实权重（锚）
    cs_date, cur_w = fetch_current_weights()
    if not quiet:
        print(f">>> 最新真实权重日期 {cs_date}，共 {len(cur_w)} 只（锚定用）")

    # 季度末日期序列
    qends = pd.period_range(start=pd.Timestamp(start_date),
                            end=pd.Timestamp(end_date), freq="Q").to_timestamp(how="end")
    qends = [d.strftime("%Y-%m-%d") for d in qends]
    if not quiet:
        print(f">>> 季度快照数: {len(qends)} （{qends[0]} ~ {qends[-1]}）")

    bs.login()
    try:
        rows = []  # (date, instrument, weight)
        fallback = cur_w.mean() if weight_mode == "current" else None
        for i, qd in enumerate(qends):
            members = fetch_hs300_members(qd, bs)
            if not members:
                if not quiet:
                    print(f"  ! {qd} 取不到成分，跳过")
                continue
            if weight_mode == "equal":
                w = pd.Series(1.0 / len(members), index=members)
            else:  # current：真实权重锚定，缺失用均值回退
                w = pd.Series([cur_w.get(m, fallback) for m in members], index=members)
                w = w / w.sum()
            for m, val in w.items():
                rows.append((qd, m, val))
            if not quiet and (i % 8 == 0 or i == len(qends) - 1):
                print(f"  [{i+1}/{len(qends)}] {qd} 成员 {len(members)} 权重和 {w.sum():.4f}")
            time.sleep(0.15)  # 避免 baostock 限频
    finally:
        bs.logout()

    # 追加最新一期真实权重快照（保证最近区间用真实权重，而非回退）
    for m, val in cur_w.items():
        rows.append((cs_date, m, val))

    out = pd.DataFrame(rows, columns=["date", "instrument", "weight"])
    out = out.sort_values(["date", "instrument"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    if not quiet:
        n_q = out["date"].nunique()
        print(f"<<< 已写出 {len(out)} 行 / {n_q} 期快照 -> {out_path}")
        print(f"    日期范围: {out['date'].min()} ~ {out['date'].max()}")
    return out


# ---------------------------------------------------------------------------
# 4) 仅最新一期（单快照，兼容旧用法）
# ---------------------------------------------------------------------------
def fetch_single_snapshot(out_path=None):
    if out_path is None:
        out_path = os.path.expanduser(
            "~/.qlib/qlib_data/cn_data/instruments/csi300/weights_day.txt"
        )
    cs_date, cur_w = fetch_current_weights()
    out = pd.DataFrame(
        {"date": cs_date, "instrument": cur_w.index, "weight": cur_w.values}
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"<<< 已写出单快照（{cs_date}，{len(out)} 只）-> {out_path}")
    print(f"    权重和: {out['weight'].sum():.4f} (应≈1.0)")
    return out


def main():
    p = argparse.ArgumentParser(description="抓取沪深300权重（季度面板）")
    p.add_argument("--start", default="2017-01-01", help="起始日期（含）")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                   help="结束日期（含），默认今天")
    p.add_argument("--mode", choices=["current", "equal"], default="current",
                   help="权重值模式：current=最新真实权重锚定(退出股均值回退)，"
                        "equal=每季度成分内等权")
    p.add_argument("--single", action="store_true",
                   help="只抓取最新一期（单快照，不按季度展开）")
    p.add_argument("--out", default=None, help="输出路径（默认 weights_day.txt）")
    args = p.parse_args()

    if args.single:
        fetch_single_snapshot(args.out)
    else:
        build_quarterly_panel(args.start, args.end, args.mode, args.out)


if __name__ == "__main__":
    main()
