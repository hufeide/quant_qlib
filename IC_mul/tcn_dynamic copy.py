"""动态窗口 TCN 训练路径（不复制时间窗口，省内存）。

对照 workflow_tcn.build_dataset 把窗口物理摊平成 (N, F*n_days) 的写法，
这里保持原始长表 (datetime, instrument) × (因子+行业onehot) 的 (T, N, F) 形态，
在 torch Dataset.__getitem__ 里按「同一只股票」切片出 [L, F+K] 窗口，
模型只吃 [B, F+K, L]。内存峰值从 N*(F+K)*n_days 降到约 N*(F+K)（~0.45 GB），
n_days=100 也能在小内存机器上跑。

产出的 pred.pkl / label.pkl 与 workflow_tcn 的 qlib 回测完全兼容
（index=(datetime, instrument)，列 "score" / "LABEL0"）。
"""
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from workflow_tcn import build_raw_features, MARKET, LABEL_EXPR, METRICS_CSV

log = logging.getLogger("tcn_dynamic")


# ---------------------------------------------------------------------------
# 模型：因果膨胀卷积 TCN（输入 [B, C, T]，输出每个样本一个标量回归值）
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    def __init__(self, chans, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # 仅左填充 -> 因果
        self.conv = nn.Conv1d(chans, chans, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, chans, T]
        xx = F.pad(x, (self.pad, 0))          # 左填充，保证输出 t 只依赖 <=t
        xx = self.conv(xx)                    # [B, chans, T]
        xx = self.relu(xx)
        xx = self.drop(xx)
        return xx + x                         # 残差（通道数一致，长度一致）


class CausalTCN(nn.Module):
    def __init__(self, in_channels, n_chans, kernel_size, num_layers, dropout):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, n_chans, 1)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(n_chans, kernel_size, 2 ** i, dropout)
             for i in range(num_layers)]
        )
        self.head = nn.Conv1d(n_chans, 1, 1)

    def forward(self, x):  # x: [B, C, T]
        x = self.in_proj(x)                   # [B, n_chans, T]
        for b in self.blocks:
            x = b(x)
        x = self.head(x)                      # [B, 1, T]
        return x[:, 0, -1]                    # 取最后一个时间步（因果预测）-> [B]


# ---------------------------------------------------------------------------
# 数据：按股票切片滑动窗口（不预先复制整张窗口矩阵）
# ---------------------------------------------------------------------------
def _segment_windows(feat_raw, label_raw, seg, L):
    """返回 seg 区间内「合法窗口」的 [S, L] 原始行号矩阵与对应结尾 (datetime, instrument) 索引。

    数据是 (datetime, instrument) 时间优先排序的，同一只股票的相邻交易日并不在相邻行，
    因此先按 (instrument, datetime) 重排，使同股票连续 L 行 = 该股票的 L 个连续交易日；
    再对每个合法结尾取这 L 行的**原始行号**（切片，不复制整张特征矩阵）。
    窗口特征 = X[window_rows[k]]（[L, C]），在 Dataset 中按需取，不在内存里展开全量窗口。
    """
    N = len(feat_raw)
    inst = pd.factorize(feat_raw.index.get_level_values(1))[0]
    datetimes = feat_raw.index.get_level_values(0).to_numpy()
    y = label_raw["LABEL0"].to_numpy(dtype=np.float32)

    # 按 (instrument, datetime) 排序：同股票交易日连续
    order = np.lexsort((datetimes, inst))
    inst_s = inst[order]
    dt_s = datetimes[order]
    y_s = y[order]
    orig = np.arange(N)[order]                     # 排序位置 -> 原始行号

    same = np.empty(N, dtype=bool)
    same[1:] = inst_s[1:] == inst_s[:-1]
    same[0] = False
    newrun = ~same
    starts = np.where(newrun, np.arange(N), 0)
    starts[0] = 0
    last_start = np.maximum.accumulate(starts)
    pos = np.arange(N) - last_start                # 该股票时间线内的 0-based 位置
    valid_run = pos >= (L - 1)

    seg_start = np.datetime64(pd.Timestamp(seg[0]))
    seg_end = np.datetime64(pd.Timestamp(seg[1]))
    in_seg = (dt_s >= seg_start) & (dt_s <= seg_end)
    label_ok = ~np.isnan(y_s)
    mask = valid_run & in_seg & label_ok

    valid_p = np.nonzero(mask)[0].astype(np.int64)
    # 每个样本窗口 = 排序位置 [p-L+1 .. p] 对应的原始行号
    lg = np.arange(L - 1, -1, -1)                  # L-1 ... 0
    pos_grid = valid_p[:, None] - lg[None, :]      # [S, L]
    window_rows = orig[pos_grid].astype(np.int32)  # [S, L] 原始行号
    end_orig = orig[valid_p]
    return window_rows, feat_raw.index[end_orig]


class PanelWindowDataset(Dataset):
    """按样本存储的 L 个原始行号切片出 [L, C] 窗口，转置为 [C, L]（TCN 通道优先）。"""

    def __init__(self, X, y, window_rows, L):
        self.X = np.ascontiguousarray(X, dtype=np.float32)   # [N, C]
        self.y = np.ascontiguousarray(y, dtype=np.float32)   # [N]
        self.window_rows = np.asarray(window_rows, dtype=np.int32)  # [S, L]
        self.L = L

    def __len__(self):
        return len(self.window_rows)

    def __getitem__(self, k):
        rows = self.window_rows[k]                # [L] 原始行号
        w = self.X[rows]                          # [L, C]
        return w.T.copy(), float(self.y[rows[-1]])  # [C, L], 结尾日标签


def _make_loader(X, y, window_rows, L, batch_size, shuffle):
    ds = PanelWindowDataset(X, y, window_rows, L)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=False)


# ---------------------------------------------------------------------------
# 训练 + 预测
# ---------------------------------------------------------------------------
def train_dynamic(feat_raw, label_raw, n_feat, args, segments):
    """动态窗口路径核心训练：输入已构建的原始长表，返回 (pred_df, label_df, model)。

    pred_df / label_df 的 index=(datetime, instrument)，列 "score" / "LABEL0"，
    与 workflow_tcn 的 qlib 回测（PortAnaRecord / SigAnaRecord）完全兼容。
    """
    L = args.n_days
    log.info(f"[dynamic] 原始长表 feat_raw={feat_raw.shape}（N×C，C=通道数={n_feat}），"
             f"未摊平窗口；回看 L={L} 天。内存峰值 ~ {feat_raw.shape[0]*n_feat*4/1e6:.0f} MB")

    X = feat_raw.values.astype(np.float32)
    # TCN 对 NaN 不鲁棒（前向传播成 NaN），故特征像 qlib 路径一样零值填充
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = label_raw["LABEL0"].to_numpy(dtype=np.float32)
    C = X.shape[1]

    # 特征标准化（仅用训练区间统计，避免标签泄漏）；常数通道保护
    tr_s, tr_e = segments["train"]
    tr_mask = ((feat_raw.index.get_level_values(0) >= pd.Timestamp(tr_s)) &
               (feat_raw.index.get_level_values(0) <= pd.Timestamp(tr_e)))
    mu = X[tr_mask].mean(axis=0)
    sd = X[tr_mask].std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    # 各 segment 的样本窗口（[S, L] 原始行号）
    tr_w, tr_idx = _segment_windows(feat_raw, label_raw, segments["train"], L)
    va_w, va_idx = _segment_windows(feat_raw, label_raw, segments["valid"], L)
    te_w, te_idx = _segment_windows(feat_raw, label_raw, segments["test"], L)
    log.info(f"[dynamic] 样本数 -> train={len(tr_w)} valid={len(va_w)} test={len(te_w)}")

    use_cuda = args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda", args.gpu) if use_cuda else torch.device("cpu")
    log.info(f"[dynamic] 设备={device}")

    model = CausalTCN(C, args.n_chans, args.kernel_size, args.num_layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    tr_loader = _make_loader(X, y, tr_w, L, args.batch_size, shuffle=True)
    va_loader = _make_loader(X, y, va_w, L, args.batch_size, shuffle=False)

    best_val, wait, best_state = float("inf"), 0, None
    for epoch in range(1, args.rounds + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= max(1, len(tr_w))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                yb = yb.to(device)
                va_loss += criterion(model(xb.to(device)), yb).item() * len(yb)
        va_loss /= max(1, len(va_w))

        log.info(f"[dynamic] epoch {epoch:03d} | train_mse={tr_loss:.6f} valid_mse={va_loss:.6f}")
        if va_loss < best_val - 1e-8:
            best_val, wait, best_state = va_loss, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.early_stop:
                log.info(f"[dynamic] 早停：验证集 {args.early_stop} 轮无改善")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"[dynamic] 载入最佳验证权重（valid_mse={best_val:.6f}）")

    # 预测测试段
    te_loader = _make_loader(X, y, te_w, L, args.batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    pred_df = pd.DataFrame({"score": preds}, index=te_idx)
    label_df = pd.DataFrame({"LABEL0": y[te_w[:, -1]]}, index=te_idx)
    log.info(f"[dynamic] 测试预测 {pred_df.shape}，已生成 pred.pkl / label.pkl 兼容格式")
    return pred_df, label_df, model


def run_dynamic(pkl_path, segments, args):
    """便捷封装：先 build_raw_features，再 train_dynamic。"""
    feat_raw, label_raw, n_feat = build_raw_features(
        pkl_path, segments, MARKET, LABEL_EXPR, METRICS_CSV,
        rank_ic_min=args.rank_ic_min, topk=args.factor_topk,
        factor_filter=not args.no_factor_filter, industry_mode=args.industry,
    )
    return train_dynamic(feat_raw, label_raw, n_feat, args, segments)
