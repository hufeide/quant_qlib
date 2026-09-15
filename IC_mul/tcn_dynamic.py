"""动态窗口 TCN 训练路径（不复制时间窗口，省内存）。

对照 workflow_tcn.build_dataset 把窗口物理摊平成 (N, F*n_days) 的写法，
这里保持原始长表 (datetime, instrument) × (因子+行业onehot) 的 (T, N, F) 形态，
在 torch Dataset.__getitem__ 里按「同一只股票」动态切片出 [L, F+K] 窗口，
模型只吃 [B, F+K, L]。

核心省内存点：只保存每个样本的「排序后结尾位置」end_pos([S])，不保存完整
[S, L] 的 window_rows。Dataset.__getitem__ 再动态切片
order[p-L+1 : p+1]（排序位置 -> 原始行号）取窗口。内存从 O(S×L) 降到 O(S)
（约 N*C*4 + S*8 字节），n_days=100、数百万样本也能在小内存机器上跑。

产出的 pred.pkl / label.pkl 与 workflow_tcn 的 qlib 回测完全兼容
（index=(datetime, instrument)，列 "score" / "LABEL0"）。
"""
import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm, trange

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
    """返回排序后 order（[N]）与 seg 区间内「合法窗口」的结尾排序位置 end_pos（[S]），
    以及对应结尾 (datetime, instrument) 索引。不保存完整 [S, L] 的 window_rows。

    数据是 (datetime, instrument) 时间优先排序的，同一只股票的相邻交易日并不在相邻行，
    因此先按 (instrument, datetime) 重排，使同股票连续 L 行 = 该股票的 L 个连续交易日；
    只保存每个合法结尾的排序位置 end_pos，Dataset.__getitem__ 再动态切片
    order[p-L+1 : p+1]（排序位置 -> 原始行号）取窗口。内存从 O(S×L) 降到 O(S)。
    """
    N = len(feat_raw)

    # 强制检查 feat / label 完全对齐，否则按 numpy 行位置配对会错位
    if not feat_raw.index.equals(label_raw.index):
        raise ValueError(
            "feat_raw.index 与 label_raw.index 不完全一致，"
            "不能直接按 numpy 行位置配对 X/y，请检查 build_raw_features 的排序。"
        )

    inst = pd.factorize(feat_raw.index.get_level_values(1))[0]
    datetimes = feat_raw.index.get_level_values(0).to_numpy()
    y = label_raw["LABEL0"].to_numpy(dtype=np.float32)

    # 按 (instrument, datetime) 排序：同股票交易日连续
    order = np.lexsort((datetimes, inst))
    inst_s = inst[order]
    dt_s = datetimes[order]
    y_s = y[order]

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

    # 只保存排序后的 end position（不保存 [S, L] 窗口行号）
    end_pos = np.flatnonzero(mask).astype(np.int32)
    end_orig = order[end_pos]
    end_idx = feat_raw.index[end_orig]
    return order.astype(np.int32), end_pos, end_idx


class PanelWindowDataset(Dataset):
    """只保存排序后 order（[N]）与每个样本的结尾排序位置 end_pos（[S]），
    __getitem__ 动态切片 order[p-L+1:p+1] 取 [L, C] 窗口，转置为 [C, L]（TCN 通道优先）。"""

    def __init__(self, X, y, order, end_pos, L):
        self.X = np.ascontiguousarray(X, dtype=np.float32)   # [N, C]
        self.y = np.ascontiguousarray(y, dtype=np.float32)   # [N]
        self.order = np.ascontiguousarray(order, dtype=np.int32)  # [N] 排序->原始行号
        self.end_pos = np.ascontiguousarray(end_pos, dtype=np.int32)  # [S] 结尾排序位置
        self.L = L

    def __len__(self):
        return len(self.end_pos)

    def __getitem__(self, k):
        p = int(self.end_pos[k])                     # 排序后结尾位置
        rows = self.order[p - self.L + 1:p + 1]      # [L] 原始行号
        w = np.ascontiguousarray(self.X[rows].T, dtype=np.float32)  # [C, L]
        return w, float(self.y[rows[-1]])            # 结尾日标签


def _make_loader(X, y, order, end_pos, L, batch_size, shuffle):
    ds = PanelWindowDataset(X, y, order, end_pos, L)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def _save_checkpoint(path, model, opt, epoch, best_val, wait):
    """保存一个检查点：模型权重 + 优化器状态 + epoch + 最佳验证损失 + 早停耐心计数。"""
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": opt.state_dict() if opt is not None else None,
            "best_val": float(best_val),
            "wait": int(wait),
        },
        path,
    )


def _load_checkpoint(path, model, opt, device):
    """加载检查点：恢复模型权重（与可选优化器状态），返回完整 ckpt 字典。"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if opt is not None and ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])
    return ckpt


def _prepare_common(feat_raw, label_raw, n_feat, args, segments):
    """动态窗口路径的公共数据准备：标准化 + 各 segment 切片 + 构造模型。

    训练（train_dynamic）与纯推理（predict_from_weights）共用，确保两侧的特征
    标准化口径（仅用训练区间统计）与窗口切片完全一致，推理结果才可复现训练时的信号。
    """
    L = args.n_days

    X = feat_raw.values.astype(np.float32)
    y = label_raw["LABEL0"].to_numpy(dtype=np.float32)
    C = X.shape[1]

    # 特征标准化：仅用训练区间统计，避免标签泄漏。
    tr_s, tr_e = segments["train"]
    tr_mask = ((feat_raw.index.get_level_values(0) >= pd.Timestamp(tr_s)) &
               (feat_raw.index.get_level_values(0) <= pd.Timestamp(tr_e)))
    train_X = X[tr_mask]
    mu = np.nanmean(train_X, axis=0)
    sd = np.nanstd(train_X, axis=0)
    mu[~np.isfinite(mu)] = 0.0
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    X = (X - mu) / sd
    # TCN 对 NaN/inf 不鲁棒，统一零值填充
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    tr_order, tr_pos, tr_idx = _segment_windows(feat_raw, label_raw, segments["train"], L)
    va_order, va_pos, va_idx = _segment_windows(feat_raw, label_raw, segments["valid"], L)
    te_order, te_pos, te_idx = _segment_windows(feat_raw, label_raw, segments["test"], L)

    use_cuda = args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda", args.gpu) if use_cuda else torch.device("cpu")
    model = CausalTCN(C, args.n_chans, args.kernel_size, args.num_layers, args.dropout).to(device)
    log.info(f"[dynamic] 设备={device}")
    return dict(
        X=X, y=y, L=L,
        tr_order=tr_order, tr_pos=tr_pos, tr_idx=tr_idx,
        va_order=va_order, va_pos=va_pos, va_idx=va_idx,
        te_order=te_order, te_pos=te_pos, te_idx=te_idx,
        use_cuda=use_cuda, device=device, model=model,
    )


def predict_from_weights(feat_raw, label_raw, n_feat, args, segments, weights_path):
    """不训练：加载已保存的 .pt 权重，仅对测试段做推理，返回 (pred_df, label_df, model)。

    与 train_dynamic 共用 _prepare_common，保证特征标准化 / 窗口切片口径一致。
    模型结构（C/n_chans/kernel_size/num_layers）必须与保存权重时一致（由同样的 args 控制）。
    """
    prep = _prepare_common(feat_raw, label_raw, n_feat, args, segments)
    model, device, use_cuda = prep["model"], prep["device"], prep["use_cuda"]
    _load_checkpoint(weights_path, model, None, device)
    log.info(f"[dynamic] 已加载权重：{weights_path}，直接对测试段推理（不训练）")

    te_loader = _make_loader(prep["X"], prep["y"], prep["te_order"], prep["te_pos"],
                             prep["L"], args.batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        te_pbar = tqdm(te_loader, desc="predict test (from weights)", unit="batch")
        for xb, _ in te_pbar:
            preds.append(model(xb.to(device, non_blocking=use_cuda)).cpu().numpy())
    preds = np.concatenate(preds, axis=0).astype(np.float32)

    pred_df = pd.DataFrame({"score": preds}, index=prep["te_idx"]).sort_index()
    te_end_orig = prep["te_order"][prep["te_pos"]]
    label_df = pd.DataFrame({"LABEL0": prep["y"][te_end_orig]}, index=prep["te_idx"]).sort_index()
    log.info(f"[dynamic] 测试预测 {pred_df.shape}（自权重推理，已生成 pred.pkl / label.pkl 兼容格式）")
    return pred_df, label_df, model


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

    prep = _prepare_common(feat_raw, label_raw, n_feat, args, segments)
    X, y = prep["X"], prep["y"]
    L = prep["L"]
    tr_order, tr_pos, tr_idx = prep["tr_order"], prep["tr_pos"], prep["tr_idx"]
    va_order, va_pos, va_idx = prep["va_order"], prep["va_pos"], prep["va_idx"]
    te_order, te_pos, te_idx = prep["te_order"], prep["te_pos"], prep["te_idx"]
    use_cuda, device, model = prep["use_cuda"], prep["device"], prep["model"]
    log.info(f"[dynamic] 样本数 -> train={len(tr_pos)} valid={len(va_pos)} test={len(te_pos)}")

    # 感受野检查：确认 TCN 实际能利用的回看长度 >= 输入窗口 L
    rf = 1 + (args.kernel_size - 1) * (2 ** args.num_layers - 1)
    log.info(f"[TCN] window={L}, receptive_field={rf}")
    if rf < L:
        log.warning(f"[TCN] receptive field={rf} < window={L}，"
                    f"后面的 {L - rf} 天输入实际上无法影响最终预测，建议增大 num_layers / kernel_size。")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 继续训练：从已保存的检查点恢复（模型权重 + 优化器状态 + 早停历史）
    start_epoch, best_val, wait, best_state = 1, float("inf"), 0, None
    resume_from = getattr(args, "resume_from", None)
    if resume_from:
        ckpt = _load_checkpoint(resume_from, model, opt, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        wait = int(ckpt.get("wait", 0))
        best_state = {k: v.detach().cpu().clone() for k, v in ckpt["model"].items()}
        log.info(f"[dynamic] 从检查点恢复训练：{resume_from}（起点 epoch={start_epoch}，"
                 f"best_val={best_val:.6f}）")

    tr_loader = _make_loader(X, y, tr_order, tr_pos, L, args.batch_size, shuffle=True)
    va_loader = _make_loader(X, y, va_order, va_pos, L, args.batch_size, shuffle=False)

    epoch_bar = trange(start_epoch, args.rounds + 1, desc="train")
    for epoch in epoch_bar:
        model.train()
        tr_loss = 0.0
        tr_pbar = tqdm(tr_loader, desc=f"epoch {epoch:03d} train", leave=False, unit="batch")
        for xb, yb in tr_pbar:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
            tr_pbar.set_postfix(loss=f"{loss.item():.6f}")
        tr_loss /= max(1, len(tr_pos))

        model.eval()
        va_loss = 0.0
        va_pbar = tqdm(va_loader, desc=f"epoch {epoch:03d} valid", leave=False, unit="batch")
        with torch.no_grad():
            for xb, yb in va_pbar:
                xb = xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                va_loss += criterion(model(xb), yb).item() * len(yb)
                va_pbar.set_postfix(loss=f"{va_loss / max(1, va_pbar.n):.6f}")
        va_loss /= max(1, len(va_pos))

        log.info(f"[dynamic] epoch {epoch:03d} | train_mse={tr_loss:.6f} valid_mse={va_loss:.6f}")
        epoch_bar.set_postfix(tr_mse=f"{tr_loss:.6f}", va_mse=f"{va_loss:.6f}")
        if va_loss < best_val - 1e-8:
            best_val, wait, best_state = va_loss, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.early_stop:
                epoch_bar.write(f"[dynamic] 早停：验证集 {args.early_stop} 轮无改善")
                break

        # 每 save_every 个 epoch 落盘一个检查点（含优化器状态，便于继续训练）
        save_every = getattr(args, "save_every", 5)
        weights_dir = getattr(args, "weights_dir", None)
        if save_every and weights_dir and (epoch % save_every == 0):
            os.makedirs(weights_dir, exist_ok=True)
            ckpt_path = os.path.join(weights_dir, f"tcn_weights_epoch_{epoch:03d}.pt")
            _save_checkpoint(ckpt_path, model, opt, epoch, best_val, wait)
            log.info(f"[dynamic] 检查点已保存：{ckpt_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"[dynamic] 载入最佳验证权重（valid_mse={best_val:.6f}）")

    # 训练结束后保存最终检查点，便于后续随时加载做推理 / 继续训练
    weights_dir = getattr(args, "weights_dir", None)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
        _save_checkpoint(
            os.path.join(weights_dir, "tcn_weights_last.pt"),
            model, opt, args.rounds, best_val, wait,
        )
        log.info(f"[dynamic] 最终检查点已保存："
                 f"{os.path.join(weights_dir, 'tcn_weights_last.pt')}")

    # 预测测试段
    te_loader = _make_loader(X, y, te_order, te_pos, L, args.batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        te_pbar = tqdm(te_loader, desc="predict test", unit="batch")
        for xb, _ in te_pbar:
            preds.append(model(xb.to(device, non_blocking=use_cuda)).cpu().numpy())
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    pred_df = pd.DataFrame({"score": preds}, index=te_idx)
    # 测试样本结尾对应的原始行号（order[end_pos]），与 X/y 行对齐
    te_end_orig = te_order[te_pos]
    label_df = pd.DataFrame({"LABEL0": y[te_end_orig]}, index=te_idx)
    # 回测对索引顺序通常有隐含要求，最终排序确保与 qlib (datetime, instrument) 格式一致
    pred_df = pred_df.sort_index()
    label_df = label_df.sort_index()
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
