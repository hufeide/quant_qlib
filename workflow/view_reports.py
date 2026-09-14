"""查看 workflow_by_code.py 跑出来的各类报表。

默认读取最新的 workflow 实验 run；可用环境变量 RUN 指定具体 run id：
    RUN=529528e24c014bb4a95f4e4e46ffb18b python view_reports.py
"""
import os
import pickle
import sqlite3
import pandas as pd

DB = "/home/fei/workspace/mlflow.db"
MLRUNS_ROOT = "/home/fei/workspace/mlruns"


def latest_workflow_run() -> str:
    """从 mlflow.db 里取 'workflow' 实验中最近一次 FINISHED 的 run。"""
    c = sqlite3.connect(DB)
    row = c.execute(
        "SELECT run_uuid FROM runs "
        "WHERE experiment_id=(SELECT experiment_id FROM experiments WHERE name='workflow') "
        "AND status='FINISHED' ORDER BY start_time DESC LIMIT 1"
    ).fetchone()
    c.close()
    if not row:
        raise SystemExit("未在 mlflow.db 的 workflow 实验中找到已完成的 run")
    return row[0]


def show(path: str):
    print("\n" + "=" * 28 + f"  {path}  " + "=" * 28)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        with pd.option_context("display.max_rows", 50, "display.width", 200,
                               "display.max_columns", 30):
            print(obj)
    elif isinstance(obj, pd.Series):
        print(obj.to_string())
    else:
        print(repr(obj))


def main():
    run_id = os.environ.get("RUN") or latest_workflow_run()
    base = f"{MLRUNS_ROOT}/2/{run_id}/artifacts"
    if not os.path.isdir(base):
        raise SystemExit(f"找不到 artifact 目录: {base}")

    print(f"Run: {run_id}")
    for rel in [
        "sig_analysis/ic.pkl",
        "sig_analysis/ric.pkl",
        "portfolio_analysis/report_normal_1day.pkl",
        "portfolio_analysis/port_analysis_1day.pkl",
        "portfolio_analysis/indicator_analysis_1day.pkl",
    ]:
        p = f"{base}/{rel}"
        if os.path.exists(p):
            show(p)
        else:
            print(f"\n[跳过] 不存在: {rel}")


if __name__ == "__main__":
    main()

    # RUN=529528e24c014bb4a95f4e4e46ffb18b python /home/fei/workspace/qlib/me/workflow/view_reports.py
