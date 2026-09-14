#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""View the analysis reports of a qlib backtest/forecast run.

This script loads the artifacts produced by a qlib workflow run and regenerates
the four standard analysis figures (model performance, backtest report, risk
analysis, score IC) as standalone interactive HTML files, plus a short text
summary of the key metrics.

Two ways to point it at a run:

1) Directly from an ``artifacts`` directory (no qlib/mlflow init needed):

    python view_analysis.py --artifacts /home/fei/workspace/mlruns/2/<rid>/artifacts

2) Via a mlflow Recorder (requires ``qlib.init`` with a provider_uri):

    python view_analysis.py --recorder-id <rid> --experiment-name workflow \
        --provider-uri /path/to/qlib/data

Outputs are written to ``--out`` (default: ./analysis_html).
"""

import argparse
import os
import pickle
import sys

import pandas as pd

# Make sure the workspace qlib (the one we patched for plotly>=5 / pandas>=2
# compatibility) is importable regardless of how the script is launched.
# NOTE: setting os.environ["PYTHONPATH"] does NOT affect the already-running
# interpreter, so we insert into sys.path directly.
_WORKSPACE_QLIB = "/home/fei/workspace/qlib"
if _WORKSPACE_QLIB not in sys.path:
    sys.path.insert(0, _WORKSPACE_QLIB)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_from_artifacts(artifacts_dir: str):
    """Load the four required objects straight from an ``artifacts`` directory."""
    base = artifacts_dir

    def _load(rel):
        p = os.path.join(base, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing artifact: {p}")
        with open(p, "rb") as f:
            return pickle.load(f)

    return {
        "pred_df": _load("pred.pkl"),
        "label_df": _load("label.pkl"),
        "report_normal_df": _load("portfolio_analysis/report_normal_1day.pkl"),
        "analysis_df": _load("portfolio_analysis/port_analysis_1day.pkl"),
        "source": base,
    }


def load_from_recorder(recorder_id: str, experiment_name: str, provider_uri: str):
    """Load the same objects via a qlib/mlflow Recorder (needs qlib.init)."""
    import qlib
    from qlib.workflow import R

    qlib.init(provider_uri=provider_uri)
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)

    return {
        "pred_df": recorder.load_object("pred.pkl"),
        "label_df": recorder.load_object("label.pkl"),
        "report_normal_df": recorder.load_object("portfolio_analysis/report_normal_1day.pkl"),
        "analysis_df": recorder.load_object("portfolio_analysis/port_analysis_1day.pkl"),
        "source": f"recorder {recorder_id} / {experiment_name}",
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def build_pred_label(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the notebook's ``pred_label`` construction."""
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    pred_label.columns = ["label", "score"]
    return pred_label


def generate_figures(objs: dict, out_dir: str):
    from qlib.contrib.report import analysis_model, analysis_position

    pred_label = build_pred_label(objs["pred_df"], objs["label_df"])
    report_normal_df = objs["report_normal_df"]
    analysis_df = objs["analysis_df"]

    os.makedirs(out_dir, exist_ok=True)
    written = []

    def _save(figs, prefix):
        for i, fig in enumerate(figs):
            path = os.path.join(out_dir, f"{prefix}_{i}.html")
            fig.write_html(path)
            written.append(path)

    _save(analysis_model.model_performance_graph(pred_label, show_notebook=False), "model_perf")
    _save(analysis_position.report_graph(report_normal_df, show_notebook=False), "report")
    _save(analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False), "risk")
    _save(analysis_position.score_ic_graph(pred_label, show_notebook=False), "score_ic")

    return written


# --------------------------------------------------------------------------- #
# Text summary
# --------------------------------------------------------------------------- #
def print_summary(objs: dict):
    from qlib.contrib.evaluate import risk_analysis

    pred_label = build_pred_label(objs["pred_df"], objs["label_df"]).dropna()
    report = objs["report_normal_df"]
    analysis_df = objs["analysis_df"]

    print("=" * 64)
    print("Backtest report_normal_1day")
    print("=" * 64)
    if len(report) > 0:
        ex = report["return"] - report["bench"]
        print(f"  period        : {report.index[0].date()} -> {report.index[-1].date()}  ({len(report)} days)")
        print(f"  excess ann ret : {risk_analysis(ex)['risk']['annualized_return']:.4f}")
        for k in ["return", "bench", "cost", "turnover"]:
            if k in report:
                print(f"  {k:<7} mean/day={report[k].mean():.6f}  cumsum={report[k].cumsum().iloc[-1]:.4f}")
    else:
        print("  (empty)")

    print()
    print("=" * 64)
    print("Prediction signal (pred_label)")
    print("=" * 64)
    ic = pred_label.groupby(level="datetime", group_keys=False).apply(lambda x: x["label"].corr(x["score"]))
    ric = pred_label.groupby(level="datetime", group_keys=False).apply(
        lambda x: x["label"].corr(x["score"], method="spearman")
    )
    print(f"  IC      mean={ic.mean():.4f}  std={ic.std():.4f}  ICIR={ic.mean() / ic.std():.3f}  IC>0 ratio={ (ic > 0).mean():.3f}")
    print(f"  RankIC  mean={ric.mean():.4f}  std={ric.std():.4f}  ICIR={ric.mean() / ric.std():.3f}")

    print()
    print("=" * 64)
    print("Loaded port_analysis (head)")
    print("=" * 64)
    with pd.option_context("display.max_rows", 12, "display.width", 200):
        print(analysis_df.head(12).to_string())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="View qlib run analysis reports.")
    parser.add_argument("--artifacts", default="/home/fei/workspace/qlib/me/IC_mul/mlruns/10/d91341f8d6d2450eba269a9b349e90e2/artifacts", help="Path to an artifacts directory (preferred).")
    parser.add_argument("--recorder-id", default="d91341f8d6d2450eba269a9b349e90e2", help="mlflow recorder id.")
    parser.add_argument("--experiment-name", default="workflow", help="mlflow experiment name.")
    parser.add_argument("--provider-uri", default=None, help="qlib data provider_uri (recorder mode only).")
    parser.add_argument(
        "--out",
        default="/home/fei/workspace/qlib/me/IC_mul/results/analysis_html",
        help="Output directory for the generated HTML files.",
    )
    args = parser.parse_args()

    if args.artifacts:
        print(f"Loading from artifacts: {args.artifacts}")
        objs = load_from_artifacts(args.artifacts)
    else:
        if not args.provider_uri:
            parser.error("recorder mode requires --provider-uri (or use --artifacts instead)")
        print(f"Loading from recorder: {args.recorder_id} / {args.experiment_name}")
        objs = load_from_recorder(args.recorder_id, args.experiment_name, args.provider_uri)

    print_summary(objs)
    print("\nGenerating figures...")
    written = generate_figures(objs, args.out)
    print(f"Wrote {len(written)} HTML figures to: {args.out}")
    for p in written:
        print("  -", os.path.basename(p))


if __name__ == "__main__":
    main()
