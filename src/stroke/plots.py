"""Figures for the README.

Three of them, each answering a question the numbers alone leave open:

1. Precision-recall against the no-skill line, so the headline metric is visible
   relative to what a useless model scores.
2. Reliability, so a reader can see whether a predicted risk means anything —
   and see what class weighting does to it.
3. The threshold tradeoff, which is the only honest way to show what using the
   model costs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1b1f24"
MUTED = "#8a9099"
ACCENT = "#c2410c"
COOL = "#1d4ed8"
GRID = "#e6e8eb"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)


def precision_recall(metrics: dict, out: Path) -> Path:
    """PR curve for the selected model, with the no-skill line for scale."""
    sel = metrics["selected"]["test"]
    curve = sel["pr_curve"]
    no_skill = sel["pr_auc_no_skill"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    ax.plot(curve["recall"], curve["precision"], color=ACCENT, linewidth=2.0, label=None)
    ax.axhline(
        no_skill,
        color=MUTED,
        linewidth=1.2,
        linestyle="--",
        label=f"no skill (prevalence = {no_skill:.3f})",
    )

    op = sel["operating_point"]
    ax.scatter(
        [op["sensitivity"]],
        [op["precision"]],
        s=48,
        color=INK,
        zorder=5,
        label="reported operating point",
    )
    ax.annotate(
        f"sens {op['sensitivity']:.2f} / prec {op['precision']:.2f}",
        (op["sensitivity"], op["precision"]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=8,
        color=INK,
    )

    lo, hi = sel["pr_auc_ci95"]
    _style(
        ax,
        f"Precision-recall  |  PR-AUC {sel['pr_auc']:.3f}  (95% CI {lo:.3f}-{hi:.3f})",
        "Recall (sensitivity)",
        "Precision",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.6, max(curve["precision"]) * 1.05))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def calibration(metrics: dict, out: Path) -> Path:
    """Reliability for the calibrated model against a class-weighted one.

    The second line is the argument for why the demo does not use a weighted
    model: class weighting leaves discrimination essentially unchanged while
    multiplying every predicted risk several-fold.
    """
    pc = metrics["per_config_test"]
    good = pc["logistic_regression__none"]
    bad = pc["logistic_regression__class_weight"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle="--", label="perfect")

    for series, colour, label in (
        (good, COOL, f"unweighted  (Brier {good['brier']:.3f})"),
        (bad, ACCENT, f"class_weight='balanced'  (Brier {bad['brier']:.3f})"),
    ):
        xs = [b["mean_predicted"] for b in series["calibration"]]
        ys = [b["observed_rate"] for b in series["calibration"]]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.6, color=colour, label=label)

    hi = max(
        [b["mean_predicted"] for b in bad["calibration"]]
        + [b["observed_rate"] for b in bad["calibration"]]
        + [0.3]
    )
    _style(
        ax,
        "Reliability  |  predicted risk vs observed rate",
        "Mean predicted risk",
        "Observed rate",
    )
    ax.set_xlim(0, hi * 1.05)
    ax.set_ylim(0, hi * 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def threshold_tradeoff(metrics: dict, out: Path) -> Path:
    """Sensitivity, precision and accuracy across every threshold.

    Accuracy appears here and nowhere else. It runs flat and high across almost
    the entire range, including the region where the model catches nobody, which
    is the clearest available statement of why it is not reported.
    """
    sel = metrics["selected"]["test"]
    sweep = sel["threshold_sweep"]
    ts = [r["threshold"] for r in sweep]

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    ax.plot(ts, [r["sensitivity"] for r in sweep], color=ACCENT, linewidth=2.0, label="sensitivity")
    ax.plot(ts, [r["precision"] for r in sweep], color=COOL, linewidth=2.0, label="precision")
    ax.plot(
        ts,
        [r["accuracy"] for r in sweep],
        color=MUTED,
        linewidth=1.6,
        linestyle=":",
        label="accuracy (why it is useless here)",
    )

    op = sel["operating_point"]
    ax.axvline(op["threshold"], color=INK, linewidth=1.0)
    ax.annotate(
        f"reported threshold {op['threshold']:.3f}",
        (op["threshold"], 0.5),
        textcoords="offset points",
        xytext=(8, 0),
        fontsize=8,
        color=INK,
        rotation=90,
        va="center",
    )

    _style(ax, "What each threshold costs", "Decision threshold", "Rate")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(metrics: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        precision_recall(metrics, out_dir / "precision_recall.png"),
        calibration(metrics, out_dir / "calibration.png"),
        threshold_tradeoff(metrics, out_dir / "threshold_tradeoff.png"),
    ]
