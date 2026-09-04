"""Post-processing: final comparison figures and metrics summary.

This script reads only the CSV/JSON files produced by previous experiment
steps. It does not recompute kernels and does not submit any QPU job.

Main outputs
------------
results_tfm/
    metrics_summary.csv

figures_tfm/
    K_train_comparison.png/.pdf
    K_test_scatter_comparison.png/.pdf

Modalities whose input files are missing are skipped with an informative
message.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfm_experiment import common, config


# ---------------------------------------------------------------------------
# Metrics included in the summary table.
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "train_accuracy",
    "test_accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "SEP_accuracy",
    # PPT_accuracy / NPPT_accuracy are per-block recall values for the
    # entangled class (label 1). They are not three-class accuracies:
    # PPT and NPPT share the same binary target label.
    "PPT_accuracy",
    "NPPT_accuracy",
]

EXTRA_METRIC_KEYS = [
    "relative_frobenius_train",
    "relative_frobenius_test",
    "min_eigenvalue_K_train",
]


# ---------------------------------------------------------------------------
# Plot style.
# ---------------------------------------------------------------------------

def _configure_plot_style() -> None:
    """Use a clean, consistent Matplotlib style suitable for the report."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    """Save both high-resolution PNG and vector PDF versions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("saved", png_path)
    print("saved", pdf_path)


# ---------------------------------------------------------------------------
# Loading helpers.
# ---------------------------------------------------------------------------

def _try_load_json(path: Path) -> dict | None:
    """Load a JSON file if it exists."""
    if path.exists():
        return common.load_json(path)

    print(f"Missing {path.name}; skipping.")
    return None


def _try_load_matrix(path: Path) -> np.ndarray | None:
    """Load a matrix if it exists."""
    if path.exists():
        return common.load_matrix_csv(path)

    print(f"Missing {path.name}; skipping.")
    return None


def _same_shape(*matrices: np.ndarray | None) -> bool:
    """Return whether all matrices are available and have the same shape."""
    if not matrices or any(matrix is None for matrix in matrices):
        return False

    return all(
        matrix.shape == matrices[0].shape
        for matrix in matrices[1:]
    )


# ---------------------------------------------------------------------------
# Kernel comparison figures.
# ---------------------------------------------------------------------------

def _plot_kernel_comparison(
    ideal: np.ndarray,
    shots: np.ndarray,
    ibm: np.ndarray,
    out_path: Path,
) -> None:
    """
    Compare the three training kernels using the same colour scale.
    """
    if not _same_shape(ideal, shots, ibm):
        raise ValueError(
            "Kernel comparison requires ideal, finite-shot and IBM "
            "training matrices with identical shapes."
        )

    matrices = [ideal, shots, ibm]
    titles = [
        r"(a) Ideal $K_{\mathrm{train}}$",
        r"(b) Finite-shot $K_{\mathrm{train}}$",
        r"(c) IBM QPU $K_{\mathrm{train}}$",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.4, 3.45),
        constrained_layout=True,
    )

    im = None

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(
            matrix,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("Training sample index")

    axes[0].set_ylabel("Training sample index")

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.84,
        pad=0.02,
    )
    cbar.set_label(r"Kernel value $K_{ij}$")

    _save_figure(fig, out_path)


def _plot_kernel_scatter_comparison(
    ideal_test: np.ndarray,
    shots_test: np.ndarray,
    ibm_test: np.ndarray,
    out_path: Path,
) -> None:
    """
    Compare all test-kernel entries with their exact ideal values.

    K_test is used instead of K_train because it contains no analytically
    imposed unit diagonal. Therefore every point shown here corresponds to
    an actually estimated test-train kernel value.
    """
    if not _same_shape(ideal_test, shots_test, ibm_test):
        raise ValueError(
            "Scatter comparison requires ideal, finite-shot and IBM "
            "test matrices with identical shapes."
        )

    ideal_flat = ideal_test.ravel()
    shots_flat = shots_test.ravel()
    ibm_flat = ibm_test.ravel()

    tab10 = plt.get_cmap("tab10")
    point_colors = [tab10(0), tab10(1)]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.35),
        constrained_layout=True,
    )

    panels = [
        (axes[0], shots_flat, "(a) Finite-shot", point_colors[0]),
        (axes[1], ibm_flat, "(b) IBM QPU", point_colors[1]),
    ]

    for ax, measured, title, color in panels:
        ax.scatter(
            ideal_flat,
            measured,
            s=14,
            alpha=0.48,
            color=color,
            edgecolors="none",
            rasterized=True,
        )

        # Identity line: perfect agreement with the exact kernel.
        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1.1,
            color="0.25",
            label=r"Ideal agreement ($y=x$)",
        )

        mae = float(np.mean(np.abs(measured - ideal_flat)))
        bias = float(np.mean(measured - ideal_flat))

        ax.text(
            0.04,
            0.95,
            f"MAE = {mae:.3f}\nMean bias = {bias:+.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.92,
            },
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Ideal kernel value $K_{ij}$")
        ax.set_ylabel(r"Estimated kernel value $\hat{K}_{ij}$")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False)

    _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Summary table.
# ---------------------------------------------------------------------------

def _save_metrics_summary(
    metrics_by_source: dict[str, dict],
) -> None:
    """
    Save the main experiment metrics in a single comparison table.
    """
    summary_rows: list[dict] = []

    for source, data in metrics_by_source.items():
        row: dict[str, object] = {"source": source}

        for key in METRIC_KEYS + EXTRA_METRIC_KEYS:
            row[key] = data.get(key, "")

        summary_rows.append(row)

    if not summary_rows:
        print("No metrics JSON files found; skipping metrics_summary.csv.")
        return

    summary_path = config.RESULTS_DIR / "metrics_summary.csv"
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = [
        "source",
    ] + METRIC_KEYS + EXTRA_METRIC_KEYS

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=header,
        )
        writer.writeheader()

        for row in summary_rows:
            formatted: dict[str, object] = {}

            for key in header:
                value = row.get(key, "")

                if isinstance(value, (float, np.floating)):
                    formatted[key] = f"{float(value):.6f}"
                else:
                    formatted[key] = value

            writer.writerow(formatted)

    print("saved", summary_path)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    _configure_plot_style()
    common.ensure_dir(config.FIGURES_DIR)

    # ------------------------------------------------------------------
    # Load matrices.
    # ------------------------------------------------------------------

    ideal_train = _try_load_matrix(
        config.RESULTS_DIR / "subset_ideal_K_train.csv"
    )
    shots_train = _try_load_matrix(
        config.RESULTS_DIR / "shots_K_train.csv"
    )
    ibm_train = _try_load_matrix(
        config.RESULTS_DIR / "ibm_K_train.csv"
    )

    ideal_test = _try_load_matrix(
        config.RESULTS_DIR / "subset_ideal_K_test.csv"
    )
    shots_test = _try_load_matrix(
        config.RESULTS_DIR / "shots_K_test.csv"
    )
    ibm_test = _try_load_matrix(
        config.RESULTS_DIR / "ibm_K_test.csv"
    )

    # ------------------------------------------------------------------
    # Combined kernel figures.
    # ------------------------------------------------------------------

    if _same_shape(
        ideal_train,
        shots_train,
        ibm_train,
    ):
        _plot_kernel_comparison(
            ideal_train,
            shots_train,
            ibm_train,
            config.FIGURES_DIR / "K_train_comparison",
        )
    else:
        print(
            "Ideal / finite-shot / IBM train kernels are not all available "
            "with matching shapes; skipping combined train-kernel figures."
        )

    if _same_shape(
        ideal_test,
        shots_test,
        ibm_test,
    ):
        _plot_kernel_scatter_comparison(
            ideal_test,
            shots_test,
            ibm_test,
            config.FIGURES_DIR / "K_test_scatter_comparison",
        )
    else:
        print(
            "Ideal / finite-shot / IBM test kernels are not all available "
            "with matching shapes; skipping K_test scatter comparison."
        )

    # ------------------------------------------------------------------
    # Load metrics and write the summary CSV.
    # ------------------------------------------------------------------

    metrics_by_source: dict[str, dict] = {}

    for label, filename in [
        ("ideal_full", "reference_metrics.json"),
        ("ideal_subset", "subset_ideal_metrics.json"),
        ("shots_subset", "shots_metrics.json"),
        ("ibm_subset", "ibm_metrics.json"),
    ]:
        loaded = _try_load_json(config.RESULTS_DIR / filename)
        if loaded is not None:
            metrics_by_source[label] = loaded

    _save_metrics_summary(metrics_by_source)

    print("\nPost-processing complete.")
    print("No kernel was recomputed and no QPU job was submitted.")


if __name__ == "__main__":
    main()