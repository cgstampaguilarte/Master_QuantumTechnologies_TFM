"""
Shared utilities for the TFM quantum-kernel experiments.

This module contains the common routines for dataset loading, PCA,
Pauli-ZZ feature-map construction, kernel evaluation, classification
metrics and file I/O. The ideal statevector workflow is adapted from 
the implementation of Acosta et al., while the finite-shot and 
hardware utilities extend it with compute-uncompute kernel estimation
and explicit kernel-matrix handling.
"""


from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import pauli_feature_map
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from . import config


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(
    dataset_root: Path = config.DATASET_ROOT,
    train_size: int = config.TRAIN_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the 3x3 entanglement dataset.
    """
    train_dir = Path(dataset_root) / "train"
    test_dir = Path(dataset_root) / "test"

    x_train = np.genfromtxt(train_dir / f"x_n_{train_size}.csv", delimiter=",", dtype=None)
    x_test = np.genfromtxt(test_dir / "x_test.csv", delimiter=",", dtype=None)
    y_train = np.genfromtxt(train_dir / f"y_n_{train_size}.csv", delimiter=",", dtype=None).astype(int)
    y_test = np.genfromtxt(test_dir / "y_test.csv", delimiter=",", dtype=None).astype(int)
    return x_train, x_test, y_train, y_test


def assert_dataset_shapes(x_train, x_test, y_train, y_test) -> None:
    """
    Fail early if the CSVs on disk do not match the expected shapes.
    """
    if x_train.shape != (config.TRAIN_SIZE, config.RAW_FEATURES):
        raise ValueError(
            f"Unexpected training shape {x_train.shape}, "
            f"expected ({config.TRAIN_SIZE}, {config.RAW_FEATURES})."
        )
    if x_test.shape != (config.TEST_SIZE, config.RAW_FEATURES):
        raise ValueError(
            f"Unexpected test shape {x_test.shape}, "
            f"expected ({config.TEST_SIZE}, {config.RAW_FEATURES})."
        )
    if y_train.shape != (config.TRAIN_SIZE,) or y_test.shape != (config.TEST_SIZE,):
        raise ValueError("Label vectors do not match feature matrices.")

    if config.TEST_SIZE % 3 != 0:
        raise ValueError("Test size is not divisible by 3; block convention broken.")
    block = config.TEST_SIZE // 3
    if (
        not np.all(y_test[:block] == config.LABEL_SEP)
        or not np.all(y_test[block : 2 * block] == config.LABEL_ENTANGLED)
        or not np.all(y_test[2 * block :] == config.LABEL_ENTANGLED)
    ):
        raise ValueError(
            "Test labels do not follow the SEP / PPT / NPPT convention "
            "of the original repository (first third label 0, "
            "remaining two thirds label 1)."
        )


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def apply_pca(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int = config.PCA_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA on the training set and transform both train and test data.
    """
    pca = PCA(n_components=n_components)
    xs_train = pca.fit_transform(x_train)
    xs_test = pca.transform(x_test)
    return xs_train, xs_test, pca


# ---------------------------------------------------------------------------
# Pauli-ZZ feature map
# ---------------------------------------------------------------------------
def make_pauli_zz_feature_map(
    n_features: int = config.PCA_COMPONENTS,
    reps: int = config.REPS,
) -> QuantumCircuit:
    """
    Return the parametrised Pauli-ZZ feature map template.
    """
    return pauli_feature_map(
        feature_dimension=n_features,
        reps=reps,
        paulis=config.FEATURE_MAP_PAULIS,
        entanglement=config.FEATURE_MAP_ENTANGLEMENT,
        alpha=config.FEATURE_MAP_ALPHA,
    )


def pauli_zz_circuit(x: np.ndarray, feature_map: QuantumCircuit) -> QuantumCircuit:
    """
    Create the feature-map circuit for one input vector.
    """
    return feature_map.assign_parameters(x, inplace=False)


# ---------------------------------------------------------------------------
# Statevector-based (ideal) kernel
# ---------------------------------------------------------------------------
def precompute_statevectors(
    X: np.ndarray,
    embedding_fn: Callable[[np.ndarray], QuantumCircuit],
) -> np.ndarray:
    """
    Compute one ideal statevector for each input sample.
    """
    states: list[np.ndarray] = []
    for x in X:
        qc = embedding_fn(x)
        psi = Statevector.from_instruction(qc).data
        states.append(psi)
    return np.asarray(states, dtype=complex)


def exact_kernel(psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
    """
    Compute the exact fidelity kernel from two sets of statevectors.
    """
    overlaps = psi_a.conj() @ psi_b.T
    return np.abs(overlaps) ** 2


# ---------------------------------------------------------------------------
# Compute-uncompute overlap circuit
# ---------------------------------------------------------------------------
def create_overlap_circuit(
    feature_map: QuantumCircuit,
    x: np.ndarray,
    y: np.ndarray,
) -> QuantumCircuit:
    """
    Build the compute-uncompute circuit S^dagger(y)S(x)|0> used to estimate K(x,y).
    """
    ux = feature_map.assign_parameters(x, inplace=False)
    uy = feature_map.assign_parameters(y, inplace=False)
    qc = QuantumCircuit(feature_map.num_qubits)
    qc.compose(ux, inplace=True)
    qc.compose(uy.inverse(), inplace=True)
    qc.measure_all()
    return qc


def train_pairs(n: int) -> list[tuple[int, int]]:
    """
    Upper-triangular index pairs (i < j) for a symmetric training kernel.
    """
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def test_pairs(n_test: int, n_train: int) -> list[tuple[int, int]]:
    """
    All (test, train) index pairs, in row-major order.
    """
    return [(i, j) for i in range(n_test) for j in range(n_train)]


def assemble_train_kernel(n: int, pairs: Sequence[tuple[int, int]], values: Sequence[float]) -> np.ndarray:
    """
    Symmetric N x N matrix with diagonal = 1.
    """
    K = np.eye(n, dtype=float)
    for (i, j), v in zip(pairs, values):
        K[i, j] = v
        K[j, i] = v
    return K


def assemble_test_kernel(n_test: int, n_train: int, pairs: Sequence[tuple[int, int]], values: Sequence[float]) -> np.ndarray:
    """
    Row-major (test x train) kernel matrix, orientation used by ``SVC``.
    """
    K = np.zeros((n_test, n_train), dtype=float)
    for (i, j), v in zip(pairs, values):
        K[i, j] = v
    return K


def fidelity_from_counts(counts: dict) -> float:
    """
    Estimate the kernel value from the all-zero measurement frequency.
    """
    total = sum(counts.values())
    if total == 0: 
        raise ValueError("No measurement counts were returned.")
    return float(counts.get(0, 0) / total)


def two_qubit_count(qc: QuantumCircuit) -> int:
    """
    Count the circuit instructions acting on two qubits.
    """
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def detailed_accuracy(y_test_pred: np.ndarray, block_size: int | None = None) -> dict:
    """
    Per-block accuracies.
    """
    if block_size is None:
        if len(y_test_pred) % 3 != 0:
            raise ValueError("Prediction length not divisible by 3.")
        block_size = len(y_test_pred) // 3
    if len(y_test_pred) != 3 * block_size:
        raise ValueError("Predictions do not fit the requested block size.")

    pred_split = np.array_split(y_test_pred, 3)
    y_sep_pred, y_ppt_pred, y_nppt_pred = pred_split

    score_sep = accuracy_score(y_sep_pred, np.full(block_size, config.LABEL_SEP))
    score_ppt = accuracy_score(y_ppt_pred, np.full(block_size, config.LABEL_ENTANGLED))
    score_nppt = accuracy_score(y_nppt_pred, np.full(block_size, config.LABEL_ENTANGLED))
    return {
        "SEP_accuracy": float(score_sep),
        "PPT_accuracy": float(score_ppt),
        "NPPT_accuracy": float(score_nppt),
    }


def compute_classification_metrics(
    y_train: np.ndarray,
    pred_train: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    block_size: int | None = None,
) -> dict:
    metrics = {
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
        "f1": float(f1_score(y_test, pred_test)),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, zero_division=0)),
    }
    metrics.update(detailed_accuracy(pred_test, block_size=block_size))
    return metrics


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """
    Relative Frobenius error ``||A - B||_F / ||B||_F``.
    """
    denom = np.linalg.norm(b, ord="fro")
    if denom == 0:
        raise ValueError("Reference matrix has zero Frobenius norm.")
    return float(np.linalg.norm(a - b, ord="fro") / denom)


def min_eigenvalue(matrix: np.ndarray) -> float:
    """
    Minimum eigenvalue of a (symmetrised) matrix.
    """
    symmetrised = 0.5 * (matrix + matrix.T)
    return float(np.linalg.eigvalsh(symmetrised).min())


# ---------------------------------------------------------------------------
# Subset IO shared by the sampler / transpile / IBM scripts
# ---------------------------------------------------------------------------
def load_subset_features_and_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PCA features and binary labels for the fixed subset.
    """
    train_path = config.RESULTS_DIR / "subset_train.csv"
    test_path = config.RESULTS_DIR / "subset_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Subset CSVs not found. Run tfm_experiment/02_prepare_subset.py first."
        )
    train_arr = np.genfromtxt(
        train_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
    test_arr = np.genfromtxt(
        test_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    pc_cols = [f"pc_{i + 1}" for i in range(config.PCA_COMPONENTS)]
    Xtr = np.column_stack([train_arr[c] for c in pc_cols])
    Xte = np.column_stack([test_arr[c] for c in pc_cols])
    ytr = train_arr["label"].astype(int)
    yte = test_arr["label"].astype(int)
    return Xtr, Xte, ytr, yte


def load_subset_ideal_kernels() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the subset ideal K_train, K_test written by step 02.
    """
    Ktr = load_matrix_csv(config.RESULTS_DIR / "subset_ideal_K_train.csv")
    Kte = load_matrix_csv(config.RESULTS_DIR / "subset_ideal_K_test.csv")
    return Ktr, Kte


def load_reference_pca_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (PCA components, integer labels) from a step-01 CSV.
    """
    array = np.genfromtxt(path, delimiter=",", names=True)
    pc_cols = [f"pc_{i + 1}" for i in range(config.PCA_COMPONENTS)]
    X = np.column_stack([array[c] for c in pc_cols])
    y = array["label"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
CSV_FLOAT_FMT = "%.18e"


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_matrix_csv(matrix: np.ndarray, path: Path, fmt: str = CSV_FLOAT_FMT) -> None:
    """Store a 2D array as a plain CSV that round-trips ``float64`` exactly."""
    path = Path(path)
    ensure_dir(path.parent)
    np.savetxt(path, matrix, delimiter=",", fmt=fmt)


def load_matrix_csv(path: Path) -> np.ndarray:
    return np.loadtxt(Path(path), delimiter=",")


def save_labelled_table(
    path: Path,
    columns: dict[str, Iterable],
    fmt: str = CSV_FLOAT_FMT,
) -> None:
    """
    Write a small CSV with a header row and one column per key.
    """
    path = Path(path)
    ensure_dir(path.parent)
    header = ",".join(columns.keys())
    arrays = [np.asarray(v) for v in columns.values()]
    n_rows = len(arrays[0])
    if any(len(a) != n_rows for a in arrays):
        raise ValueError("All columns must have the same length.")
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in range(n_rows):
            fields = []
            for arr in arrays:
                v = arr[row]
                if np.issubdtype(arr.dtype, np.integer):
                    fields.append(str(int(v)))
                elif np.issubdtype(arr.dtype, np.floating):
                    fields.append(fmt % float(v))
                else:
                    fields.append(str(v))
            f.write(",".join(fields) + "\n")
