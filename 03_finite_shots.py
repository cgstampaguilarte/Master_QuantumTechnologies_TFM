"""
Evaluate the fixed subset with finite-shot noiseless simulation.

The compute-uncompute circuits are sampled with Qiskit's
``StatevectorSampler`` using the same subset and Pauli-ZZ feature map as
the hardware experiment. This provides a finite-shot reference between
the exact statevector results and the IBM QPU results.

Outputs are written to ``results_tfm/``:
    - shots_K_train.csv
    - shots_K_test.csv
    - shots_metrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from qiskit.primitives import StatevectorSampler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfm_experiment import common, config


def _run_batch(circuits, shots: int, seed: int) -> np.ndarray:
    """
    Return one estimated fidelity per circuit.
    """
    sampler = StatevectorSampler(seed=seed)
    result = sampler.run(circuits, shots=shots).result()
    values = np.empty(len(circuits), dtype=float)
    for k, pub_result in enumerate(result):
        counts = pub_result.data.meas.get_int_counts()
        values[k] = common.fidelity_from_counts(counts)
    return values


def build_train_kernel(X, feature_map, shots, seed):
    pairs = common.train_pairs(len(X))
    circuits = [common.create_overlap_circuit(feature_map, X[i], X[j]) for i, j in pairs]
    values = _run_batch(circuits, shots=shots, seed=seed)
    return common.assemble_train_kernel(len(X), pairs, values)


def build_test_kernel(X_test, X_train, feature_map, shots, seed):
    pairs = common.test_pairs(len(X_test), len(X_train))
    circuits = [
        common.create_overlap_circuit(feature_map, X_test[i], X_train[j])
        for i, j in pairs
    ]
    values = _run_batch(circuits, shots=shots, seed=seed)
    return common.assemble_test_kernel(len(X_test), len(X_train), pairs, values)


def _require_ideal_subset() -> tuple[np.ndarray, np.ndarray]:
    ideal_train = config.RESULTS_DIR / "subset_ideal_K_train.csv"
    ideal_test = config.RESULTS_DIR / "subset_ideal_K_test.csv"
    if not ideal_train.exists() or not ideal_test.exists():
        raise FileNotFoundError(
            "Ideal subset kernels not found. Run tfm_experiment/02_prepare_subset.py first."
        )
    return common.load_subset_ideal_kernels()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=config.SHOTS)
    parser.add_argument("--seed", type=int, default=config.SUBSET_SEED)
    args = parser.parse_args()

    common.ensure_dir(config.RESULTS_DIR)

    print("=" * 60)
    print(f"Finite-shot subset kernel (shots={args.shots}, seed={args.seed})")
    print("=" * 60)

    Xtr, Xte, ytr, yte = common.load_subset_features_and_labels()
    feature_map = common.make_pauli_zz_feature_map()

    print(f"Estimating training kernel ({len(Xtr)}x{len(Xtr)} with i<j pairs).")
    Ktr = build_train_kernel(Xtr, feature_map, args.shots, args.seed)
    print(f"Estimating test kernel ({len(Xte)}x{len(Xtr)}).")
    Kte = build_test_kernel(Xte, Xtr, feature_map, args.shots, args.seed + 1)

    common.save_matrix_csv(Ktr, config.RESULTS_DIR / "shots_K_train.csv")
    common.save_matrix_csv(Kte, config.RESULTS_DIR / "shots_K_test.csv")

    clf = SVC(kernel="precomputed")
    clf.fit(Ktr, ytr)
    pred_train = clf.predict(Ktr)
    pred_test = clf.predict(Kte)

    n_test_per_subclass = len(Xte) // 3
    metrics = common.compute_classification_metrics(
        ytr, pred_train, yte, pred_test, block_size=n_test_per_subclass
    )

    Ktr_ideal, Kte_ideal = _require_ideal_subset()
    metrics["relative_frobenius_train"] = common.relative_frobenius(Ktr, Ktr_ideal)
    metrics["relative_frobenius_test"] = common.relative_frobenius(Kte, Kte_ideal)
    metrics["max_symmetry_error_K_train"] = float(np.max(np.abs(Ktr - Ktr.T)))
    metrics["min_eigenvalue_K_train"] = common.min_eigenvalue(Ktr)
    metrics["shots"] = int(args.shots)
    metrics["seed"] = int(args.seed)

    common.save_json(metrics, config.RESULTS_DIR / "shots_metrics.json")

    print("\nFinite-shot metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.10f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
