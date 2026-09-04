"""
Ideal statevector reference for the TFM quantum-kernel experiment.

This script evaluates the selected Pauli-ZZ configuration (PCA=8,
reps=1) using exact statevector simulation, following the workflow of
Acosta et al. It computes the full training and test kernel matrices,
trains the SVM, and saves the reference data required by the subsequent
experiment steps.

Outputs are written to ``results_tfm/``:
    - reference_metrics.json
    - reference_pca_train.csv
    - reference_pca_test.csv
    - reference_K_train.csv
    - reference_K_test.csv
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfm_experiment import common, config


def main() -> None:
    common.ensure_dir(config.RESULTS_DIR)

    print("=" * 60)
    print("TFM reference statevector experiment (Pauli-ZZ, PCA=8, reps=1)")
    print("Based on Acosta et al., IEEE CAI 2026.")
    print("=" * 60)

    print("Loading dataset from", config.DATASET_ROOT)
    x_train_raw, x_test_raw, y_train, y_test = common.load_dataset()
    print(f"Raw shapes: X_train={x_train_raw.shape}, X_test={x_test_raw.shape}")
    common.assert_dataset_shapes(x_train_raw, x_test_raw, y_train, y_test)

    print(f"\nFitting PCA(n_components={config.PCA_COMPONENTS}) on training only.")
    x_train_pca, x_test_pca, _ = common.apply_pca(x_train_raw, x_test_raw)

    pca_columns_train = {
        f"pc_{i + 1}": x_train_pca[:, i] for i in range(config.PCA_COMPONENTS)
    }
    pca_columns_train["label"] = y_train
    common.save_labelled_table(
        config.RESULTS_DIR / "reference_pca_train.csv", pca_columns_train
    )
    pca_columns_test = {
        f"pc_{i + 1}": x_test_pca[:, i] for i in range(config.PCA_COMPONENTS)
    }
    pca_columns_test["label"] = y_test
    common.save_labelled_table(
        config.RESULTS_DIR / "reference_pca_test.csv", pca_columns_test
    )

    print(
        f"\nBuilding Pauli-ZZ feature map "
        f"(qubits={config.PCA_COMPONENTS}, reps={config.REPS}, alpha={config.FEATURE_MAP_ALPHA})."
    )
    feature_map = common.make_pauli_zz_feature_map()
    embedding_fn = partial(common.pauli_zz_circuit, feature_map=feature_map)

    print("Computing ideal statevectors (this is O(N) simulations)...")
    psi_train = common.precompute_statevectors(x_train_pca, embedding_fn)
    psi_test = common.precompute_statevectors(x_test_pca, embedding_fn)

    print("Assembling kernel matrices from precomputed states.")
    K_train = common.exact_kernel(psi_train, psi_train)
    K_test = common.exact_kernel(psi_test, psi_train)

    print("Saving kernel matrices as CSV (this may take a moment).")
    common.save_matrix_csv(K_train, config.RESULTS_DIR / "reference_K_train.csv")
    common.save_matrix_csv(K_test, config.RESULTS_DIR / "reference_K_test.csv")

    print("Training SVC(kernel='precomputed') on the ideal kernel matrix.")
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    pred_train = clf.predict(K_train)
    pred_test = clf.predict(K_test)

    metrics = common.compute_classification_metrics(
        y_train, pred_train, y_test, pred_test
    )
    metrics.update(
        {
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "pca_components": int(config.PCA_COMPONENTS),
            "reps": int(config.REPS),
            "feature_map": config.FEATURE_MAP_NAME,
        }
    )

    common.save_json(metrics, config.RESULTS_DIR / "reference_metrics.json")

    print("\nReference metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.10f}")
        else:
            print(f"  {key}: {value}")
    print("\nSaved everything to", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
