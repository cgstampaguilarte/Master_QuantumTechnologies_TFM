"""
Run the full compute-uncompute kernel experiment on IBM Quantum hardware.

The script builds and transpiles all circuits required for the fixed
training and test subsets, submits them to the selected IBM backend,
reconstructs the kernel matrices and evaluates the precomputed-kernel SVC.

Before submission, the script displays the experiment configuration and
requires explicit user confirmation. IBM credentials must be configured
separately through ``QiskitRuntimeService`` and must never be stored here.

Outputs are written to ``results_tfm/``:
    - ibm_K_train.csv
    - ibm_K_test.csv
    - ibm_metrics.json
    - ibm_execution_metadata.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import platform
import sys
from pathlib import Path

import numpy as np
import qiskit
import qiskit_ibm_runtime
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfm_experiment import common, config



def transpile(circuits, backend):
    """
    Transpile the circuits for the selected backend and collect basic statistics.
    """
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=config.TRANSPILE_OPTIMIZATION_LEVEL,
        seed_transpiler=config.TRANSPILE_SEED,
    )
    transpiled_circuits = pm.run(circuits)
    depths = np.asarray([qc.depth() for qc in transpiled_circuits], dtype=int)
    two_qubits = np.asarray(
        [common.two_qubit_count(qc) for qc in transpiled_circuits], 
        dtype=int
    )
    stats = {
        "depth_min": int(depths.min()),
        "depth_mean": float(depths.mean()),
        "depth_max": int(depths.max()),
        "two_qubit_min": int(two_qubits.min()),
        "two_qubit_mean": float(two_qubits.mean()),
        "two_qubit_max": int(two_qubits.max()),
        "n_circuits": int(len(transpiled_circuits)),
    }
    return transpiled_circuits, stats


def run_sampler(transpiled_circuits, backend, shots: int) -> tuple[np.ndarray, dict]:
    """
    Run the transpiled circuits on the QPU and return kernel values and metadata.
    """
    sampler = SamplerV2(mode=backend)
    submitted_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    job = sampler.run(transpiled_circuits, shots=shots)
    job_id = job.job_id()
    print(f"Submitted job {job_id}. Waiting for results...")
    result = job.result()
    finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    values = np.empty(len(transpiled_circuits), dtype=float)
    for k, pub_result in enumerate(result):
        counts = pub_result.data.meas.get_int_counts()
        values[k] = common.fidelity_from_counts(counts)

    try:
        metrics = job.metrics()
    except Exception as exc:
        metrics = {"error": f"job.metrics() failed: {exc!r}"}

    metadata = {
        "job_id": job_id,
        "backend": backend.name,
        "shots": int(shots),
        "submitted_at": submitted_at,
        "finished_at": finished_at,
        "qiskit_version": qiskit.__version__,
        "qiskit_ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "python_version": platform.python_version(),
        "optimization_level": int(config.TRANSPILE_OPTIMIZATION_LEVEL),
        "seed_transpiler": int(config.TRANSPILE_SEED),
        "job_metrics": metrics,
    }
    return values, metadata


def _confirm(prompt: str) -> bool:
    """
    Return whether the user explicitly confirms the QPU submission.
    """
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _load_subset_info() -> dict:
    """
    Load the metadata describing the fixed experimental subset.
    """
    path = config.RESULTS_DIR / "subset_indices.json"
    if not path.exists():
        raise FileNotFoundError(
            "subset_indices.json not found. Run 02_prepare_subset.py first."
        )
    return common.load_json(path)


def run_hardware(args, Xtr, Xte, ytr, yte) -> None:
    """
    Build, transpile and execute the full subset kernel on IBM hardware.
    """
    feature_map = common.make_pauli_zz_feature_map()
    tr_pairs = common.train_pairs(len(Xtr))
    te_pairs = common.test_pairs(len(Xte), len(Xtr))

    train_circuits = [
        common.create_overlap_circuit(feature_map, Xtr[i], Xtr[j])
        for i, j in tr_pairs
    ]
    test_circuits = [
        common.create_overlap_circuit(feature_map, Xte[i], Xtr[j])
        for i, j in te_pairs
    ]
    all_circuits = train_circuits + test_circuits

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)

    subset_info = _load_subset_info()

    print("=" * 60)
    print("IBM QPU run summary")
    print("=" * 60)
    print(f"Backend                  : {backend.name} ({backend.num_qubits} qubits)")
    print(f"Subset seed              : {subset_info.get('seed')}")
    print(f"Subset train_per_class   : {subset_info.get('train_per_class')}")
    print(f"Subset test_per_subclass : {subset_info.get('test_per_subclass')}")
    print(
        f"Subset totals            : "
        f"{len(subset_info.get('train_indices', []))} train, "
        f"{len(subset_info.get('test_indices', []))} test"
    )
    print(f"Training circuits (i<j)  : {len(train_circuits)}")
    print(f"Test circuits            : {len(test_circuits)}")
    print(f"Total circuits           : {len(all_circuits)}")
    print(f"Shots per circuit        : {args.shots}")
    print("=" * 60)

    print("\nTranspiling circuits for the selected backend...")
    transpiled_circuits, stats = transpile(all_circuits, backend)

    print("\nTranspilation stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("=" * 60)

    if not args.yes and not _confirm(
        "These transpiled circuits will now be submitted to the QPU. "
        "Type 'yes' to proceed: "
    ):
        print("Aborted by user. No job was submitted.")
        return

    values, metadata = run_sampler(transpiled_circuits, backend, args.shots)
    tr_values = values[: len(train_circuits)]
    te_values = values[len(train_circuits):]

    Ktr = common.assemble_train_kernel(len(Xtr), tr_pairs, tr_values)
    Kte = common.assemble_test_kernel(len(Xte), len(Xtr), te_pairs, te_values)

    common.save_matrix_csv(Ktr, config.RESULTS_DIR / "ibm_K_train.csv")
    common.save_matrix_csv(Kte, config.RESULTS_DIR / "ibm_K_test.csv")

    clf = SVC(kernel="precomputed")
    clf.fit(Ktr, ytr)
    pred_train = clf.predict(Ktr)
    pred_test = clf.predict(Kte)

    n_test_per_subclass = len(Xte) // 3
    metrics = common.compute_classification_metrics(
        ytr, pred_train, yte, pred_test, block_size=n_test_per_subclass
    )

    Ktr_ideal, Kte_ideal = common.load_subset_ideal_kernels()
    metrics["relative_frobenius_train"] = common.relative_frobenius(Ktr, Ktr_ideal)
    metrics["relative_frobenius_test"] = common.relative_frobenius(Kte, Kte_ideal)
    metrics["max_symmetry_error_K_train"] = float(np.max(np.abs(Ktr - Ktr.T)))
    metrics["min_eigenvalue_K_train"] = common.min_eigenvalue(Ktr)
    metrics["shots"] = int(args.shots)

    common.save_json(metrics, config.RESULTS_DIR / "ibm_metrics.json")
    metadata["transpilation_stats"] = stats
    common.save_json(metadata, config.RESULTS_DIR / "ibm_execution_metadata.json")

    print("\nIBM run metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.10f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        type=str,
        default=config.IBM_BACKEND_USED,
        help="IBM backend used for the hardware experiment.",
    )
    parser.add_argument("--shots", type=int, default=config.SHOTS)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (use with care).",
    )
    args = parser.parse_args()

    Xtr, Xte, ytr, yte = common.load_subset_features_and_labels()

    run_hardware(args, Xtr, Xte, ytr, yte)


if __name__ == "__main__":
    main()
