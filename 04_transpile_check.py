"""
Check transpilation of the compute-uncompute circuits for IBM hardware.

The script connects to IBM Quantum and transpiles a small set of subset
circuits using the same transpilation configuration as the hardware
experiment. No job is submitted and no QPU time is consumed.

Outputs are written to ``results_tfm/``:
    - transpile_report.json
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import qiskit
import qiskit_ibm_runtime
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tfm_experiment import common, config





def circuit_metrics(qc) -> dict:
    return {
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "two_qubit_gates": int(common.two_qubit_count(qc)),
        "ops": {str(k): int(v) for k, v in qc.count_ops().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        type=str,
        default=config.IBM_BACKEND_USED,
        help="IBM backend used for transpilation.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of subset samples used to build compute-uncompute pairs.",
    )
    args = parser.parse_args()

    common.ensure_dir(config.RESULTS_DIR)

    Xtr, _, _, _ = common.load_subset_features_and_labels()
    n = min(args.n_samples, len(Xtr))
    if n < 2:
        raise ValueError("Need at least two subset samples to build overlap circuits.")

    feature_map = common.make_pauli_zz_feature_map()

    pairs = common.train_pairs(n)
    logical_circuits = [
        common.create_overlap_circuit(feature_map, Xtr[i], Xtr[j]) for i, j in pairs
    ]

    print("Python:", platform.python_version())
    print("Qiskit:", qiskit.__version__)
    print("qiskit-ibm-runtime:", qiskit_ibm_runtime.__version__)

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    print(f"Selected backend: {backend.name} ({backend.num_qubits} qubits)")

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=config.TRANSPILE_OPTIMIZATION_LEVEL,
        seed_transpiler=config.TRANSPILE_SEED,
    )
    transpiled = pm.run(logical_circuits)

    report = {
        "environment": {
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
            "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__,
        },
        "backend": {
            "name": backend.name,
            "num_qubits": int(backend.num_qubits),
        },
        "feature_map": {
            "name": config.FEATURE_MAP_NAME,
            "n_features": int(config.PCA_COMPONENTS),
            "reps": int(config.REPS),
            "paulis": list(config.FEATURE_MAP_PAULIS),
            "entanglement": config.FEATURE_MAP_ENTANGLEMENT,
            "alpha": config.FEATURE_MAP_ALPHA,
        },
        "transpilation": {
            "optimization_level": int(config.TRANSPILE_OPTIMIZATION_LEVEL),
            "seed_transpiler": int(config.TRANSPILE_SEED),
        },
        "circuits": [],
    }

    for (i, j), logical, isa in zip(pairs, logical_circuits, transpiled):
        entry = {
            "pair": [int(i), int(j)],
            "logical": circuit_metrics(logical),
            "transpiled": circuit_metrics(isa),
        }
        report["circuits"].append(entry)
        print()
        print(f"pair ({i}, {j})")
        print("  logical:   ", entry["logical"])
        print("  transpiled:", entry["transpiled"])

    common.save_json(report, config.RESULTS_DIR / "transpile_report.json")
    print("\nSaved:", config.RESULTS_DIR / "transpile_report.json")
    print("No QPU job was submitted.")


if __name__ == "__main__":
    main()
