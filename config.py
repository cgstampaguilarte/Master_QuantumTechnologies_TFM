"""Central configuration for the quantum-kernel experiment."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "tfm_experiment" / "dataset" / "efficiency_study"
RESULTS_DIR = PROJECT_ROOT / "tfm_experiment" / "results_tfm"
FIGURES_DIR = PROJECT_ROOT / "tfm_experiment" / "figures_tfm"

TRAIN_SIZE = 500
TEST_SIZE = 300
RAW_FEATURES = 80

PCA_COMPONENTS = 8

FEATURE_MAP_NAME = "Pauli-ZZ"
FEATURE_MAP_PAULIS = ["ZZ"]
FEATURE_MAP_ENTANGLEMENT = "full"
FEATURE_MAP_ALPHA = 2.0
REPS = 1

SUBSET_SEED = 42
SUBSET_TRAIN_PER_CLASS = 10
SUBSET_TEST_PER_SUBCLASS = 10

SHOTS = 512
IBM_BACKEND_USED = "ibm_marrakesh"

TRANSPILE_OPTIMIZATION_LEVEL = 3
TRANSPILE_SEED = 42

LABEL_SEP = 0
LABEL_ENTANGLED = 1
