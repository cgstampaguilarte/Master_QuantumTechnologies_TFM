# Quantum-kernel experiment for entanglement detection

This repository contains the implementation and archived results of the
quantum-kernel experiment developed for the Master's Thesis.

The experiment studies binary entanglement detection in 3x3
bipartite quantum systems using a classical SVM with a precomputed
Pauli-ZZ quantum kernel. The selected configuration is evaluated under
three conditions:

1. exact statevector simulation;
2. finite-shot noiseless sampling;
3. execution on IBM Quantum hardware.

The final configuration uses PCA with 8 components, one feature-map
repetition, 512 shots per estimated kernel entry and the
`ibm_marrakesh` backend.

## Repository structure

```text
tfm_experiment/
├── README.md
├── requirements.txt
├── __init__.py
├── config.py
├── common.py
├── 01_reference_statevector.py
├── 02_prepare_subset.py
├── 03_finite_shots.py
├── 04_transpile_check.py
├── 05_ibm_kernel.py
├── 06_analyse_results.py
├── results_tfm/
└── figures_tfm/
```

`config.py` contains the common experimental parameters and `common.py`
contains the shared utilities used by the six execution scripts.

The numerical outputs of the final experiment are kept in
`results_tfm/`, while the final comparison figures are stored in
`figures_tfm/`.

## Software environment

The final experiment was performed with Python 3.14.3 and the package
versions listed in `requirements.txt`.

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The dataset is not redistributed with this repository.

The experiments use the 3x3 entanglement dataset introduced by
Martínez-Sabiote et al. and subsequently used by Acosta et al.

Source repositories:

- https://github.com/anamarsabi/QML-Research
- https://github.com/QTCG-UGR/qsvm_paulifm

To reproduce the complete pipeline, place the required files inside the
repository using the following structure:

```text
dataset/
└── efficiency_study/
    ├── train/
    │   ├── x_n_500.csv
    │   └── y_n_500.csv
    └── test/
        ├── x_test.csv
        └── y_test.csv
```

The expected files contain 500 training samples and 300 test samples,
with 80 real-valued features per sample.

## Execution

The scripts are intended to be executed in numerical order from the
repository root.

### 1. Exact statevector reference

```bash
python 01_reference_statevector.py
```

Loads the complete dataset, fits PCA on the training set, computes the
exact statevector kernels and evaluates the precomputed-kernel SVM.

### 2. Fixed experimental subset

```bash
python 02_prepare_subset.py
```

Selects the fixed training and test subsets using the seed defined in
`config.py` and extracts their ideal kernel matrices.

### 3. Finite-shot evaluation

```bash
python 03_finite_shots.py
```

Recomputes the subset kernel using compute-uncompute circuits and
finite-shot noiseless sampling.

### 4. Transpilation check

```bash
python 04_transpile_check.py
```

Optional pre-flight check for the IBM backend. Circuits are transpiled
using the final hardware configuration, but no QPU job is submitted.

### 5. IBM Quantum execution

```bash
python 05_ibm_kernel.py
```

Requires a locally configured `QiskitRuntimeService`. No IBM Quantum
credentials are stored in the repository.

The script displays the complete configuration and requests explicit
confirmation before submitting the QPU job.

The archived hardware results in `results_tfm/` correspond to the final
TFM execution on `ibm_marrakesh`. Running this script again constitutes
a new physical realization of the experiment and may overwrite the
stored IBM outputs.

### 6. Final analysis

```bash
python 06_analyse_results.py
```

Reads the stored ideal, finite-shot and IBM results and regenerates the
final comparison figures and `metrics_summary.csv`.

This step does not recompute kernels and does not require QPU access.

## Reproducibility

The repository includes the numerical outputs of the final experiment.
Therefore, the analysis and figures can be reproduced directly from the
archived files without repeating the IBM Quantum execution.

The complete pipeline from the original input data requires the external
dataset described above.

A new hardware execution is not expected to reproduce the archived
kernel values exactly, since measurement outcomes and device conditions
may vary between runs.

## Related work

The implementation builds on the quantum-kernel workflow and dataset
used in:

- Martínez-Sabiote et al., *Machine Learning Based Detection of
  Entanglement*.
- Acosta et al., *Quantum Kernels for SVM Entanglement Detection*,
  IEEE CAI 2026.

The original third-party datasets and source implementations are not
redistributed here.

## Author

Clara G. Stampa Guilarte

Master's Thesis supervised by Daniel Manzano Diosdado  
Master's Degree in Quantum Technologies, UIMP.