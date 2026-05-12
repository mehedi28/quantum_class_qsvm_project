# quantum_class_qsvm_project

Quantum Support Vector Machine (QSVM) for stock market regime classification, benchmarked directly against a classical SVM with an RBF kernel. The quantum kernel is built from a ZZ feature map (forward pass followed by adjoint) and consumed by `sklearn.svm.SVC` via `kernel="precomputed"`.

Course project for the graduate Quantum Computing course at Texas Tech University, Whitacre College of Engineering (Spring 2026).

---

## Team

| Member | Role |
|---|---|
| Md Mehedi Hasan (R11919067) | Quantum kernel design and implementation, train/val/test split, result analysis, literature review |
| Rushikesh Reddy Bayyapu (R11870538) | Topic selection, data acquisition, feature analysis, ZZ feature map, result analysis |
| Afahan Javed (R11985175) | Project template, coordination, writing and editing, literature review, result analysis |

---

## Problem statement

Three-class daily regime detection on Apple Inc. (AAPL) returns:

- Class 0: Upward trend (top-tercile daily return)
- Class 1: Downward trend (bottom-tercile daily return)
- Class 2: Flat consolidation (middle tercile)

The data is AAPL daily OHLCV from 2019-01-01 to 2024-01-01 (about 1,238 trading days), pulled from Yahoo Finance via `yfinance`.

Two classifiers are compared on the same engineered features:

1. QSVM with a ZZ-feature-map quantum kernel
2. Classical SVM with an RBF kernel

The classical SVM is the baseline. The question is whether the entangled pairwise interactions in the ZZ feature map produce a kernel that separates regimes better than RBF on identical inputs.

---

## Pipeline

```
yfinance  ->  feature engineering  ->  regime labels  ->  MinMax scale to [0, 2π]
   |              |                       |                      |
   v              v                       v                      v
 AAPL OHLCV    Return,                 Up/Down/Flat       angle-encoded
 2019-2024     Volatility,             tercile cut         features ready
               RSI, Momentum                                for 4 qubits
                                                                |
                                                                v
                                                  +-----------------------+
                                                  |                       |
                                                  v                       v
                                             QSVM (ZZ kernel)    Classical SVM (RBF)
                                                  |                       |
                                                  +-----------+-----------+
                                                              |
                                                              v
                                                       Test acc / F1
                                                       Confusion matrix
                                                       Scaling curve
```

### Features (one per qubit)

| Feature | Definition |
|---|---|
| Return | Daily percentage change in closing price |
| Volatility | 20-day rolling standard deviation of returns |
| RSI | Relative Strength Index over a 14-day window |
| Momentum | Close - Close 20 days ago |

All four features are scaled with `MinMaxScaler` to the range [0, 2π] so they sit on the natural period of the Rz rotation on each qubit.

### Labels

Tercile cut on the daily return distribution: the lower 33rd percentile is labeled Downward (1), the upper 67th percentile Upward (0), the middle Flat (2).

### Splits

Stratified two-stage split with `random_state=42`:

- Train: 54%
- Validation: 13%
- Test: 33%

For attempts 6-9, smaller stratified subsets were drawn from the start of each split (100/200/300/500 training samples) so the kernel-matrix build stayed tractable on a laptop CPU.

---

## Quantum kernel

The kernel value between two samples x1 and x2 is computed as

```
K(x1, x2) = |<0...0 | U†(x2) U(x1) | 0...0>|²
```

estimated as the fraction of measurement shots that returned the all-zeros bitstring after running U(x1) followed by U†(x2) on the 4-qubit register. Each kernel entry uses 4,000 shots.

### ZZ feature map U(x)

Two stages on 4 qubits:

1. **Single-qubit encoding.** Apply Hadamard to each qubit, then `Rz(2·xᵢ)` on qubit i.
2. **Pairwise entanglement.** For every qubit pair (i, j), apply `CNOT(i,j)`, then `Rz(2·(π−xᵢ)·(π−xⱼ))` on qubit j, then `CNOT(i,j)`.

With 4 qubits this gives 6 entangling blocks. The adjoint U†(x) reverses gate order and negates every rotation angle.

### From kernel to classifier

The full train / validation / test kernel matrices are precomputed once. Training is then a one-line call to `sklearn`:

```python
clf = svm.SVC(kernel="precomputed", class_weight="balanced", random_state=42)
clf.fit(K_train, Y_train)
```

The classical baseline uses `svm.SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)` on the same feature subset.

---

## Results

Headline numbers from `Cell 17` (200 training samples, 100 test samples, 4,000 shots per kernel entry):

| Model | Test accuracy | Test F1 (weighted) |
|---|---|---|
| Classical SVM (RBF) | 0.720 | 0.715 |
| QSVM (ZZ kernel) | **0.740** | **0.738** |

QSVM gain over the classical baseline: +0.020 accuracy, +0.023 F1.

Per-class behavior: the classical SVM handled Up/Down well but collapsed on Flat (F1 ≈ 0.55). QSVM lifted Flat to F1 ≈ 0.69, a +0.14 gain, because the entangled feature interactions separated samples that looked identical to RBF in raw Euclidean space.

### Sample-size crossover

The crossover between classical and quantum sits between 100 and 200 training samples:

| Training samples | QSVM test acc | Classical SVM test acc |
|---|---|---|
| 100 | 0.583 | 0.700 |
| 200 | 0.740 | 0.720 |
| 300 | (see notebook Cell 22) | (see notebook Cell 22) |
| 500 | (see notebook Cell 24) | (see notebook Cell 24) |

The richer quantum feature space needs enough support vectors before it pays off. This crossover is the most actionable finding for anyone benchmarking QML on small datasets.

Figures generated by the notebook:

- `circuit_feature_map.png` - ZZ feature map forward pass U(x)
- `circuit_full_kernel.png` - Full kernel circuit U(x1) followed by U†(x2)
- `qsvm_results.png` - 2x2 panel: kernel heatmap, QSVM confusion matrix, per-class F1 comparison, sample-size crossover
- `confusion_matrices.png` - Side-by-side classical SVM vs QSVM confusion matrices at 200 samples
- `scaling_qsvm_3point.png` - QSVM-only scaling across 100/300/500 samples

---

## Development log (7 documented attempts)

The notebook logs every attempt in a running ASCII summary table (`Cell 6`, `show_summary()`). The failures are kept on purpose because they explain why the final design looks the way it does.

| # | Approach | Outcome |
|---|---|---|
| 1 | QNN, 30 samples, η=0.04, finite-difference SGD | Failed. Loss flat 0.24-0.27. Shot noise overwhelmed the gradient signal. |
| 2 | QNN, 80 samples, η=0.1, 30 epochs | Failed. Train acc 0.30-0.45, no learning trend. QNN abandoned. |
| 3 | QSVM, Rx/Rz kernel, features in [0, π] | Failed. Discrimination gap < 0.10. No entanglement. |
| 4 | QSVM, Rx/Rz kernel, features in [0, 2π] | Failed. Wider range helped but gap still below 0.15. No entanglement. |
| 5 | QSVM, ZZ kernel, **broken adjoint** (feature map applied twice) | Failed. Self-kernel ≈ 0.07 instead of 1.0. Mathematically invalid. |
| 6 | QSVM, ZZ kernel, **fixed adjoint**, 100 samples | Partial. QSVM test acc 0.583, classical 0.700. Below classical. |
| 7 | QSVM, ZZ kernel, fixed adjoint, **200 samples** | Success. QSVM test acc 0.740, classical 0.720. QSVM wins. |

Cells 21-24 extend the run to 300 and 500 training samples to map the scaling curve past the crossover.

### Key lessons

1. Entanglement is not optional. Without the CNOT blocks the ZZ kernel reduces to a product of single-qubit overlaps and cannot capture feature interactions.
2. Inverse circuits are easy to get wrong. The correct adjoint reverses gate order and negates every rotation angle. Always sanity-check `K(x, x) ≈ 1.0` before building the full kernel matrix.
3. QNN with finite-difference gradients did not work on 4-qubit shot-based simulation. Kernel methods sidestep the gradient-noise problem.
4. Quantum needs enough data. Reporting one sample size is not enough. The crossover only appeared once the training set crossed 100 samples.

---

## Repository structure

```
quantum_class_qsvm_project/
├── README.md                       # this file
├── project_code_stock_market.ipynb # full notebook: 25 cells, 7+ attempts
├── QSVM_Stock_Regime_Presentation.pptx
├── circuit_feature_map.png         # generated by Cell 13
├── circuit_full_kernel.png         # generated by Cell 13
├── qsvm_results.png                # generated by Cell 18
├── confusion_matrices.png          # generated by Cell 19
└── scaling_qsvm_3point.png         # generated by Cell 25
```

---

## Requirements

### Python packages

The notebook was developed and tested with:

| Package | Version |
|---|---|
| `numpy` | 2.4.4 |
| `pandas` | 3.0.2 |
| `matplotlib` | 3.10.8 |
| `seaborn` | 0.13.2 |
| `scikit-learn` | 1.8.0 |
| `qiskit` | 2.3.0 |
| `yfinance` | 1.2.2 |

Minimal `requirements.txt`:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
qiskit
yfinance
```

Install with:

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn qiskit yfinance
```

### Hardware / environment

- MacBook Air (M1, 8 GB RAM) was sufficient for the 100-sample and 200-sample runs.
- Python 3 inside Anaconda Navigator was used for development.
- Quantum execution used `qiskit.providers.basic_provider.BasicSimulator` (no real quantum hardware, no GPU required).

### Runtime expectations

Approximate wall-clock for the kernel-matrix builds on the MacBook Air:

| Sample size | Kernel evaluations | Approximate runtime |
|---|---|---|
| 100 train | 20,000 | 15-20 min |
| 200 train | 72,000 | 30-50 min |
| 300 train | 156,000 | 60-90 min |
| 500 train | 430,000 | 3-4 hours |

---

## How to run

1. Clone the repo:

```bash
git clone https://github.com/mehedi28/quantum_class_qsvm_project.git
cd quantum_class_qsvm_project
```

2. Set up a clean Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

3. Launch Jupyter and open the notebook:

```bash
jupyter notebook project_code_stock_market.ipynb
```

4. Run cells in order. Cells 1-20 reproduce the headline result (QSVM 0.740 vs classical 0.720 at 200 samples). Cells 21-25 extend to 300 and 500 samples and produce the QSVM-only scaling figure.

### Reproducibility

All splits use `random_state=42`. The Qiskit `BasicSimulator` is shot-based, so kernel values vary at the 1/sqrt(shots) level between runs; with 4,000 shots this is below 0.02 per kernel entry and does not change the qualitative result.

---

## Future work

- Run the same kernel on real IBM Quantum hardware to measure the cost of device noise versus the kernel-design gains seen here.
- Scale to 8+ qubits and bring in more features (longer-window RSI, MACD, volume-based features).
- Test richer quantum kernels (projected quantum kernels, trainable feature maps) for better expressivity.
- Repeat the benchmark on other tickers and on cryptocurrency time series to see whether the Flat-class advantage is specific to AAPL.

---

## References

1. V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, pp. 209-212, 2019. doi:10.1038/s41586-019-0980-2
2. M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces," *Phys. Rev. Lett.*, vol. 122, p. 040504, 2019. doi:10.1103/PhysRevLett.122.040504
3. P. Rebentrost, M. Mohseni, and S. Lloyd, "Quantum support vector machine for big data classification," *Phys. Rev. Lett.*, vol. 113, p. 130503, 2014.
4. Qiskit Contributors, "Qiskit: An Open-source Framework for Quantum Computing," IBM Quantum, 2024.
5. Yahoo Finance, AAPL daily OHLCV via `yfinance`, accessed 2024.

---

## Acknowledgments

Submitted as a course project for the Quantum Computing course at Texas Tech University, Whitacre College of Engineering, Spring 2026.
