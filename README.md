# quantum_class_qsvm_project

Quantum Support Vector Machine (QSVM) for stock market regime classification, compared against a classical SVM. Uses a ZZ feature map quantum kernel on Apple (AAPL) daily returns from 2019-2024.

Course project for the Quantum Computing course at Texas Tech University (Spring 2026).

## Team

| Member | Role |
|---|---|
| Md Mehedi Hasan (R11919067) | Split train/test/validation set, Quantum kernel design and implementation, Built the ZZ feature map, Result analysis, Literature review |
| Rushikesh Reddy Bayyapu (R11870538) | Topic selection, Data acquisition, Feature analysis, Result analysis, Literature review, Background study |
| Afahan Javed (R11985175) | Prepare project template, Project coordination, Team meeting management, Literature review, Writing and editing, Result analysis |

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

## How to run

The project was developed using **Anaconda Navigator** on a MacBook Air M1.

1. Clone the repository:

   ```bash
   git clone https://github.com/mehedi28/quantum_class_qsvm_project.git
   cd quantum_class_qsvm_project
   ```

2. Open **Anaconda Navigator**, go to the **Environments** tab, click **Create**, name it `qsvm`, and select **Python 3.11**.

3. With `qsvm` selected, click the play icon and choose **Open Terminal**. Then install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Go to the **Home** tab in Anaconda Navigator, make sure `qsvm` is the active environment, and **Launch** Jupyter Notebook.

5. Open `project_code_stock_market.ipynb` and run the cells in order. Use **Shift + Enter** to run one cell at a time, or click **Cell > Run All** to run the full notebook end to end.

The headline result is produced by Cell 17. Total runtime is about 45-60 minutes on a laptop CPU.

## Project files

| File | Description |
|---|---|
| `project_code_stock_market.ipynb` | Main Jupyter notebook with 25 cells covering data download, feature engineering, all 7 development attempts, QSVM training, and result figures. |
| `requirements.txt` | Pinned package versions used to reproduce the results. |
| `QSVM_Stock_Regime_Presentation.pptx` | Final project presentation slides. |

## Requirements

| Package | Version |
|---|---|
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| scikit-learn | 1.8.0 |
| qiskit | 2.3.0 |
| yfinance | 1.2.2 |

See `requirements.txt`.

---

## Reproducibility

All splits use `random_state=42`.
---

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

| Model | Test Accuracy | Test F1 |
|---|---|---|
| Classical SVM (RBF) | 0.720 | 0.715 |
| QSVM (ZZ kernel, 200 samples) | **0.740** | **0.738** |

QSVM lifted the Flat-class F1 from 0.55 to 0.69. The crossover between classical and quantum sits between 100 and 200 training samples.

---

### Sample-size crossover

The crossover between classical and quantum sits between 100 and 200 training samples:

| Training samples | QSVM test acc | Classical SVM test acc |
|---|---|---|
| 100 | 0.583 | 0.700 |
| 200 | 0.740 | 0.720 |
| 300 | (see notebook Cell 22) | (see notebook Cell 22) |
| 500 | (see notebook Cell 24) | (see notebook Cell 24) |

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


## What each cell does

The notebook is meant to be run top-to-bottom on the first pass. Each block below describes what one group of cells does and what to expect.

**Cells 1-5: Setup and data preparation (under 30 seconds)**

- Cell 1 imports all libraries and prints package versions.
- Cell 2 downloads AAPL daily OHLCV from 2019-01-01 to 2024-01-01 via `yfinance`.
- Cell 3 builds the four engineered features: Return, 20-day Volatility, 14-day RSI, 20-day Momentum.
- Cell 4 assigns the three-class regime label using the 33rd and 67th percentile of daily returns.
- Cell 5 scales features to [0, 2π] with `MinMaxScaler` and produces stratified train/val/test splits with `random_state=42`.

**Cell 6: Attempt tracker**

Defines the `attempt_log` list and the `show_summary()` printer. Every subsequent attempt appends one row, so the running table is reprinted after each experiment.

**Cells 7-8: QNN attempts (Attempts 1-2, failed)**

A small variational quantum neural network trained with finite-difference SGD. These were left in the notebook on purpose to document why the project moved to kernel methods. Each cell takes a few minutes; output is mostly flat loss curves.

**Cells 9-12: Early QSVM kernel probes (Attempts 3-5, failed)**

Defines the QSVM infrastructure (`build_kernel_matrix`, `evaluate_qsvm`), then tests three broken or under-expressive kernels. Each one ends with a discriminability check (self-kernel vs cross-kernel gap). The diagnostic prints are the point of these cells: they show why the final ZZ kernel with a correct adjoint is needed.

**Cell 13: The working ZZ kernel (Attempt 6 setup)**

Defines `quantum_kernel(x1, x2)` with the correct U(x1) followed by U†(x2) structure. Also draws the feature map and full kernel circuit and saves them as `circuit_feature_map.png` and `circuit_full_kernel.png`. This is the kernel function used by every later cell.

**Cells 14-15: First successful QSVM run at 100 training samples (Attempt 6)**

Cell 14 builds the train (100x100), validation (40x100), and test (60x100) kernel matrices. This step is the main time cost: about 15-20 minutes on a MacBook Air M1. Cell 15 trains the QSVM on the precomputed kernel and a classical RBF SVM on the same features for comparison. Expected output: QSVM test acc near 0.58, classical SVM near 0.70.

**Cells 16-17: Headline result at 200 training samples (Attempt 7)**

Cell 16 rebuilds the kernel matrices at 200 train, 60 val, 100 test (about 30-50 minutes). Cell 17 produces the headline numbers: QSVM test accuracy 0.740, classical SVM 0.720, with QSVM lifting the Flat-class F1 from about 0.55 to about 0.69.

**Cells 18-20: Figures and final summary**

- Cell 18 saves `qsvm_results.png`, the 2x2 panel with the kernel heatmap, QSVM confusion matrix, per-class F1 comparison, and sample-size crossover.
- Cell 19 saves `confusion_matrices.png`, classical vs QSVM side by side.
- Cell 20 prints the final comparison table and the full attempt log.

**Cells 21-25: Scaling beyond 200 samples (Attempts 8-9, optional)**

These extend the run to 300 and 500 training samples. The 300-sample build takes about an hour; the 500-sample build takes 3-4 hours. Cell 25 produces `scaling_qsvm_3point.png`, the QSVM-only scaling curve across 100/300/500. Skip these on a first pass if compute time is tight.

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