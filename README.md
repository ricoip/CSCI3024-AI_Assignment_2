# Probabilistic digit recognition with evidential uncertainty

An AI-generated, local implementation for **CISC3024 AI Assignment #2**.
Selected method: Duan et al., **Evidential Uncertainty Quantification:
A Variance-Based Perspective**, WACV 2024.
[Publication](https://openaccess.thecvf.com/content/WACV2024/html/Duan_Evidential_Uncertainty_Quantification_A_Variance-Based_Perspective_WACV_2024_paper.html)
| [Paper](https://arxiv.org/html/2311.11367v1)
| [Authors' implementation](https://github.com/KerryDRX/EvidentialADA)

A neural network predicts parameters of a Dirichlet distribution over class
probabilities. The law of total covariance separates aleatoric and epistemic
uncertainty and gives class-level uncertainty and between-class correlation.
This is based on **probability theory**; the network weights are deterministic,
so this is not a Bayesian posterior over weights.

## Run locally

Python 3.12 was used. The installed `.venv` is ready:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.experiment
.venv/bin/python -m src.predict --index 0
.venv/bin/python -m src.report
```

For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records the exact experiment environment.
`requirements.txt` specifies compatible ranges for other environments.
The dataset is bundled with scikit-learn; no separate download or GPU is needed.

Quick trial, in a separate directory so the full results remain available:

```bash
.venv/bin/python -m src.experiment --epochs 30 --seeds 42 --output results/quick
```

To predict your own **8x8 grayscale** digit (bright strokes on dark background):

```bash
.venv/bin/python -m src.predict --image digit.png
```

Use `--invert` for dark strokes. These images should resemble the bundled
optical digit images; prediction quality on arbitrary drawings is unverified.
The `--index` demo can use training images. Experiment metrics use only saved
test indices, so the demo is not an independent evaluation.

## Implemented method

For K classes, the model produces evidence `e = softplus(z)`, concentration
`alpha = e + 1`, strength `S = sum(alpha)`, and predictive mean `p = alpha/S`.
The unit offset is a uniform prior, and prediction is `argmax(p)`.

The supervised loss is the paper's NLL plus masked-Dirichlet KL regularization:

```text
L = log(S) - log(alpha_true) + lambda * KL(Dir(masked_alpha) || Dir(ones))
masked_alpha = y + (1-y)*alpha
```

Let `C = diag(p) - p p^T`. Paper Eqs. 19-27 become:

```text
Total covariance       = C
Epistemic covariance   = C / (S+1)
Aleatoric covariance   = S*C / (S+1)
Class uncertainties    = covariance diagonal
Sample uncertainties   = covariance trace
```

`src/evidential.py` also implements correlation (Eq. 28) and the entropy
comparison (Eqs. 8-10). Larger strength reduces epistemic uncertainty at fixed
predictive mean. A negative between-class correlation arises from mutually
exclusive categorical outcomes; it should not be read as a measured confusion
rate between classes.

## Experiment protocol

- Data: 1,797 optical handwritten digit images, 8x8, 10 classes; pixels divided
  by 16. This is **not MNIST**.
- Each seed: stratified 1,077 training / 360 validation / 360 test examples.
- Identical 64-128-10 ReLU MLP backbone, initial parameters, and training order
  for softmax and evidential models within each seed.
- Adam, learning rate 0.003, batch size 64, 250 epochs. Evidential KL coefficient
  rises linearly to 0.01 in the first 10 epochs.
- Restore the epoch with lowest clean validation predictive NLL. No test data
  used for training or selection.
- Evaluate clean test data and Gaussian pixel noise with standard deviations
  0.3 and 0.6, clipped to [0,1]. Within a seed the same noise array is used for
  both methods and scaled across noise levels.
- Report accuracy, macro F1, predictive NLL, multiclass Brier score (sum over
  classes), and 15-bin ECE. AUROC uses misclassification as the positive class;
  larger uncertainty should identify mistakes. It is not OOD-detection AUROC.
- Seeds 42, 43, 44; report mean and **sample standard deviation**, not confidence
  intervals. The random splits overlap across runs, so runs are not independent
  datasets. No hyperparameter search or significance claim is made.

## Measured results

Mean +/- sample SD over seeds 42, 43, 44:

| Condition | Softmax accuracy | Evidential accuracy | Softmax NLL | Evidential NLL |
|---|---:|---:|---:|---:|
| Clean | 97.04 +/- 0.16% | 96.76 +/- 0.42% | 0.09 +/- 0.01 | 0.16 +/- 0.01 |
| Noise SD 0.3 | 80.56 +/- 3.74% | 80.28 +/- 1.55% | 0.80 +/- 0.29 | 0.74 +/- 0.04 |
| Noise SD 0.6 | 43.15 +/- 2.94% | 44.26 +/- 1.05% | 4.24 +/- 1.34 | 2.11 +/- 0.06 |

Softmax has better clean probability metrics. The evidential model has lower
mean predictive NLL under noise here, but similar accuracy. Its variance-based
epistemic error-detection AUROC is 0.970 on clean data and 0.673 under severe
noise. These small experiments do not establish general superiority.

Verification: 9 numerical/behavioral tests passed. All 18 saved prediction sets
reproduce the reported classification metrics; all six clean checkpoints
reproduce their saved predictions. See `results/verification.txt`.

## Outputs and source map

- `output/pdf/assignment_report.pdf`: report draft generated from measured results.
- `results/metrics.json` and `metrics.csv`: all per-run metrics and mean/SD.
- `results/history.json`: training/validation histories.
- `results/*.png`: accuracy, confusion, uncertainty, and class analysis plots.
- `results/*.npz`: trained models, split indices, and prediction arrays.
- `src/evidential.py`: formulas, analytic gradients, MLP, Adam, checkpoints.
- `src/experiment.py`: leakage-free splits, paired experiments, metrics, plots.
- `src/predict.py`: prediction CLI.
- `src/report.py`: reproducible PDF report builder.
- `tests/test_evidential.py`: numerical and experiment integrity checks.
- `docs/research_log.md`: AI search and implementation provenance.

## Scope and remaining report fields

This implements the paper's **classification and uncertainty core**, with a
small MLP and digits dataset. It does not implement the full active domain
adaptation loop, unsupervised uncertainty guidance, certainty pseudo-labeling,
or Office-Home/VisDA benchmarks. Both variance and entropy calculations use
the same evidential classifier and therefore have identical classification
accuracy; they differ in uncertainty scoring.

Author/student ID, personal learning reflections, and the source-code webpage
field are blank as requested. All work remains local. Before submission, supply
those fields and publish the source to obtain the webpage link required by the
assignment. The original reference repository is not a link to this project's
source. No Moodle upload or public publication has been performed.
