# ML models — single-transistor leakage

Four models, one per case (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF), each predicting the four terminal currents
`id, ig, is, ib` of one transistor from the operating point and the process parameters.
Model file `45nm_MGK.pm`, nominal VDD 0.9 V, seed `12175305`, 20,000 samples per case (see
[`../Data_Generation/README.md`](../Data_Generation/README.md)).

## Result summary (test set, 2,000 samples per case, average over the four currents)

| Case | Chosen model | R² (log\|I\|) | Median error | 95th pct error | Mean error |
|---|---|---|---|---|---|
| NMOS_ON | MLP 128-64-32-32 | 0.9997 | **0.22 %** | **1.27 %** | 0.77 % |
| NMOS_OFF | MLP 128-64-32-16 + Kirchhoff loss (λ = 10) | 0.9852 | **0.40 %** | **2.60 %** | 6.67 % ¹ |
| PMOS_ON | MLP 128-64-32-16 | 0.9998 | **0.35 %** | **1.56 %** | 0.65 % |
| PMOS_OFF | MLP 128-64-32-32 + Kirchhoff loss (λ = 30) | 0.9921 | **0.36 %** | **2.13 %** | 1.30 % |

Error = |I_pred / I_SPICE − 1| × 100 for each sample and current (current magnitudes). The test set was used only once, in
[`Results/model_evaluation.ipynb`](Results/model_evaluation.ipynb), after all choices were made on the validation set.

* **Typical accuracy is the same for NMOS and PMOS:** median ≤ 0.40 % and 95th percentile ≤ 2.6 % in all four cases.
* ¹ **The NMOS_OFF mean is dominated by a few rows** (see [below](#why-the-nmos_off-mean-error-is-high-while-its-median-is-not)):
  the 5 worst of 2,000 test rows account for 72 % of the mean error of `id` and 95 % of that of `is`; without them the mean
  errors are 1.3 % (`id`) and 1.1 % (`is`). The mean is therefore not a reliable measure of NMOS_OFF accuracy.
* Transistor-level accuracy only. Gate-level and circuit-level accuracy (Parts B and C of the assignment) are not evaluated yet.
* **A follow-up study found a loss function that lowers the typical error of all four cases** (Huber instead of MSE; NMOS_OFF median 0.40 % → 0.13 %, 95th percentile 2.6 % → about 1.1 %), see
  [Follow-up study](#follow-up-study-nmos_off-error-structured-splits-and-a-better-loss). The models in this table are the official ones; the Huber models are in `Saved_Models/huber/`.

Per current, test set (median / 95th percentile / mean error in %):

| Current | NMOS_ON | NMOS_OFF | PMOS_ON | PMOS_OFF |
|---|---|---|---|---|
| `id` | 0.20 / 1.02 / 0.72 | 0.55 / 3.58 / 4.83 | 0.34 / 1.34 / 0.92 | 0.49 / 2.90 / 2.17 |
| `ig` | 0.23 / 1.64 / 0.73 | 0.38 / 1.97 / 0.75 | 0.28 / 1.73 / 0.54 | 0.31 / 1.68 / 0.78 |
| `is` | 0.20 / 1.09 / 1.21 | 0.54 / 3.26 / 20.29 | 0.34 / 1.39 / 0.51 | 0.48 / 2.67 / 1.18 |
| `ib` | 0.26 / 1.33 / 0.43 | 0.11 / 1.60 / 0.80 | 0.44 / 1.79 / 0.64 | 0.14 / 1.27 / 1.07 |
| average | 0.22 / 1.27 / 0.77 | 0.40 / 2.60 / 6.67 | 0.35 / 1.56 / 0.65 | 0.36 / 2.13 / 1.30 |

Full table: [`Results/test_results.csv`](Results/test_results.csv). Predicted-vs-SPICE plots (test set, log axes):
[NMOS_ON](Plots/parity_NMOS_ON.png), [NMOS_OFF](Plots/parity_NMOS_OFF.png), [PMOS_ON](Plots/parity_PMOS_ON.png),
[PMOS_OFF](Plots/parity_PMOS_OFF.png).

![NMOS_ON parity plot](Plots/parity_NMOS_ON.png)
![NMOS_OFF parity plot](Plots/parity_NMOS_OFF.png)

## Method (identical for all four cases)

| Step | Choice |
|---|---|
| Data | 20,000 rows per case: rows 1 – 10,000 are the original 10k dataset, rows 10,001 – 20,000 continue the same seed |
| Split | random 80 / 10 / 10 → 16,000 train / 2,000 validation / 2,000 test, `train_test_split(random_state=12175305)`; one split of the row indices shared by all models of a case |
| Inputs | 10 raw (`vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj`) + 15 physics features = 25. `vg` and `vb` are constant or equal to `vdd`, so they are not features |
| Physics features | terminal differences (`vgs, vgd, vds, vbs, vbd`); effective source (for NMOS the lower of `vd` / `vs`): `vgs_eff, vbs_eff, vds_abs`; sub-threshold terms `overdrive_vt = (vgs_eff − \|vth0\|)/Vt`, `log(1 − exp(−\|vds\|/Vt))`, `1/T`, `log w`; gate-tunnelling terms `1/toxe, vgs/toxe, vgd/toxe` |
| Targets | `log10 \|I\|` of `id, ig, is, ib` (each current spans 3 – 10 decades). The sign is not predicted; it is kept separately |
| Scaling | `StandardScaler` fitted on the training rows only (inputs; targets too for the MLP) |
| Selection | per case, the model with the lowest **validation mean error** (average over the four currents) |
| Test | the chosen model is evaluated once, on the test rows |

Model types compared (all four currents predicted at once):

| Model | Main settings |
|---|---|
| Random forest | 300 trees, `max_features=0.5`, `min_samples_leaf=1` |
| XGBoost | `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, up to 10,000 trees with early stopping (200 rounds) |
| Polynomial regression | polynomial features of the standardized inputs → `StandardScaler` → Ridge; degree 1 – 3 and alpha 1e-4 / 1e-2 / 1 / 100 searched on the validation set (best: degree 3, alpha 0.01 for both NMOS cases) |
| MLP | 25 → 128 → 64 → 32 → (16 or 32) → 4, SiLU, MSE on the standardized log targets, AdamW (lr 0.01, weight decay 1e-4), batch 256, learning rate halved after 50 epochs without validation improvement, early stopping after 300 epochs, at most 3,000 epochs; last hidden layer (16 / 32) chosen per case on the validation median error |
| MLP + Kirchhoff loss | the MLP above plus λ · mean[(Σ s_k·\|Î_k\| / Σ \|Î_k\|)²], which penalizes violations of `id + ig + is + ib = 0` using the true signs s_k during training; λ ∈ {10, 30, 100} chosen on the validation mean error |

The NMOS settings were **carried over from the PMOS study unchanged** (features, split, hyper-parameters, selection rule). Only
the choices that the training notebook itself makes on validation data were made again on NMOS data.

## Model comparison (validation set, 2,000 samples per case)

Each cell: R² (log\|I\|) / median / 95th percentile / mean error in %, averaged over the four currents.
Per-current values: [`Results/validation_comparison_NMOS.csv`](Results/validation_comparison_NMOS.csv),
[`Results/validation_comparison_PMOS.csv`](Results/validation_comparison_PMOS.csv).

| Model | NMOS_ON | NMOS_OFF | PMOS_ON | PMOS_OFF |
|---|---|---|---|---|
| Random forest | 0.9882 / 14.34 / 56.09 / 23.44 | 0.9199 / 10.16 / 47.46 / 201.05 | 0.9877 / 14.21 / 54.42 / 20.32 | 0.9226 / 9.75 / 49.83 / 29.17 |
| XGBoost | 0.9986 / 3.14 / 14.42 / 5.55 | 0.9818 / 3.45 / 20.70 / 69.06 | 0.9989 / 2.86 / 12.61 / 4.37 | 0.9851 / 3.33 / 19.42 / 13.43 |
| Polynomial (Ridge) | 0.9958 / 5.79 / 27.96 / 9.20 | 0.8692 / 13.67 / 67.03 / 244.18 | 0.9968 / 4.46 / 22.10 / 7.02 | 0.8683 / 14.59 / 77.30 / 69.96 |
| MLP | 0.9999 / **0.24** / **1.30** / **0.54** | 0.9763 / **0.40** / 2.96 / 8.52 | 1.0000 / **0.34** / **1.62** / **0.56** | 0.9897 / **0.33** / **2.02** / 1.83 |
| MLP + Kirchhoff loss | 0.9996 / 0.60 / 3.16 / 1.34 | 0.9903 / 0.41 / **2.83** / **7.25** | 0.9999 / 0.43 / 2.15 / 0.76 | 0.9915 / 0.35 / 2.25 / **1.50** |

Bold: best median / 95th percentile / mean in the column. The MLP is ahead of every other model type by roughly an order of
magnitude in median error in all four cases (e.g. NMOS_ON 0.24 % vs 3.14 % for the next best, XGBoost).

NMOS choices made on the validation set:

| | NMOS_ON | NMOS_OFF |
|---|---|---|
| Last hidden layer 16 / 32 (val. median, 95th) | 0.39 %, 1.69 % / **0.24 %, 1.30 %** | **0.40 %, 2.96 %** / 0.85 %, 4.88 % |
| λ = 10 / 30 / 100 (val. mean, median) | 1.58 % / **1.34 %** / 3.59 % (0.81 / 0.60 / 1.98 %) | **7.25 %** / 8.61 % / 22.84 % (0.41 / 0.65 / 4.37 %) |
| Chosen | plain MLP (the Kirchhoff term hurts for every λ) | MLP + Kirchhoff, λ = 10 |

The Kirchhoff term works as intended — the predictions obey `id + ig + is + ib = 0` 4.7× (ON) and 7.5× (OFF) better
(median relative residual 4.2e-4 → 8.9e-5 and 1.7e-3 → 2.2e-4) — but it only helps the *mean* of an OFF case.

## Chosen models

| File | Model | Parameters | Epochs trained | Validation median / 95th / mean | Test median / 95th / mean |
|---|---|---|---|---|---|
| `Saved_Models/NMOS_ON.pt` | MLP 128-64-32-32 | 14,852 | 1,194 | 0.24 / 1.30 / 0.54 % | 0.22 / 1.27 / 0.77 % |
| `Saved_Models/NMOS_OFF.pt` | MLP 128-64-32-16, λ = 10 | 14,260 | 3,000 (limit) | 0.41 / 2.83 / 7.25 % | 0.40 / 2.60 / 6.67 % |
| `Saved_Models/PMOS_ON.pt` | MLP 128-64-32-16 | 14,260 | 783 | 0.34 / 1.62 / 0.56 % | 0.35 / 1.56 / 0.65 % |
| `Saved_Models/PMOS_OFF.pt` | MLP 128-64-32-32, λ = 30 | 14,852 | 3,000 (limit) | 0.35 / 2.25 / 1.50 % | 0.36 / 2.13 / 1.30 % |

Each file is self-contained (weights, hidden layers, feature list, both scalers, dataset name, seed). The evaluation notebook
recomputes the validation error from every file and checks that it equals the value recorded at training time (it does, for all four).

## Why the NMOS_OFF mean error is high while its median is not

`id` and `is` of an OFF transistor are differences between opposite-signed terms (channel leakage, gate / junction leakage).
In NMOS_OFF 15 % of the `id` values are positive and 85 % negative, so the signed current crosses zero inside the sampled
space. Near a crossing |I| becomes tiny and `log10 |I|` dives; the smooth MLP (typical error ≈ 0.4 %) cannot follow a
dip that deep, and the percentage error of such a row is huge. [`Results/model_evaluation.ipynb`](Results/model_evaluation.ipynb)
(section 5, table [`Results/error_concentration.csv`](Results/error_concentration.csv)) quantifies this:

* **Concentration (test set):** the 5 worst rows (0.25 % of the rows) cause 72 % of the NMOS_OFF `id` mean (4.8 % → 1.3 % without them) and
  95 % of the `is` mean (20.3 % → 1.1 %). PMOS_OFF: 53 % (`id`) and 19 % (`is`). On the validation set the NMOS_OFF outlier is in `id`
  (one row with |id| = 2.9e-15 A next to |ig| = 1.6e-9 A and an error of 41,230 %), on the test set in `is`.
* **Dose-response (held-out rows, `id` and `is`, mean error by current size relative to the largest current of the sample):**

| \|I\| / largest current | 10 – 100 % | 1 – 10 % | 0.1 – 1 % | < 0.1 % |
|---|---|---|---|---|
| NMOS_OFF: pairs / mean error | 7,063 / 0.94 % | 859 / 4.0 % | 68 / ≈ 40 % | 10 / ≈ 9,200 % |
| PMOS_OFF: pairs / mean error | 7,299 / 0.76 % | 637 / 2.4 % | 51 / ≈ 43 % | 13 / ≈ 475 % |

  NMOS_ON and PMOS_ON have no pair below 1 %.
* **NMOS_OFF has deeper dips than PMOS_OFF** (smallest |id| 2.9e-15 A vs 4.7e-14 A; worst error 4.1e4 % vs 1.5e3 %), hence its larger mean.
* **The SPICE data is not the limit for most of these rows:** the data's own Kirchhoff residual is 2 – 5 % of the worst NMOS_OFF
  targets. One PMOS_OFF `ib` row (3.3e-16 A) is the exception: its residual is twice its value, i.e. at the solver's numerical floor.
* No rows were removed from the datasets (as for PMOS). Median and 95th percentile are the robust measures of typical accuracy.
  For NMOS_OFF the plain MLP and the Kirchhoff MLP are indistinguishable on those two measures (validation 0.40 % / 2.96 % vs 0.41 % / 2.83 %);
  the mean-based selection rule picked the Kirchhoff version on the strength of a few rows.

## Follow-up study: NMOS_OFF error, structured splits and a better loss

Questions: why is the NMOS_OFF mean error high, what is it in amperes, can it be lowered, and how do the models behave on widths and temperatures that were not in training? Written up for the report in
[`../Report/error_analysis_NMOS_OFF.md`](../Report/error_analysis_NMOS_OFF.md); every number is reproduced by [`Results/nmos_off_study.ipynb`](Results/nmos_off_study.ipynb) from the runs in `Results/offstudy/`
(trained by `Scripts/run_offstudy.py`, library `Scripts/offstudy.py`; stages are documented in the script header).

| Finding | Result |
|---|---|
| Why the mean is high | `id` / `is` at the lower-potential terminal are the difference of the channel current and the gate / junction leakage; where they nearly cancel (1.2 % of the values) the absolute error stays at 0.04 – 0.26 % of the row's largest current, so the percentage explodes. Median error is 0.47 % – 0.65 % in every temperature, width, vth0 and VDD bin |
| The mean is unstable | bootstrap of the NMOS_OFF `id` mean over test-size samples: 2.0 % – 46 %; of the median: 0.54 % – 0.60 % |
| In amperes | mean absolute error 65 pA for `id` / `is` on currents averaging 7 nA (0.9 %); 0.76 % weighted over all four currents; the worst percentage row is a 60 pA error on a 1.6e-13 A current |
| Signed-log target | worse: best of four settings halves the mean in one seed but triples the median and multiplies the ampere error of `id` by 4.7 |
| Huber loss | **median 0.40 / 1.08 / 0.62 % (MSE, 3 seeds) → 0.13 / 0.13 % (β = 0.002); 95th percentile 3.0 – 5.5 % → 1.0 – 1.2 %; the seed lottery disappears; PMOS_OFF, NMOS_ON, PMOS_ON improve too** |
| Learning curve (Huber) | test median 0.33 / 0.22 / 0.14 / 0.13 % for 4,000 / 8,000 / 12,000 / 16,000 training rows: flattening, more samples are not the main lever for the typical error |
| What the loss does not fix | the mean (3 – 6 %): one near-cancelled test row, the same in every run, contributes 1.5 – 8 points of it |
| Structured splits | interpolation in width or temperature is within the seed noise of the random split; widths extrapolate; temperature does not (OFF cases: median 14 % – 24 % above the trained range) |
| Seed noise | with the notebooks' MSE loss the test median of one network and split varies 0.35 % – 1.08 % with the training seed alone |

Final Huber models (β = 0.002; of the two trained seeds the one with the lower validation median error is kept: NMOS_ON 12175305, NMOS_OFF 12175306, PMOS_ON 12175306, PMOS_OFF 12175305; no candidate had a blown-up prediction). Test set, 2,000 rows, average over the four currents; the ampere-weighted error is Σ|I_pred − I_true| / Σ|I_true|. [`ML_Models/Results/huber_test_results.csv`](Results/huber_test_results.csv), parity plots [`ML_Models/Plots/parity_huber_models.png`](Plots/parity_huber_models.png).

| Case | Model | R² (log\|I\|) | Median error | 95th percentile | Mean error | Ampere-weighted error |
|---|---|---|---|---|---|---|
| NMOS_ON | official (MSE loss) | 0.9997 | 0.22 % | 1.27 % | 0.77 % | 0.19 % |
| NMOS_ON | **Huber loss** | 0.9997 | 0.10 % | 0.62 % | 0.58 % | 0.08 % |
| NMOS_OFF | official (MSE loss) | 0.9852 | 0.40 % | 2.60 % | 6.67 % | 0.76 % |
| NMOS_OFF | **Huber loss** | 0.9923 | 0.13 % | 0.99 % | 5.73 % | 0.35 % |
| PMOS_ON | official (MSE loss) | 0.9998 | 0.35 % | 1.56 % | 0.65 % | 0.36 % |
| PMOS_ON | **Huber loss** | 0.9998 | 0.09 % | 0.60 % | 0.31 % | 0.08 % |
| PMOS_OFF | official (MSE loss) | 0.9921 | 0.36 % | 2.13 % | 1.30 % | 0.71 % |
| PMOS_OFF | **Huber loss** | 0.9964 | 0.12 % | 0.86 % | 0.85 % | 0.25 % |

The median error falls by 2.3× (NMOS_ON) to 3.7× (PMOS_ON), the 95th percentile by 2.1× to 2.6×, the ampere-weighted error by 2.2× to 4.7×; the mean of NMOS_OFF (5.73 %) is still set by the one near-cancelled row of the report section 2.3 (its `is` mean is 20.1 %, 20.3 % for the official model; its `id` mean falls from 4.8 % to 2.3 %).

The Huber models were trained as the official ones (same network, features and 80 / 10 / 10 split, 3000 epochs) with the loss and the epoch selection changed (details and the transition-width sweep in the report). They are saved in
`Saved_Models/huber/<case>.pt`, load with `LeakageModel` like the official ones, and `Results/huber_test_results.csv` / `Plots/parity_huber_models.png` compare them with the official models. **The official models and
notebooks are unchanged**; adopting the Huber loss means re-running the two training notebooks with it.

## Using a saved model

```python
import sys; sys.path.insert(0, "ML_Models/Scripts")
from leakage_model import LeakageModel

model = LeakageModel("ML_Models/Saved_Models/NMOS_OFF.pt")
# df: columns vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj  (one row per transistor)
currents = model.predict(df)      # DataFrame with |id|, |ig|, |is|, |ib| in A (magnitudes, signs not predicted)
```

## Files

```
Notebooks/pmos_models.ipynb, nmos_models.ipynb   preprocessing, the five model types, selection, saving (one per device type)
Results/model_evaluation.ipynb                   validation comparison, chosen models, test-set results, parity plots, error diagnosis
Results/validation_comparison_{PMOS,NMOS}.csv    every model x case x current, validation set
Results/test_results.csv                         chosen models, test set
Results/error_concentration.csv                  share of each mean error caused by the 5 worst rows
Plots/parity_<case>.png                          predicted vs SPICE, test set
Saved_Models/<case>.pt                           the four chosen models
Scripts/leakage_model.py                         feature code, split, MLP builder, LeakageModel loader (used by Parts B and C)
Scripts/offstudy.py, run_offstudy.py             follow-up study: targets, splits, trainer (Huber loss, schedules), runs by stage
Scripts/export_huber_models.py                   choose the final Huber model per case on validation, compare with the official models, parity plots
Results/nmos_off_study.ipynb                     follow-up study: error analysis, signed-log, structured splits, Huber loss
Results/offstudy/{runs,preds,final}/             one JSON of metrics and one file of held-out predictions per training run
Results/huber_test_results.csv                   Huber vs official models, validation and test set
Saved_Models/huber/<case>.pt                     Huber-loss models (the official ones are in Saved_Models/)
Plots/off_error_vs_current_size.png, nmos_off_*.png, offstudy_splits.png, parity_huber_models.png   follow-up study figures
```

## Reproduce

Python 3 (developed under 3.12, re-run under 3.14) with the packages in [`../requirements.txt`](../requirements.txt) (CPU is enough). From the repository root, in order:

1. `Data_Generation/Scripts/generate_*_20k.py` (datasets, see the data README) and `Data_Generation/Data_Check/data_check.ipynb`
2. `ML_Models/Notebooks/pmos_models.ipynb` and `nmos_models.ipynb` (NMOS: about 27 minutes on 8 CPU cores, no GPU)
3. `ML_Models/Results/model_evaluation.ipynb`
4. Follow-up study (optional): `python3 ML_Models/Scripts/run_offstudy.py --stage N` for N = 1 … 13 (about 5 minutes per run; the finished runs are kept in `Results/offstudy/`), `python3 ML_Models/Scripts/export_huber_models.py`, then `Results/nmos_off_study.ipynb`

Open each notebook from inside the repository (they locate the root by searching for `Data_Generation/`). All random
seeds are `12175305` (split, scikit-learn / XGBoost `random_state`, `torch.manual_seed`). Training on CPU is deterministic for a
given library version; re-running with other versions can change individual results slightly (the PMOS models, trained
under Python 3.12, reproduce their recorded test metrics to within 1e-4 relative under Python 3.14 / torch 2.15).

## Limitations and open points

* **Single seed per configuration.** The training seed alone moves the NMOS_OFF test median between 0.35 % and 1.08 % (MSE loss, five seeds); differences between candidates that are small compared with that (for example the two NMOS_OFF MLPs of the
  model comparison) should not be over-read; λ = 100 was unstable in every case. With the Huber loss the spread is much smaller (0.13 % at both seeds).
* **Epoch limit.** The chosen NMOS_OFF model and the chosen PMOS_OFF model ran to the 3,000-epoch limit without early stopping.
* **Hyper-parameters were not re-tuned for NMOS** (deliberately, to keep NMOS and PMOS comparable).
* **Currents are predicted as magnitudes.** The sign (direction) of each current must come from the circuit context in Parts B and C.
* **Not evaluated yet:** gate-level and circuit-level leakage. Whether a badly predicted, strongly cancelled terminal current
  matters there depends on whether that terminal carries the supply current.
* **Tested without success:** a signed-log target, deriving the cancelling current from Kirchhoff's law, bigger networks, a seed ensemble (it matches the best seed but not a lower mean), the Kirchhoff term with the Huber loss.
  **Untested:** more samples near the sign changes, a loss that weights the near-cancelled rows, the structured splits with the Huber loss.
* **Extrapolation:** none of the models extrapolates in temperature beyond the trained range (OFF cases: median error 14 % – 24 %; the OFF body current is flat up to 85 °C and rises above it); widths extrapolate.
