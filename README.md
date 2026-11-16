# Transaction Watch

An interactive fraud-scoring lab built from the original Real-time Transaction Monitoring System classroom project. It scores a single synthetic payment, reviews a CSV batch, and replays a labelled demonstration queue. The model is trained offline and shipped with the app, not trained again for each visitor.

**Try it:** https://real-time-transaction-monitoring-system-9ytqnem6bhnwwxak4wawtz.streamlit.app/

This is a portfolio demonstration, not a production banking service. It has no bank feed, payment blocking, automatic alerts, persistent transaction database, authentication or financial compliance certification. "Real-time" here means interactive inference on submitted inputs, not monitoring actual accounts.

## Run it

Python 3.10 or newer. The shipped experiment used Python 3.10.12. Package versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The existing Community Cloud deployment uses `updated_app_3.12.py`, which is now a small compatibility entrypoint for the same app. No local Windows paths, terminal prompts or startup training are needed.

## What you can try

- **Single scoring:** type, amount and simulation hour in, model score and review flag out.
- **Batch review:** CSV upload, strict validation, ranked review queue and scored CSV download. Maximum 10,000 rows and 2 MB.
- **Replay demo:** advance through 100 selected synthetic transactions ten at a time. This is a replay, not a live feed.
- **Model & evidence:** evaluation metrics, confusion matrix, feature choices and reproducibility record.

Use invented or sample data, not real customer records. The app has no application database and does not use submitted inputs to retrain. Session data and uploaded files are processed by the hosting service; hosting infrastructure/logs are outside this app's control. Do not treat this as a private financial-data service.

CSV input columns: `step,type,amount`. Types: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`. Step is an integer from 1 to 743; amount is a non-negative value in PaySim simulation currency units, not rupees or dollars. Extra columns are ignored and removed from scored exports. The model score is **not a calibrated real-world fraud probability**. A flag means review, not confirmed fraud; a low score does not mean safe.

## Training and honest results

Data: the public [PaySim synthetic mobile-money dataset](https://www.kaggle.com/datasets/ealaxi/paysim1), 6,362,620 rows and 8,213 simulated fraud labels. It is not real customer transaction data. Download mirror used for this build: [theman10/paysim](https://huggingface.co/datasets/theman10/paysim). Dataset SHA-256:

`16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`

Histogram gradient-boosted trees, 150 iterations, 15 leaves, learning rate 0.08, L2 regularization 10, seed 42. Features: log amount, cyclical hour of day, one-hot transaction type. No account IDs, existing fraud flags, raw simulation-hour trend, target labels or balance fields. The dataset publisher specifically warns that cancelled fraud affects balances, so **all four balance columns are excluded**. No SMOTE or balancing of the test set.

| Split | Simulation hours | Rows | Fraud labels |
|---|---|---:|---:|
| Training pool | 1-281 | 3,820,599 | 3,193 |
| Actual training sample (seed 42) | 1-281 | 1,000,000 | 795 |
| Validation | 282-355 | 1,293,285 | 770 |
| Untouched test | 356-743 | 1,248,736 | 4,250 |

There is no overlap of simulation hours between the splits. Threshold 0.0628855583 was selected on validation only: maximize F2 among thresholds with validation precision at least 10%. Validation precision was 40.63%, recall 38.31%. The final model was not refit on validation or test data.

| Later-hour test metric | Value |
|---|---:|
| Average precision (PR summary) | 0.3994 |
| No-skill average precision (fraud prevalence) | 0.0034 |
| ROC-AUC | 0.9504 |
| Precision at shipped threshold | 51.95% |
| Recall at shipped threshold | 40.35% |
| F1 | 0.4542 |

Confusion matrix at the shipped threshold:

| | Below threshold | Review flag |
|---|---:|---:|
| Actually normal | 1,242,900 | 1,586 |
| Actually fraud | 2,535 | 1,715 |

**The model misses 59.65% of fraud on this test window.** Nearly half of flagged payments are normal. That is a material limitation, not a production-ready result. Fraud prevalence changes substantially between windows; this temporal shift is intentionally left visible. Results are from one simulator run, without confidence intervals or independent-bank validation. Real-world data, new attack patterns and currency changes may behave very differently. High ROC-AUC alone is not a reason to trust the system.

`demo_transactions.csv` deliberately contains 80 normal and 20 fraud test examples. It is a teaching sample, not a benchmark or the expected fraud rate.

## Reproduce the experiment

Download the exact CSV from the linked source or mirror. Keep it outside the repository. Then, from this directory:

```bash
python train.py --csv /path/to/paysim.csv
```

The script verifies the dataset checksum, takes the fixed training sample, fits the model, selects the threshold on validation only, then writes `model.joblib`, `metrics.json` and `demo_transactions.csv`. `metrics.json` includes exact counts, package versions, confusion matrix and model checksum. Numerical results should be checked when changing Python, packages, platform or training settings. The app verifies the shipped model's checksum before loading it; only load trusted model files because joblib/pickle can execute code.

Smoke tests:

```bash
python -m unittest test_app.py
```

The original script's repeated training fragments, broken upload string, undefined variables, machine-specific CSV paths, and console input loops were replaced. The original work remains in Git history, rather than being silently presented as a working deployed model.

## What production would require

A permissioned transaction feed, secure API/authentication, data governance, account-aware and temporal features, latency/load measurements, calibrated decisions, investigation tooling, monitoring for drift, independent evaluation, and a human-approved false-positive/missed-fraud policy. This demo supplies none of those guarantees. It must not be used to approve, decline or investigate real people automatically.

## Credits

Original classroom project: **POOJA J (1KG22CB045)** and **PRAKRUTHI GOWDA D (1KG22CB047)**. Both original README credits and the original commit history are preserved. The current app is a rebuilt, reproducible demonstration of that project.

Dataset citation: E. A. Lopez-Rojas, A. Elmir and S. Axelsson, "PaySim: A financial mobile money simulator for fraud detection," EMSS, 2016. [Original simulator](https://github.com/EdgarLopezPhD/PaySim). The dataset page lists CC BY-SA 4.0; the included 100-row derived demonstration sample follows that attribution/share-alike license. This does not relicense the original project code.

[Original project demonstration recording](https://drive.google.com/file/d/1nhIcrXXrLftvEA8CF9kQmupYBU5KgMVK/view?usp=sharing) (historical reference, not evidence of the current deployment).
