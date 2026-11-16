"""Rebuild the shipped model and report. Run: python train.py --csv /path/to/paysim.csv"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)
from fraud_core import features

EXPECTED_SHA256 = '16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()
    hasher = hashlib.sha256()
    with open(args.csv, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(chunk)
    sha = hasher.hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError('Dataset checksum differs. Refusing to label this the same experiment.')
    data = pd.read_csv(args.csv, usecols=['step', 'type', 'amount', 'isFraud'])
    # Fixed disjoint simulation-hour windows. No future-hour training rows.
    train_all = data[data.step <= 281]
    train = train_all.sample(n=1000000, random_state=42).sort_index()
    val = data[(data.step > 281) & (data.step <= 355)]
    test = data[data.step > 355]
    model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=.08,
                                           l2_regularization=10, early_stopping=False, random_state=42)
    model.fit(features(train), train.isFraud)
    vp = model.predict_proba(features(val))[:, 1]
    precision, recall, thresholds = precision_recall_curve(val.isFraud, vp)
    f2 = 5 * precision[:-1] * recall[:-1] / (4*precision[:-1] + recall[:-1] + 1e-15)
    # Validation-only policy selection: emphasize recall, but require >=10% precision.
    eligible = precision[:-1] >= .10
    scores = np.where(eligible, f2, -1)
    idx = int(np.argmax(scores if eligible.any() else f2))
    threshold = float(thresholds[idx])
    p = model.predict_proba(features(test))[:, 1]
    pred = p >= threshold
    def count(d):
        return {'rows': len(d), 'fraud': int(d.isFraud.sum()), 'fraud_rate': float(d.isFraud.mean()),
                'step_min': int(d.step.min()), 'step_max': int(d.step.max())}
    report = {'dataset': {'sha256': sha, 'rows': len(data), 'fraud': int(data.isFraud.sum()),
                          'source': 'https://www.kaggle.com/datasets/ealaxi/paysim1',
                          'mirror': 'https://huggingface.co/datasets/theman10/paysim'},
              'train_pool': count(train_all), 'train': count(train), 'validation': count(val), 'test': count(test),
              'seed': 42, 'features': ['log1p(amount)', 'hour_sin', 'hour_cos'] + list(['type_'+t for t in ['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']]),
              'threshold': threshold, 'threshold_policy': 'Maximum validation F2 among thresholds with validation precision >= 0.10.',
              'validation_precision': float(precision[idx]), 'validation_recall': float(recall[idx]),
              'test_metrics': {'average_precision': float(average_precision_score(test.isFraud, p)),
                               'roc_auc': float(roc_auc_score(test.isFraud, p)),
                               'precision': float(precision_score(test.isFraud, pred, zero_division=0)),
                               'recall': float(recall_score(test.isFraud, pred, zero_division=0)),
                               'f1': float(f1_score(test.isFraud, pred, zero_division=0)),
                               'confusion_matrix_tn_fp_fn_tp': confusion_matrix(test.isFraud, pred).ravel().tolist(),
                               'no_skill_average_precision': float(test.isFraud.mean())},
              'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}}
    joblib.dump(model, 'model.joblib', compress=3)
    report['model_sha256'] = hashlib.sha256(Path('model.joblib').read_bytes()).hexdigest()
    Path('metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    # Small, explicitly selected demonstration set, not an evaluation sample.
    demo = pd.concat([test[test.isFraud == 0].sample(80, random_state=42),
                      test[test.isFraud == 1].sample(20, random_state=42)]).sample(frac=1, random_state=42)
    demo.to_csv('demo_transactions.csv', index=False)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
