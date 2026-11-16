"""Build compact retrospective account analytics. Never trains or changes the model.
Run: python build_account_analysis.py --csv /path/to/paysim.csv
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from fraud_core import features
from train import EXPECTED_SHA256

ROOT = Path(__file__).parent
ROLES = {'origin': 'nameOrig', 'destination': 'nameDest'}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def aggregate_events(events):
    """Only labelled/flagged events are needed to identify repeat candidates."""
    return events.groupby(['role', 'account']).agg(
        fraud_count=('isFraud', 'sum'), review_count=('review_flag', 'sum'),
        max_score=('model_score', 'max')).reset_index()

def prior_fraud_steps(events):
    """Strictly earlier hours only. File order within an hour is not evidence of time."""
    first = events.loc[events.isFraud == 1, 'step'].min()
    return (events.step > first).fillna(False).astype(bool)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()
    if sha256(args.csv) != EXPECTED_SHA256:
        raise ValueError('Dataset checksum differs from the pinned experiment.')
    report = json.loads((ROOT/'metrics.json').read_text())
    if sha256(ROOT/'model.joblib') != report['model_sha256']:
        raise ValueError('Model integrity check failed.')
    model = joblib.load(ROOT/'model.joblib')
    selected = []
    rows = fraud = flagged = 0
    for chunk in pd.read_csv(args.csv, usecols=['step','type','amount','nameOrig','nameDest','isFraud'], chunksize=200000):
        chunk['row_id'] = chunk.index + 1  # one-based data row, header excluded
        with threadpool_limits(limits=1):
            chunk['model_score'] = model.predict_proba(features(chunk))[:, 1]
        chunk['review_flag'] = chunk.model_score >= report['threshold']
        rows += len(chunk); fraud += int(chunk.isFraud.sum()); flagged += int(chunk.review_flag.sum())
        interesting = chunk[(chunk.isFraud == 1) | chunk.review_flag]
        for role, col in ROLES.items():
            part = interesting[['row_id','step','type','amount','isFraud','model_score','review_flag']].copy()
            part['role'] = role
            part['account'] = interesting[col]
            selected.append(part)
    events = pd.concat(selected, ignore_index=True)
    aggregates = aggregate_events(events)
    candidates = aggregates[(aggregates.fraud_count >= 2) | (aggregates.review_count >= 2)].copy()
    keys = set(zip(candidates.role, candidates.account))
    # Second pass gets all transactions for candidates, not just fraud/flag rows.
    history = []
    for chunk in pd.read_csv(args.csv, usecols=['step','type','amount','nameOrig','nameDest','isFraud'], chunksize=200000):
        chunk['row_id'] = chunk.index + 1
        for role, col in ROLES.items():
            ids = {account for r, account in keys if r == role}
            part = chunk[chunk[col].isin(ids)].copy()
            if part.empty:
                continue
            with threadpool_limits(limits=1):
                part['model_score'] = model.predict_proba(features(part))[:, 1]
            part['review_flag'] = part.model_score >= report['threshold']
            part['role'] = role; part['account'] = part[col]
            part['counterparty'] = part['nameDest' if role == 'origin' else 'nameOrig']
            history.append(part.drop(columns=['nameOrig','nameDest']))
    all_history = pd.concat(history, ignore_index=True).sort_values(['role','account','step','row_id'])
    accounts = []
    for (role, account), group in all_history.groupby(['role','account'], sort=True):
        labelled = group[group.isFraud == 1]
        reviews = group[group.review_flag]
        # Ship every labelled/flagged event and first/last context, not an unbounded normal history.
        show = group[(group.isFraud == 1) | group.review_flag | group.row_id.isin([group.row_id.iloc[0],group.row_id.iloc[-1]])].copy()
        show['prior_fraud_at_earlier_hour'] = prior_fraud_steps(group).loc[show.index]
        show['model_score'] = show.model_score.round(8)
        accounts.append({'role':role, 'account':account, 'transactions':len(group),
                         'fraud_count':len(labelled), 'review_count':len(reviews),
                         'first_hour':int(group.step.min()), 'last_hour':int(group.step.max()),
                         'first_fraud_hour':int(labelled.step.min()) if len(labelled) else None,
                         'last_fraud_hour':int(labelled.step.max()) if len(labelled) else None,
                         'max_score':round(float(group.model_score.max()),8),
                         'history_rows_shown':len(show),
                         'events':show[['row_id','step','type','amount','counterparty','isFraud','model_score','review_flag','prior_fraud_at_earlier_hour']].to_dict('records')})
    totals = {}
    for role in ROLES:
        a = aggregates[aggregates.role == role]
        totals[role] = {'fraud_accounts':int((a.fraud_count > 0).sum()),
                        'repeat_fraud_accounts':int((a.fraud_count >= 2).sum()),
                        'repeat_review_accounts':int((a.review_count >= 2).sum()),
                        'repeat_fraud_across_hours':sum(1 for x in accounts if x['role']==role and x['fraud_count']>=2 and x['last_fraud_hour']>x['first_fraud_hour'])}
    artifact = {'schema_version':1, 'dataset_sha256':EXPECTED_SHA256,
                'model_sha256':report['model_sha256'], 'threshold':report['threshold'],
                'source':report['dataset']['source'], 'rows':rows, 'fraud_rows':fraud,
                'review_rows':flagged, 'hours':[1,743], 'totals':totals,
                'scope':'Full synthetic run, including training and validation rows. Retrospective description, not a new performance test or prediction of repeat victimization.',
                'selection':'All role/account pairs with >=2 fraud labels OR >=2 model review flags. Histories contain every labelled/flagged event plus first/last transaction context; other normal rows omitted.',
                'accounts':accounts}
    out = ROOT/'account_analysis.json'
    out.write_text(json.dumps(artifact, separators=(',',':'), allow_nan=False)+'\n')
    manifest = {'artifact_sha256':sha256(out), 'dataset_sha256':EXPECTED_SHA256,
                'model_sha256':report['model_sha256'], 'builder_sha256':sha256(__file__)}
    (ROOT/'account_analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in artifact.items() if k!='accounts'},indent=2))
    print('Accounts:',len(accounts),'Artifact bytes:',out.stat().st_size)

if __name__ == '__main__':
    main()
