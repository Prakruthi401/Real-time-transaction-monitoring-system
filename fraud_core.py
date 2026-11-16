"""Shared pre-transaction features and strict input validation. No account IDs or balances."""
import numpy as np
import pandas as pd

TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
COLUMNS = ['step', 'type', 'amount']

def validate(frame):
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    data = frame[COLUMNS].copy()
    data['type'] = data['type'].astype(str).str.upper().str.strip()
    if not data['type'].isin(TYPES).all():
        raise ValueError('Type must be CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER.')
    for col in ['step', 'amount']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if not np.isfinite(data[col]).all():
            raise ValueError(col + ' must contain only finite numbers.')
    if ((data.step < 1) | (data.step > 743) | (data.step % 1 != 0)).any():
        raise ValueError('Step must be a whole simulation hour from 1 to 743.')
    if ((data.amount < 0) | (data.amount > 92445516.64)).any():
        raise ValueError('Amount must be between 0 and 92,445,516.64 simulation units.')
    return data

def features(frame):
    data = validate(frame)
    hour = (data.step.to_numpy() - 1) % 24
    return np.column_stack([np.log1p(data.amount.to_numpy()), np.sin(2*np.pi*hour/24),
                            np.cos(2*np.pi*hour/24)] +
                           [(data.type == t).to_numpy(dtype=float) for t in TYPES])
