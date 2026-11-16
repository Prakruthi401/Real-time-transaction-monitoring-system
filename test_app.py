import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from fraud_core import features, validate

class FraudDemoTests(unittest.TestCase):
    def test_invalid_rows(self):
        rows = [{'step':0,'type':'TRANSFER','amount':1}, {'step':2,'type':'unknown','amount':1},
                {'step':2,'type':'PAYMENT','amount':float('nan')}, {'step':2.5,'type':'DEBIT','amount':1},
                {'step':2,'type':'PAYMENT','amount':float('inf')}, {'step':2,'type':'PAYMENT','amount':-1}]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate(pd.DataFrame([row]))
        with self.assertRaises(ValueError):
            validate(pd.DataFrame([{'amount':1}]))

    def test_label_and_extra_data_excluded(self):
        data = pd.read_csv('demo_transactions.csv')
        clean = data[['step','type','amount']]
        np.testing.assert_array_equal(features(data), features(clean))
        model = joblib.load('model.joblib')
        pred = model.predict_proba(features(data))[:,1]
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())
        metrics = json.loads(Path('metrics.json').read_text())
        self.assertEqual(sum(metrics['test_metrics']['confusion_matrix_tn_fp_fn_tp']), metrics['test']['rows'])

    def test_streamlit_forms_and_replay(self):
        app = AppTest.from_file('updated_app_3.12.py', default_timeout=30).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == 'Score transaction').click().run()
        self.assertFalse(app.exception)
        self.assertIn('single_result', app.session_state)
        next(b for b in app.button if b.label == 'Score next 10').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.replay_count,10)
        next(b for b in app.button if b.label == 'Reset replay').click().run()
        self.assertEqual(app.session_state.replay_count,0)

if __name__ == '__main__':
    unittest.main()
