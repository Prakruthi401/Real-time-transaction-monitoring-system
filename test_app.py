import hashlib
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

class AccountAnalysisTests(unittest.TestCase):
    def test_integrity_and_counts(self):
        raw = Path('account_analysis.json').read_bytes()
        data = json.loads(raw)
        manifest = json.loads(Path('account_analysis_manifest.json').read_text())
        report = json.loads(Path('metrics.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['artifact_sha256'])
        self.assertEqual(hashlib.sha256(Path('build_account_analysis.py').read_bytes()).hexdigest(), manifest['builder_sha256'])
        self.assertEqual(data['model_sha256'], report['model_sha256'])
        self.assertEqual(data['dataset_sha256'], report['dataset']['sha256'])
        self.assertEqual(data['threshold'], report['threshold'])
        self.assertEqual(data['rows'], 6362620)
        self.assertEqual(data['totals']['origin']['repeat_fraud_accounts'], 0)
        self.assertEqual(data['totals']['destination']['repeat_fraud_accounts'], 44)
        self.assertEqual(data['totals']['destination']['repeat_review_accounts'], 391)
        self.assertEqual(len(data['accounts']), 428)
        self.assertLess(len(raw), 600000)
        for a in data['accounts']:
            events = pd.DataFrame(a['events'])
            self.assertTrue(a['fraud_count'] >= 2 or a['review_count'] >= 2)
            self.assertEqual(int(events.isFraud.sum()), a['fraud_count'])
            self.assertEqual(int(events.review_flag.sum()), a['review_count'])
            self.assertGreaterEqual(a['transactions'], len(events))
            for row in a['events']:
                expected = a['first_fraud_hour'] is not None and row['step'] > a['first_fraud_hour']
                self.assertEqual(row['prior_fraud_at_earlier_hour'], expected)

    def test_strictly_past_not_same_hour(self):
        from build_account_analysis import prior_fraud_steps
        events = pd.DataFrame({'step':[1,2,2,3], 'isFraud':[0,1,0,1]})
        self.assertEqual(prior_fraud_steps(events).tolist(), [False,False,False,True])
        events.isFraud = 0
        self.assertEqual(prior_fraud_steps(events).tolist(), [False]*4)

    def test_account_controls(self):
        app = AppTest.from_file('updated_app_3.12.py', default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn('Account patterns', [t.label for t in app.tabs])
        account = next(s for s in app.selectbox if s.label == 'Inspect a synthetic account')
        account.set_value(account.options[-1]).run()
        self.assertFalse(app.exception)
        next(r for r in app.radio if r.label == 'Pattern to explore').set_value('Repeated model-review flags').run()
        self.assertFalse(app.exception)
        next(s for s in app.selectbox if s.label == 'Account role').set_value('Origin (nameOrig)').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('No matching repeat accounts' in i.value for i in app.info))

if __name__ == '__main__':
    unittest.main()
