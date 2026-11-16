"""Read-only UI for precomputed, synthetic account histories. No model features here."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

@st.cache_data
def load_analysis():
    manifest = json.loads((ROOT/'account_analysis_manifest.json').read_text())
    raw = (ROOT/'account_analysis.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['artifact_sha256']:
        raise ValueError('Account analysis integrity check failed.')
    data = json.loads(raw)
    report = json.loads((ROOT/'metrics.json').read_text())
    if data['model_sha256'] != report['model_sha256'] or data['dataset_sha256'] != report['dataset']['sha256'] or data['threshold'] != report['threshold']:
        raise ValueError('Account analysis does not match the shipped experiment.')
    return data

def render_account_analysis():
    data = load_analysis()
    st.subheader('Account patterns, with a history behind each flag')
    st.write('Look back at repeated fraud labels and repeated model-review flags in the synthetic PaySim dataset. Account IDs connect the history here; they never enter the scoring model.')
    st.warning('Retrospective synthetic analysis, not a prediction about a real person. A previous fraud label is a reason to look again, not evidence that an account will be defrauded again. No future-risk probability is estimated.')
    a,b,c = st.columns(3)
    a.metric('Repeat fraud origins', data['totals']['origin']['repeat_fraud_accounts'])
    b.metric('Repeat fraud receivers', data['totals']['destination']['repeat_fraud_accounts'])
    c.metric('Repeat high-score receivers', data['totals']['destination']['repeat_review_accounts'])
    st.caption(f"Full run: {data['rows']:,} transactions, {data['fraud_rows']:,} fraud labels, hours 1-743. Counts are distinct accounts within each role. Categories overlap. Repeat means at least two transactions; review means score >= {data['threshold']:.4f}.")
    st.info('No repeat-victim claim: all 8,213 fraud-labelled origin IDs occur only once in the fraud rows. PaySim records transaction origins and receivers, not verified victims or mule identities. A repeat receiver is an investigation pattern, not proof of wrongdoing.')
    category = st.radio('Pattern to explore', ['Repeated fraud labels', 'Repeated model-review flags'], horizontal=True)
    role_label = st.selectbox('Account role', ['Receiver (nameDest)', 'Origin (nameOrig)'])
    role = 'destination' if role_label.startswith('Receiver') else 'origin'
    count_key = 'fraud_count' if category == 'Repeated fraud labels' else 'review_count'
    chosen = [a for a in data['accounts'] if a['role']==role and a[count_key]>=2]
    if not chosen:
        st.info('No matching repeat accounts in this dataset. Zero is a result, not a missing feature.')
    else:
        chosen = sorted(chosen, key=lambda a:(-a[count_key],a['account']))
        summary = pd.DataFrame([{k:a[k] for k in ['account','transactions','fraud_count','review_count','first_hour','last_hour','first_fraud_hour','last_fraud_hour','max_score']} for a in chosen])
        st.markdown(f'#### {len(chosen):,} matching accounts')
        st.dataframe(summary, hide_index=True, width='stretch')
        account = st.selectbox('Inspect a synthetic account', [a['account'] for a in chosen])
        detail = next(a for a in chosen if a['account']==account)
        st.markdown(f'#### History: {account}')
        st.caption(f"{detail['transactions']:,} transactions in this role; {detail['fraud_count']} fraud labels; {detail['review_count']} model-review flags. Showing {detail['history_rows_shown']} rows: every fraud-labelled or model-flagged event plus first/last transaction context. Other normal transactions are omitted.")
        events = pd.DataFrame(detail['events']).rename(columns={'step':'simulation_hour','isFraud':'simulation_fraud_label'})
        st.dataframe(events, hide_index=True, width='stretch', column_config={'model_score':st.column_config.NumberColumn(format='%.4f')})
        if detail['fraud_count']>=2:
            st.write(f"First fraud-labelled receipt at hour {detail['first_fraud_hour']}; another at hour {detail['last_fraud_hour']}. Later events can be marked 'prior fraud at an earlier hour' using only the earlier label. This is a replay of known labels, not an advance warning.")
        st.caption('The prior-fraud marker requires a strictly earlier simulation hour. Events in the same hour have no known ordering. A simulation hour is not a calendar date; fraud labels would not necessarily be available immediately in a real system.')
        st.download_button('Download matching account summary', summary.to_csv(index=False), 'synthetic_account_patterns.csv','text/csv')
    with st.expander('Scope, method and limits'):
        st.write(data['scope'])
        st.write('Model-review flags use the existing model and threshold. They can be false positives; repeated flags do not confirm fraud. Counts include the training and validation windows and are not an independent model evaluation.')
        st.write('Origins and receivers are counted separately. No identity resolution across roles, no linked real accounts, no bank data, no estimated chance of recurrence. This view does not alter scores, train the model, or send alerts.')
        st.write('The full 493 MB CSV is processed offline in chunks. The app loads a checksum-verified 502 KB summary, not the source dataset. All repeated-pattern accounts are included; normal history is reduced as labelled above.')
        st.link_button('Dataset definitions and simulation limits', data['source'])
        st.download_button('Download analysis provenance', json.dumps({k:v for k,v in data.items() if k!='accounts'},indent=2), 'account_analysis_provenance.json','application/json')
