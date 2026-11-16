from pathlib import Path
import hashlib
import json
from io import BytesIO
import joblib
import pandas as pd
import streamlit as st
from threadpoolctl import threadpool_limits
from fraud_core import COLUMNS, TYPES, features, validate

ROOT = Path(__file__).parent
st.set_page_config(page_title='Transaction Watch | Fraud scoring lab', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp {background:#f5f7fa;color:#172c40}
.block-container{max-width:1180px;padding-top:2.5rem}
h1,h2,h3{color:#142e46;font-family: sans-serif;letter-spacing:-.04em}
h1{font-size:2.8rem!important} [data-testid="stMetric"]{background:white;border:1px solid #dce4ec;border-radius:8px;padding:16px}
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primaryFormSubmit"]{background:#174c68;border-color:#174c68}
button[role="tab"][aria-selected="true"]{color:#174c68!important}
[data-baseweb="tab-highlight"]{background:#174c68!important}
[data-testid="stFormSubmitButton"] button{background:#174c68!important;border-color:#174c68!important;color:white!important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:#516478!important}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    report = json.loads((ROOT/'metrics.json').read_text())
    model_path = ROOT/'model.joblib'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != report['model_sha256']:
        raise ValueError('Model integrity check failed.')
    return joblib.load(model_path), report

model, report = load_model()
threshold = report['threshold']

def score(data):
    clean = validate(data)
    result = clean.reset_index(drop=True).copy()
    with threadpool_limits(limits=1):
        result['model_score'] = model.predict_proba(features(clean))[:, 1]
    result['review_flag'] = result.model_score >= threshold
    return result

def csv_bytes(data):
    return data.to_csv(index=False).encode('utf-8')

st.caption('TRANSACTION WATCH / A FRAUD SCORING LAB')
st.title('A second look at suspicious payments.')
st.write('Score a transaction, work through a review queue, or replay a small batch. A trained model runs each score, with its limits in plain sight.')
st.info('Portfolio demo, not a banking service. Uses synthetic PaySim data. No bank connection, payment blocking, live alerts or production fraud guarantees. Please use sample or invented data only.')
tabs = st.tabs(['Score a transaction', 'Batch review', 'Replay demo', 'Model & evidence'])

with tabs[0]:
    left, right = st.columns([1.35, 1], gap='large')
    with left:
        st.subheader('Transaction details')
        st.caption('Amount is in PaySim simulation currency units, not INR or USD. Step is the simulation hour, not a timestamp.')
        with st.form('transaction'):
            kind = st.selectbox('Transaction type', TYPES, index=4)
            amount = st.number_input('Amount (simulation units)', min_value=0.0, max_value=92445516.64, value=250000.0, step=1000.0)
            step = st.number_input('Simulation hour (1-743)', min_value=1, max_value=743, value=400)
            submitted = st.form_submit_button('Score transaction', type='primary', width='stretch')
        if submitted:
            row = score(pd.DataFrame([{'type': kind, 'amount': amount, 'step': step}])).iloc[0]
            st.session_state['single_result'] = row.to_dict()
        if 'single_result' in st.session_state:
            row = st.session_state['single_result']
            st.markdown('#### Sent to review' if row['review_flag'] else '#### Below the review threshold')
            st.metric('Model score (not a calibrated probability)', f"{row['model_score']:.4f}")
            st.caption(f"Scored input: {row['type']}, {row['amount']:,.2f} units, hour {int(row['step'])}. Review threshold: {threshold:.4f}.")
            st.warning('A flag asks for human review. A low score does not prove a payment is safe.')
    with right:
        st.subheader('What the model sees')
        st.write('Transaction type, amount and hour of day. Nothing else.')
        st.write('Balance columns are deliberately excluded: PaySim warns they can reveal the outcome of cancelled fraud. Account identifiers, labels and the existing fraud flag are excluded too.')
        st.markdown('**No made-up explanations.** The app does not invent per-feature percentages or treat a score as proof of fraud.')
        st.divider()
        st.caption('The model is already trained and shipped with the app. Your input is scored, never used to retrain it. Inputs are held in this session; there is no application database or bank integration. Hosting-provider logs and infrastructure remain outside this app\'s control.')

with tabs[1]:
    st.subheader('A review queue, not an automatic verdict')
    st.write('Upload up to 10,000 rows / 2 MB. Required columns: step, type, amount. Extra columns are ignored and not included in the output. Do not upload real customer or account data.')
    template = pd.DataFrame([{'step':400,'type':'TRANSFER','amount':250000.0}, {'step':400,'type':'PAYMENT','amount':1200.0}])
    st.download_button('Download CSV template', csv_bytes(template), 'transaction_template.csv', 'text/csv')
    file = st.file_uploader('Sample transactions (.csv)', type=['csv'])
    if file is not None:
        try:
            if file.size > 2*1024*1024:
                raise ValueError('File is over 2 MB. Use a smaller sample.')
            raw = pd.read_csv(BytesIO(file.getvalue()), nrows=10001)
            if len(raw) < 1 or len(raw) > 10000:
                raise ValueError('Use between 1 and 10,000 rows.')
            result = score(raw)
            a,b,c = st.columns(3)
            a.metric('Transactions', len(result)); b.metric('Sent to review', int(result.review_flag.sum()))
            c.metric('Review threshold', f'{threshold:.4f}')
            st.dataframe(result.sort_values('model_score', ascending=False), hide_index=True, width='stretch')
            st.download_button('Download scored CSV', csv_bytes(result), 'scored_transactions.csv', 'text/csv')
            st.caption('Rows are sorted by score on screen. The downloaded CSV preserves input order. Review flags are model outputs, not verified fraud labels.')
        except (ValueError, pd.errors.ParserError, UnicodeError, KeyError) as error:
            st.error(str(error))

with tabs[2]:
    st.subheader('Replay a selected demonstration batch')
    st.write('100 held-out synthetic transactions: deliberately selected as 80 normal and 20 fraud examples. This is not natural fraud prevalence, a live feed, or a benchmark.')
    demo = pd.read_csv(ROOT/'demo_transactions.csv')
    if 'replay_count' not in st.session_state:
        st.session_state.replay_count = 0
    a,b = st.columns([1,4])
    if a.button('Score next 10', type='primary'):
        st.session_state.replay_count = min(100, st.session_state.replay_count + 10)
    if b.button('Reset replay'):
        st.session_state.replay_count = 0
    n = st.session_state.replay_count
    st.progress(n/100, text=f'{n} / 100 transactions processed')
    if n:
        result = score(demo.iloc[:n])
        result['simulation_label'] = demo.iloc[:n].isFraud.to_numpy()
        a,b = st.columns(2)
        a.metric('Review flags', int(result.review_flag.sum()))
        b.metric('Known fraud in these rows', int(result.simulation_label.sum()))
        st.dataframe(result, hide_index=True, width='stretch')
        st.caption('Simulation labels are shown only for learning. They are not inputs to the model.')

with tabs[3]:
    st.subheader('Measured, with the misses included')
    m = report['test_metrics']
    a,b,c,d = st.columns(4)
    a.metric('Average precision', f"{m['average_precision']:.3f}")
    b.metric('Review precision', f"{m['precision']:.1%}")
    c.metric('Fraud recall', f"{m['recall']:.1%}")
    d.metric('ROC-AUC', f"{m['roc_auc']:.3f}")
    st.caption(f"Untouched later-hour test set: {report['test']['rows']:,} rows, {report['test']['fraud']:,} fraud. No-skill average precision: {m['no_skill_average_precision']:.4f}. No balanced resampling of the test set.")
    tn,fp,fn,tp = m['confusion_matrix_tn_fp_fn_tp']
    st.warning(f'At the shipped threshold, the model missed {fn:,} of {fn+tp:,} fraud transactions ({fn/(fn+tp):.1%}). It flagged {fp:,} normal transactions. It is not suitable for unattended financial decisions.')
    st.table(pd.DataFrame([[tn,fp],[fn,tp]], index=['Actually normal','Actually fraud'], columns=['Below threshold','Review flag']))
    st.markdown('''#### How this experiment works
- Histogram gradient-boosted trees, trained offline on a seeded sample of 1,000,000 rows from hours 1-281.
- Threshold selected only on hours 282-355: maximize F2 while requiring at least 10% validation precision.
- Final evaluation on all hours 356-743. No overlap of simulation hours between the splits.
- No SMOTE, balance fields, account IDs, target label, existing fraud flag or raw simulation-hour trend as model inputs.
- Fraud frequency changes from the training window to the test window. This is one simulator run, not evidence of performance on real payments or independent banks.
- No calibrated probability claim. Scores depend on synthetic patterns. An unseen fraud pattern may get a very low score.

#### Reproduce it
The repository includes the training script, exact dataset checksum, package versions, model checksum, split counts and confusion matrix. The original classroom project credits POOJA J and PRAKRUTHI GOWDA D; both credits are preserved.
''')
    st.link_button('Open code & reproduction steps', 'https://github.com/Prakruthi401/Real-time-transaction-monitoring-system')
    st.link_button('Read the original PaySim dataset notes', report['dataset']['source'])
    st.download_button('Download full evaluation JSON', json.dumps(report, indent=2), 'metrics.json', 'application/json')

st.divider()
st.caption('Transaction Watch · Original project: POOJA J and PRAKRUTHI GOWDA D · PaySim: Lopez-Rojas, Elmir & Axelsson (2016) · Educational scoring only')
