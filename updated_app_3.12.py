import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Random Forest Model Deployment")

uploaded_file = st.file_uploader("Upload CSV file", type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    st.subheader("Basic Info")
    st.write(df.info())
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isna().sum())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    if 'isFraud' in df.columns:
        df_fraud = df[df['isFraud'] == 1]
        df_non_fraud = df[df['isFraud'] == 0]

        st.subheader("Fraud Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='isFraud', data=df, ax=ax1)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.hist(df_fraud['amount'], bins=100, alpha=0.5, label='Fraud', color='red')
        ax2.hist(df_non_fraud['amount'], bins=100, alpha=0.5, label='Non-Fraud', color='green')
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(0, 10000)
        ax2.legend()
        ax2.set_title("Transaction Amount Distribution")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.pie([len(df_fraud), len(df_non_fraud)], labels=['Fraud', 'Non-Fraud'], autopct='%1.2f%%',
                colors=['red', 'green'], startangle=90)
        st.pyplot(fig3)

    drop_cols = ['zipCodeOri', 'zipMerchant', 'nameOrig', 'nameDest']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    if 'type' in df.columns:
        df = pd.get_dummies(df, columns=['type'], drop_first=True)

    if 'isFraud' in df.columns:
        X = df.drop(columns=['isFraud'])
        y = df['isFraud']

        num_cols = X.select_dtypes(include=np.number).columns
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, rf_clf.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        fig4, ax4 = plt.subplots()
        ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax4.set_title("ROC Curve")
        ax4.legend()
        st.pyplot(fig4)

