
import streamlit as st
import pandas as pd
import numpy as np

st.title("Random Forest Model Deployment")

# Load Dataset
st.write(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv")
uploaded_file = st.file_uploader("r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv"", type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)  # Correct indentation here
    st.write(data.head())             # Correct indentation here

# Add your additional code below if required


# Insert the merged code here (adjusting for compatibility)
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier




# read the data and show first 5 rows
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv")
data.head(5)

data.info()

df_dd=data.copy()

#Counting no. of rows and columns
print("Number of rows: {}".format(df_dd.shape[0]))
print("Number of columns: {}".format(df_dd.shape[1]))

#data types of each columns
df_dd.dtypes

#Checking number of values 
df_dd.isna().sum()

# Calculate the summary statistics using describe()
summary = df_dd.describe()

# Access specific statistics from the summary_stats DataFrame
count_values = summary.loc['count']
mean_values = summary.loc['mean']
std_values = summary.loc['std']
min_values = summary.loc['min']
median_values = summary.loc['50%']
max_values = summary.loc['max']

# Calculate the range separately
range_values = max_values - min_values

# Print the results
print("Count:")
print(count_values)

print("\nMean:")
print(mean_values)

print("\nStandard Deviation:")
print(std_values)

print("\nMinimum:")
print(min_values)

print("\nMedian ")
print(median_values)


print("\nMaximum:")
print(max_values)

print("\nRange:")
print(range_values)


import seaborn as sns
import matplotlib.pyplot as plt

# Check if 'isFraud' exists in the dataframe
if 'isFraud' in df_dd.columns:
    # Create separate DataFrames for fraud and non-fraud transactions
    df_fraud = df_dd[df_dd['isFraud'] == 1]
    df_non_fraud = df_dd[df_dd['isFraud'] == 0]

    # Plot fraud distribution
    sns.countplot(x='isFraud', data=df_dd)
    plt.title("Count of Fraudulent Payments")
    plt.show()

    # Print fraud counts
    print("Number of normal transactions:", len(df_non_fraud))
    print("Number of fraudulent transactions:", len(df_fraud))

else:
    print("⚠️ Error: Column 'isFraud' not found in the dataset!")
    print("Available columns:", df_dd.columns)


plt.figure(figsize=(10, 6))
plt.hist(df_fraud['amount'], alpha=0.5, label='Fraud', bins=100, color='red')
plt.hist(df_non_fraud['amount'], alpha=0.5, label='Non-Fraud', bins=100, color='green')

plt.ylim(0, 10000)
plt.xlim(0, 1000)
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.legend()
plt.title("Transaction Amount Distribution: Fraud vs Non-Fraud")
plt.show()


import matplotlib.pyplot as plt

if 'isFraud' in df_dd.columns:
    fcount = df_dd['isFraud'].sum()
    nfcount = len(df_dd) - fcount

    labels = ['Fraud', 'Non-Fraud']
    sizes = [fcount, nfcount]
    colors = ['red', 'green']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title('Fraudulent vs. Non-Fraudulent Transactions')
    plt.show()
else:
    print("⚠️ Error: Column 'isFraud' not found in the dataset!")
    print("Available columns:", df_dd.columns)


print("Available columns:", df_dd.columns)

# Check if the columns exist before proceeding
if 'zipCodeOri' in df_dd.columns and 'zipMerchant' in df_dd.columns:
    print("Unique zipCodeOri values:", df_dd['zipCodeOri'].nunique())
    print("Unique zipMerchant values:", df_dd['zipMerchant'].nunique())

    # Drop columns
    data_reduced = df_dd.drop(['zipCodeOri', 'zipMerchant'], axis=1)
else:
    print("Columns 'zipCodeOri' and/or 'zipMerchant' not found in the dataset.")


import pandas as pd

# Load the dataset (ensure the file path is correct)
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv")

# Create a copy of the dataset to work with
data_reduced = df.copy()

# Identify categorical columns
col_categorical = data_reduced.select_dtypes(include=['object']).columns

if not col_categorical.empty:
    for col in col_categorical:
        data_reduced[col] = data_reduced[col].astype('category')

    # Convert categorical values to numeric codes
    data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)

data_reduced.head()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load dataset (Ensure this path is correct)
df_path =r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv"
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv")

# Verify column names
print(df.columns)

# Rename 'isFraud' to 'fraud' for consistency (if needed)
df.rename(columns={'isFraud': 'fraud'}, inplace=True)

# Check if 'fraud' column exists
if 'fraud' not in df.columns:
    raise ValueError("Column 'fraud' is missing from dataset!")

# Create a reduced dataset with necessary features
data_reduced = df[['step', 'amount', 'fraud']]  # Add more relevant features if needed

# Split data
X = data_reduced.drop('fraud', axis=1)
y = data_reduced['fraud']
numerical_columns = ['step', 'amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Train model
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced", verbose=0)
rf_clf.fit(X_train, y_train)

# Predictions
y_pred = rf_clf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_clf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df['amount'], bins=50, kde=True, color='blue')
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.xlim(0, df['amount'].quantile(0.95))  # Focus on 95% of transactions
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv")

# Drop unnecessary columns
X = df.drop(columns=['isFraud', 'nameOrig', 'nameDest'])  # Remove ID-like columns
y = df['isFraud']

# Convert categorical column(s) to numerical
X = pd.get_dummies(X, columns=['type'], drop_first=True)  # One-hot encoding

# Select numerical columns
numerical_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Train Random Forest Model
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=6, class_weight="balanced", n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)

# Predictions
y_pred = rf_clf.predict(X_test)

# Evaluation
print("Classification Report: \n", classification_report(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_clf.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


plt.figure(figsize=(8,5))
fraud_rates = df.groupby('type')['isFraud'].mean()
fraud_rates.plot(kind='bar', color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title('Fraud Rate by Transaction Type')
plt.xlabel('Transaction Type')
plt.ylabel('Fraud Rate')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df['oldbalanceOrg'], bins=50, kde=True, color='purple')
plt.title('Distribution of Old Balance Before Transaction')
plt.xlabel('Old Balance')
plt.ylabel('Frequency')
plt.xlim(0, df['oldbalanceOrg'].quantile(0.95))
plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv")

# Display basic dataset info
print(df.info())
print(df.head())

# Rename columns for clarity if needed
df.columns = df.columns.str.lower()

# Group by `nameOrig` to analyze transactions per person
transaction_summary = df.groupby('nameorig')['amount'].sum().reset_index()

# Define percentiles for top 1%, 10%, 20%, 50%
percentiles = [0.99, 0.90, 0.80, 0.50]
quantiles = transaction_summary['amount'].quantile(percentiles)
print("Transaction Percentiles:\n", quantiles)

# Analyzing balances
balance_summary = df.groupby('nameorig')['oldbalanceorg'].mean().reset_index()
balance_quantiles = balance_summary['oldbalanceorg'].quantile(percentiles)
print("\nBalance Percentiles:\n", balance_quantiles)

# Fraud analysis based on balance
df['high_balance'] = df['oldbalanceorg'] > balance_quantiles[0.90]  # Top 10% balances
df['moderate_balance'] = (df['oldbalanceorg'] <= balance_quantiles[0.90]) & (df['oldbalanceorg'] > balance_quantiles[0.50])
df['low_balance'] = df['oldbalanceorg'] <= balance_quantiles[0.50]

# Fraud Probability Calculation
fraud_cases = df[df['isfraud'] == 1]
total_cases = len(df)

prob_high = len(fraud_cases[fraud_cases['high_balance']]) / total_cases
prob_moderate = len(fraud_cases[fraud_cases['moderate_balance']]) / total_cases
prob_low = len(fraud_cases[fraud_cases['low_balance']]) / total_cases

print("\nFraud Probability by Balance Group:")
print(f"High Balance: {prob_high:.4f}")
print(f"Moderate Balance: {prob_moderate:.4f}")
print(f"Low Balance: {prob_low:.4f}")

# 🎯 Fraud probability bar chart (ZOOMED)
fraud_probs = [prob_high, prob_moderate, prob_low]
labels = ["High Balance", "Moderate Balance", "Low Balance"]

plt.figure(figsize=(6, 6))
sns.barplot(x=labels, y=fraud_probs, palette="coolwarm")
plt.ylabel('Probability of Fraud', fontsize=14)
plt.xlabel('Balance Level', fontsize=14)
plt.title('Fraud Probability Based on Balance Level (Zoomed-in)', fontsize=16)
plt.ylim(0, max(fraud_probs) + 0.05)  # Zoom in
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# 🎯 Box Plot: Transaction Amount vs Fraud Status
plt.figure(figsize=(6, 6))
sns.boxplot(x=df['isfraud'], y=df['amount'], palette="coolwarm")
plt.yscale('log')  # Log scale to handle large values
plt.xlabel("Fraud Status (0 = No Fraud, 1 = Fraud)", fontsize=14)
plt.ylabel("Transaction Amount (Log Scale)", fontsize=14)
plt.title("Box Plot of Transaction Amounts Based on Fraud Status", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


import pandas as pd

# Load the existing dataset
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv")

# Function to manually add a new transaction
def add_manual_transaction():
    print("\n🔹 Enter new transaction details manually 🔹")
    
    step = int(input("Step (time step): "))
    transaction_type = input("Transaction Type (TRANSFER / CASH_OUT / etc.): ").upper()
    amount = float(input("Amount: "))
    nameOrig = input("Sender ID: ")
    oldbalanceOrg = float(input("Sender's Balance Before Transaction: "))
    newbalanceOrig = float(input("Sender's Balance After Transaction: "))
    nameDest = input("Receiver ID: ")
    oldbalanceDest = float(input("Receiver's Balance Before Transaction: "))
    newbalanceDest = float(input("Receiver's Balance After Transaction: "))
    isFraud = int(input("Is Fraud? (1 = Yes, 0 = No): "))
    isFlaggedFraud = int(input("Is Flagged as Fraud? (1 = Yes, 0 = No): "))

    # Create a new row
    new_transaction = pd.DataFrame([{
        'step': step,
        'type': transaction_type,
        'amount': amount,
        'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'nameDest': nameDest,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFraud': isFraud,
        'isFlaggedFraud': isFlaggedFraud
    }])

    return new_transaction

# Ask the user if they want to add manual data
while True:
    choice = input("\nDo you want to manually add a new transaction? (yes/no): ").strip().lower()
    if choice == 'yes':
        new_data = add_manual_transaction()
        df = pd.concat([df, new_data], ignore_index=True)
        print("\n✅ New transaction added successfully!")
        print(new_data)
    else:
        break

# Save the updated dataset
df.to_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv", index=False)
print("\n📁 Updated dataset saved as 'updated_transactions.csv'!")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the existing dataset
data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv")

# Function to safely get user input
def get_valid_input(prompt, dtype=float, condition=lambda x: True, error_message="Invalid input!"):
    while True:
        try:
            value = input(prompt).strip()
            if dtype == int:
                value = int(value)
            elif dtype == float:
                value = float(value)
            else:
                value = value.upper()
            
            if condition(value):
                return value
            else:
                print(f"⚠️ {error_message}")
        except ValueError:
            print(f"⚠️ {error_message}")

# Function to manually add a new transaction
def add_manual_transaction():
    print("\n🔹 Enter new transaction details manually 🔹")
    
    step = get_valid_input("Step (time step): ", int, lambda x: x >= 0, "Step must be a non-negative integer.")
    transaction_type = get_valid_input("Transaction Type (TRANSFER / CASH_OUT / etc.): ", str)
    amount = get_valid_input("Amount: ", float, lambda x: x > 0, "Amount must be positive.")
    nameOrig = input("Sender ID: ").strip()
    oldbalanceOrg = get_valid_input("Sender's Balance Before Transaction: ", float, lambda x: x >= 0, "Balance must be non-negative.")
    newbalanceOrig = get_valid_input("Sender's Balance After Transaction: ", float, lambda x: x >= 0, "Balance must be non-negative.")
    nameDest = input("Receiver ID: ").strip()
    oldbalanceDest = get_valid_input("Receiver's Balance Before Transaction: ", float, lambda x: x >= 0, "Balance must be non-negative.")
    newbalanceDest = get_valid_input("Receiver's Balance After Transaction: ", float, lambda x: x >= 0, "Balance must be non-negative.")
    isFraud = get_valid_input("Is Fraud? (1 = Yes, 0 = No): ", int, lambda x: x in [0, 1], "Enter 0 or 1 only.")
    isFlaggedFraud = get_valid_input("Is Flagged as Fraud? (1 = Yes, 0 = No): ", int, lambda x: x in [0, 1], "Enter 0 or 1 only.")

    # Create a new row
    new_transaction = pd.DataFrame([{
        'step': step,
        'type': transaction_type,
        'amount': amount,
        'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'nameDest': nameDest,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFraud': isFraud,
        'isFlaggedFraud': isFlaggedFraud
    }])

    return new_transaction

# Ask the user if they want to add manual data
while True:
    choice = input("\nDo you want to manually add a new transaction? (yes/no): ").strip().lower()
    if choice == 'yes':
        new_data = add_manual_transaction()
        df = pd.concat([df, new_data], ignore_index=True)
        print("\n✅ New transaction added successfully!")
        print(new_data)
    elif choice == 'no':
        break
    else:
        print("⚠️ Please enter 'yes' or 'no'.")

# Save the updated dataset
df.to_csv(r"C:\Users\admin\OneDrive\Desktop\project\PS_20174392719_1491204439457_log.csv", index=False)
print("\n📁 Updated dataset saved as 'updated_transactions.csv'! ✅")
print("\n📊 Dataset Summary:")
print(df.tail(5))  # Show last 5 transactions

# ---------------------------- MACHINE LEARNING MODEL ----------------------------

# Convert categorical 'type' column to numerical
df['type'] = df['type'].astype('category').cat.codes

# Select features & target variable
features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
X = df[features]
y = df['isFraud']

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy and other evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Print results
print("\n📊 Model Performance:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")


st.write("### Model Training and Evaluation Complete")
st.write("Run the app in Streamlit to visualize the outputs.")
