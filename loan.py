# -- coding: utf-8 --
"""Merged Loan Prediction System
Combining Colab pre-processing and Streamlit application for loan prediction.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from termcolor import colored

# Load dataset
file_path = r'D:\loanModel\train_u6lujuX_CVtuZ9i.csv'
loan_data = pd.read_csv(file_path)

# Data Cleaning and Pre-Processing
loan_data.drop(['Loan_ID'], axis=1, inplace=True)
print(loan_data.info())

# Handle missing values
loan_data["Credit_History"] = loan_data["Credit_History"].fillna(loan_data["Credit_History"].median())
loan_data["LoanAmount"] = loan_data["LoanAmount"].fillna(loan_data["LoanAmount"].median())
loan_data["Loan_Amount_Term"] = loan_data["Loan_Amount_Term"].fillna(loan_data["Loan_Amount_Term"].mode()[0])
loan_data["Gender"] = loan_data["Gender"].fillna(loan_data["Gender"].mode()[0])
loan_data["Married"] = loan_data["Married"].fillna(loan_data["Married"].mode()[0])
loan_data["Dependents"] = loan_data["Dependents"].fillna(loan_data["Dependents"].mode()[0])
loan_data["Self_Employed"] = loan_data["Self_Employed"].fillna(loan_data["Self_Employed"].mode()[0])

# One-hot encoding for categorical variables
loan_data = pd.get_dummies(loan_data)
loan_data = loan_data.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 'Self_Employed_No', 'Loan_Status_N'], axis=1)
new_column_names = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed', 'Loan_Status_Y': 'Loan_Status'}
loan_data.rename(columns=new_column_names, inplace=True)

# Feature Scaling
X = loan_data.drop(columns=['Loan_Status'])
Y = loan_data['Loan_Status']
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Streamlit App for Loan Prediction
st.title('Loan Approval Prediction System')

# Collect user input through Streamlit
with st.form(key='loan_form'):
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
    loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=1, value=360)
    credit_history = st.radio("Credit History (1 for good, 0 for bad)", [1, 0])
    gender = st.radio("Is the applicant male?", [1, 0])
    married = st.radio("Is the applicant married?", [1, 0])
    dependents_0 = st.radio("Does the applicant have 0 dependents?", [1, 0])
    dependents_1 = st.radio("Does the applicant have 1 dependent?", [1, 0])
    dependents_2 = st.radio("Does the applicant have 2 dependents?", [1, 0])
    dependents_3 = st.radio("Does the applicant have 3+ dependents?", [1, 0])
    education = st.radio("Is the applicant a graduate?", [1, 0])
    self_employed = st.radio("Is the applicant self-employed?", [1, 0])
    property_area_rural = st.radio("Does the applicant live in a rural area?", [1, 0])
    property_area_semiurban = st.radio("Does the applicant live in a semiurban area?", [1, 0])
    property_area_urban = st.radio("Does the applicant live in an urban area?", [1, 0])

    submit_button = st.form_submit_button(label='Predict')

# Model Training and Prediction
if submit_button:
    input_data = np.array([[applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history,
                            gender, married, dependents_0, dependents_1, dependents_2, dependents_3,
                            education, self_employed, property_area_rural, property_area_semiurban, property_area_urban]])
    input_data = scaler.transform(input_data)

    # RandomForestClassifier for prediction
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, Y_train)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Denied")

# Data Visualization for Outliers
plt.figure(figsize=(15, 10))
outliersColumns = loan_data.get(["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"])
sns.stripplot(data=outliersColumns, color="red", jitter=0.3, size=5)
plt.title("Outliers")
plt.show()

# Min-Max Scaling verification and outlier removal using IQR
Q1 = loan_data.astype(np.float32).quantile(0.25)
Q3 = loan_data.astype(np.float32).quantile(0.75)
IQR = Q3 - Q1
loan_data = loan_data[~((loan_data < (Q1 - 1.5 * IQR)) | (loan_data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Check for duplicates in numerical columns
columns_to_check = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for column_name in columns_to_check:
    duplicate_count = loan_data[column_name].duplicated().sum()
    if duplicate_count == 0:
        print(colored(f"No duplicate entries found in the {column_name} column.", "green", attrs=['reverse']))
    else:
        print(colored(f"Number of duplicate entries found in the {column_name} column: {duplicate_count}", "cyan", attrs=['bold']))

