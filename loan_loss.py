import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding',
            'income', 'years_employed', 'fico_score']
target = 'default'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

def predict_expected_loss(borrower_data, loan_amount, recovery_rate=0.1):
    """
    Given borrower details and the loan amount, predict the probability of default (PD)
    and compute the expected loss: Expected Loss = PD * (1 - recovery_rate) * loan_amount.
    
    borrower_data: dict with keys matching feature names
    loan_amount: float, the loan amount
    recovery_rate: float, typically between 0 and 1 (default is 0.1)
    """
    input_df = pd.DataFrame([borrower_data])
    pd_prob = model.predict_proba(input_df)[0][1]
    expected_loss = pd_prob * (1 - recovery_rate) * loan_amount
    return round(pd_prob, 4), round(expected_loss, 2)

if __name__ == "__main__":
    borrower = {
        'credit_lines_outstanding': 2,
        'loan_amt_outstanding': 8000,
        'total_debt_outstanding': 12000,
        'income': 45000,
        'years_employed': 3,
        'fico_score': 580
    }
    loan_amount = 8000
    pd_prob, exp_loss = predict_expected_loss(borrower, loan_amount, recovery_rate=0.1)
    print("Probability of Default:", pd_prob)
    print("Expected Loss ($):", exp_loss)