# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import pickle


# ================================
# 2. LOAD DATA
# ================================
df = pd.read_csv(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\real_projects\customer_churn_dataset-training-master.csv')


# ================================
# 3. DATA CLEANING
# ================================

# Missing values are very few → safe to drop
df = df.dropna()

# Remove unnecessary column (just an ID, no predictive value)
df = df.drop(columns=['CustomerID'])

# ================================
# 4. FEATURE ENGINEERING
# ================================

# "Support Calls" was extremely dominant → caused unrealistic predictions
# Removing it to avoid model bias and make model more realistic
df = df.drop(columns=['Support Calls'])

# Convert categorical columns to numeric
df = pd.get_dummies(df, columns=['Gender', 'Subscription Type', 'Contract Length'])


# ================================
# 5. DEFINE FEATURES & TARGET
# ================================
X = df.drop(columns=['Churn'])
y = df['Churn']


# ================================
# 6. TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# ================================
# 7. SCALING (ONLY FOR LR)
# ================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ================================
# 8. TRAIN MODELS
# ================================

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

# Random Forest (final expected best model)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
# 9. EVALUATION

print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))
print("Decision Tree Accuracy:", accuracy_score(y_test, pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))

# Detailed evaluation for Random Forest (best model)
cm = confusion_matrix(y_test, pred_rf)

print("\nConfusion Matrix:\n", cm)

print("Precision:", precision_score(y_test, pred_rf))
print("Recall:", recall_score(y_test, pred_rf))
print("F1 Score:", f1_score(y_test, pred_rf))
# 10. FEATURE IMPORTANCE (BUSINESS INSIGHT)
# ================================

feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print("\nTop Features:\n", feature_importances.head(5))


# ================================
# 11. MODEL SELECTION
# ================================

# Random Forest selected because:
# - Highest accuracy
# - Best balance between precision and recall
# - Stable performance


# ================================
# 12. SAVE MODEL & SCALER
# ================================

pickle.dump(X.columns, open('columns.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(rf, open('churn_model.pkl', 'wb'))

print("\nModel and scaler saved successfully.") 