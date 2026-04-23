import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score , precision_score , f1_score
df = pd.read_csv(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\real_projects\train_u6lujuX_CVtuZ9i.csv')
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.info())
print(df['Loan_Status'].value_counts())
print(pd.crosstab(df['Credit_History'],df['Loan_Status']))
"""If Credit_History = 1 → mostly approved
If Credit_History = 0 → mostly rejected"""
print(df['Property_Area'].value_counts())
#For categorical-like (0 or 1)
#Mode is best
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace = True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace = True)
df['Gender'].fillna(df['Gender'].mode()[0],inplace = True)
df['Married'].fillna(df['Married'].mode()[0],inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace = True)
print(df.isnull().sum())
df = df.drop(columns = 'Loan_ID',axis=1)
df = pd.get_dummies(df , columns = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status'])
print(df.info())
x = df.drop(columns = ['Loan_Status_N','Loan_Status_Y'],axis = 1)
y = df['Loan_Status_Y']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20,random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model1 = LogisticRegression(class_weight='balanced')
model1.fit(x_train_scaled,y_train)
prediction1 = model1.predict(x_test_scaled)
model2 = DecisionTreeClassifier()
model2.fit(x_train,y_train)
prediction2 = model2.predict(x_test)
model3 = RandomForestClassifier()
model3.fit(x_train,y_train)
prediction3 = model3.predict(x_test)
print('Accuracy of the LR : ',accuracy_score(y_test,prediction1))
print('Accuracy of DT : ',accuracy_score(y_test,prediction2))
print('Accuracy of RF :',accuracy_score(y_test,prediction3))
cm = confusion_matrix(y_test,prediction1)
print('confution metrics : ',cm)
print('precision of lr : ',precision_score(y_test,prediction1))
print('recalled of lr: ',recall_score(y_test,prediction1))
print('f1 score of lr : ',f1_score(y_test,prediction1))
