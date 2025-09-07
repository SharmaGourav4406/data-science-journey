import pandas as pd
import numpy as np
df =pd.read_csv(r"C:\Users\GOURAV SHARMA\Downloads\ecommerce_orders_messy.csv")
print(df.info())
#orderdate should be int
#returned or not should be boolean
print(df.head())
print(df.describe())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['DiscountPct'] = df['DiscountPct'].fillna(df['DiscountPct'].mean())
df['TotalAmount'] = df['TotalAmount'].fillna(df['TotalAmount'].mean())
df['DeliveryDays'] = df['DeliveryDays'].fillna(df["DeliveryDays"].median())
df['UnitPrice'] = df['UnitPrice'].fillna(df['UnitPrice'].mean())
print(df.isnull().sum())
#to covert the datatypes of different columns
df['OrderDate'] = pd.to_datetime(df['OrderDate'],errors = 'coerce')
df['Returned'] = df['Returned'].map({'Yes':True,'No':False})
df['ProductCategory'] = df['ProductCategory'].astype("category")
df['PaymentMethod'] = df['PaymentMethod'].astype("category")
df[df['Age']<0]
df['Age'] = df['Age'].clip(lower=0, upper=100)
median_val = df.loc[df['Quantity'] > 0, 'Quantity'].median()
df.loc[df['Quantity'] <= 0, 'Quantity'] = median_val
print(df.info())

#df.loc[df["Quantities"] > 0, "Quantities"].median() → computes the median from valid values only.
#df.loc[df["Quantities"] <= 0, "Quantities"] = median_val → directly replaces all invalids in one step.
df['PaymentMethod'] = df['PaymentMethod'].replace({
'upi' : 'UPI',
'cod': 'Cash On Delivery'})
df['City'] = df['City'].replace({'Mumbay' : 'Mumbai'})
df['Returned'] = df['Returned'].replace({
    'Y' : 'Yes',
    'N' : 'No' , 
    '1' : 'Yes',
    '0' : 'No',
    'maybe' : np.nan
})
df['Returned'] = df['Returned'].map({'Yes': True , 'No' : False})
print(df['TotalAmount'].describe())
df = df[df['TotalAmount'] < 999999]
df['TotalAmount'] = df['TotalAmount'].clip(upper = 100000)