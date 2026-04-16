import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\customer Segmentation\customer_data.csv')

# Basic sanity checks
print(df.info())
print(df.isnull().sum())

# 2. FEATURE SELECTION
# Choosing key behavioral + financial features for segmentation
features = ['Annual_Income', 'Total_Amount_Spent', 'Customer_Age', 'Last_Purchase_Recency']
X = df[features]

# 3. SCALING (CRUCIAL for KMeans)
# KMeans is distance-based → scaling ensures fair clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. ELBOW METHOD (find optimal k)
inertia = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

print('Inertia:', inertia)
# Look for point where drop slows → optimal number of clusters

# 5. FINAL MODEL
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(df['Cluster'].value_counts())
# 6. CLUSTER ANALYSIS (MOST IMPORTANT)
# Understanding behavior of each segment
print(df.groupby('Cluster')[features].mean())

# 7. VISUALIZATION
# Sample for cleaner visualization (avoids overcrowding)
df_sample = df.sample(2000)

# Income vs Spending
plt.scatter(df_sample['Annual_Income'], df_sample['Total_Amount_Spent'], c=df_sample['Cluster'])
plt.xlabel('Annual Income')
plt.ylabel('Total Spend')
plt.title('Customer Segments (Income vs Spending)')
plt.savefig('Annual_Income_vs_Total_Spend.png',dpi = 300,bbox_inches = 'tight')
plt.show()
# Age vs Spending
plt.scatter(df['Customer_Age'], df['Total_Amount_Spent'], c=df['Cluster'], alpha=0.3)
plt.xlabel('Customer Age')
plt.ylabel('Total Spend')
plt.title('Customer Segments (Age vs Spending)')
plt.savefig('Customer_Age_vs_Total_Amount_Spent.png',dpi = 300,bbox_inches = 'tight')
plt.show()