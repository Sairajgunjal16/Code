import pandas as pd

import numpy as np

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

# Realistic 15-investor dataset

data = pd.DataFrame({

'InvestorID': range(1, 16),

'AvgTradeSize': [1200, 8500, 3200, 15000, 4800, 9200, 1800, 7500, 4200, 11000,

950, 6800, 12500, 3000, 5800], # USD

'TradesPerMonth': [4, 22, 8, 15, 6, 18, 3, 12, 9, 20, 2, 10, 16, 5, 14],

'RiskTolerance': [2, 8, 5, 9, 4, 7, 3, 6, 5, 8, 1, 7, 9, 4, 6], # Scale 1-10

'PortfolioValue': [25000, 180000, 45000, 350000, 60000, 220000, 15000, 120000,

75000, 280000, 8000, 95000, 400000, 30000, 110000] # USD

})

print("Investor Data:")

print(data)

# Feature Selection & Scaling

X = data[['AvgTradeSize', 'TradesPerMonth', 'RiskTolerance']]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# KNN Model (unsupervised similarity search)
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')

knn.fit(X_scaled)

# Analyze Investor #5's peer group

investor_id = 5

distances, indices = knn.kneighbors([X_scaled[investor_id-1]]) # -1 for 0-based index

print(f"\nSimilar investors to Investor {investor_id}:")

print(data.iloc[indices[0]])

# Visualization

plt.figure(figsize=(12, 6))

# 2D Plot: Trade Size vs Frequency

plt.subplot(1, 2, 1)

sns.scatterplot(x='AvgTradeSize', y='TradesPerMonth',

size='PortfolioValue', hue='RiskTolerance',

data=data, palette='viridis', sizes=(20, 200))

plt.scatter(data.loc[investor_id-1, 'AvgTradeSize'],

data.loc[investor_id-1, 'TradesPerMonth'],

s=200, c='red', marker='X', label=f'Target (Investor {investor_id})')

plt.title("Trade Behavior Segmentation")

plt.xlabel("Average Trade Size ($)")

plt.ylabel("Trades/Month")

plt.legend(bbox_to_anchor=(1.05, 1))

# 3D Interactive Plot

from mpl_toolkits.mplot3d import Axes3D

plt.subplot(1, 2, 2, projection='3d')

ax = plt.gca()
ax.scatter(data['AvgTradeSize'], data['TradesPerMonth'], data['RiskTolerance'],

c=data['RiskTolerance'], cmap='viridis', s=data['PortfolioValue']/5000)

ax.scatter(data.loc[investor_id-1, 'AvgTradeSize'],

data.loc[investor_id-1, 'TradesPerMonth'],

data.loc[investor_id-1, 'RiskTolerance'],

s=200, c='red', marker='X')

ax.set_xlabel('Trade Size ($)')

ax.set_ylabel('Trades/Month')

ax.set_zlabel('Risk Tolerance')

plt.title("3D Investor Profile")

plt.tight_layout()

plt.show()
