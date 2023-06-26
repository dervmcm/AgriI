import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def detect_anomalies_with_isolation_forest(series):
    data = series.values.reshape(-1, 1)
    
    #model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
    model.fit(data)
    anomalies = model.predict(data)
    anomalies_series = pd.Series(anomalies, index=series.index)
    return anomalies_series


# Read the data from a CSV file
df = pd.read_csv('individualrecords/982 123766109258.csv',sep=",", index_col=0)

# Convert the 'date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
# Cleaning 'date' so it only shows once
df = df.groupby('Date')['Weight'].median().reset_index()

# Scaling the weight data
scaler = StandardScaler()
df['scaled_weight'] = scaler.fit_transform(df[['Weight']])

# Set the timestamp column as the index and convert to a series
series = df.set_index('Date')['Weight'].squeeze()

anomalies = detect_anomalies_with_isolation_forest(series)

plt.subplots(figsize=(14, 10)) 
plt.plot(df['Date'], df['Weight'], color='blue', label='test')
plt.scatter(anomalies[anomalies==-1].index, series[anomalies==-1].values, color='red', label='Anomalies')
plt.legend()
plt.title('Isolation Forest')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.grid()
plt.show()
