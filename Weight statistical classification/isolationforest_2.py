import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Read the data from a CSV file
data = pd.read_csv('individualrecords/982 123766109258.csv',sep=",", index_col=0)

# Convert the 'date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Cleaning 'date' so it only shows once
df = data.groupby('Date')['Weight'].median().reset_index()

# Handle invalid recordings (replace with NaN)
data['Weight'] = np.where(data['Weight'] < 0, np.nan, data['Weight'])  # Assuming negative weights are invalid

# Interpolate missing values
data['Weight'] = data['Weight'].interpolate()

# Preprocess the data
scaler = StandardScaler()
data['scaled_weight'] = scaler.fit_transform(data[['Weight']])

# Train the anomaly detection model
model = IsolationForest(contamination=0.08)  # Adjust the contamination parameter based on your data and requirements
model.fit(data[['scaled_weight']])

# Predict anomalies
data['anomaly'] = model.predict(data[['scaled_weight']])
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})  # Convert predictions to binary (0: normal, 1: anomaly)

# Plot the weight data with anomalies
plt.figure(figsize=(10, 6))
plt.scatter(data['Date'], data['Weight'], c=data['anomaly'], cmap='viridis')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.title('Weight Data with Anomalies (Isolation Forest)')
plt.colorbar(label='Anomaly')
plt.xticks(rotation=45)
plt.show()

# Remove anomalies from the data
clean_data = data[data['anomaly'] == 0]

# Preprocess the clean data again
clean_data['scaled_weight'] = scaler.transform(clean_data[['Weight']])

# Train the final anomaly detection model with clean data
final_model = IsolationForest(contamination=0.08)  # Adjust the contamination parameter based on your data and requirements
final_model.fit(clean_data[['scaled_weight']])

# Predict final anomalies
clean_data['anomaly'] = final_model.predict(clean_data[['scaled_weight']])
clean_data['anomaly'] = clean_data['anomaly'].map({1: 0, -1: 1})  # Convert predictions to binary (0: normal, 1: anomaly)

# Plot the weight data without anomalies
plt.figure(figsize=(10, 6))
plt.scatter(clean_data['Date'], clean_data['Weight'], c=clean_data['anomaly'], cmap='viridis')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.title('Weight Data without Anomalies (Isolation Forest)')
plt.colorbar(label='Anomaly')
plt.xticks(rotation=45)
plt.show()


