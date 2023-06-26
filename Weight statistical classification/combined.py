import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import datetime
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Read the data from a CSV file
data = pd.read_csv('individualrecords/982 123766109258.csv',sep=",", index_col=0)
#data = pd.read_csv('individualrecords/982 123724743222.csv',sep=",", index_col=0)
#data = pd.read_csv('individualrecords/982 123724743229.csv',sep=",", index_col=0)

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

# Cleaning 'date' so it only shows once
clean_data = clean_data.groupby('Date')['Weight'].median().reset_index()

print(clean_data)

clean_data.index = pd.to_datetime(clean_data['Date'], format='%Y-%m-%d')

#get % of whole dataset to train/test
index = round(len(clean_data.index) * 0.8)
date = clean_data.iloc[index]['Date']

del clean_data['Date']

sns.set()

train = clean_data[clean_data.index < pd.to_datetime(date, format='%Y-%m-%d')]
test = clean_data[clean_data.index > pd.to_datetime(date, format='%Y-%m-%d')]

plt.plot(train, color = "black")
plt.plot(test, color = "red", label = "Test Data")
plt.ylabel('Weight')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Weight Data")
#plt.show()

#Training set
y = train['Weight']

#ARMA model
ARMAmodel = SARIMAX(y, order = (1, 0, 1))
ARMAmodel = ARMAmodel.fit()

y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 

y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 

arma_rmse = np.sqrt(mean_squared_error(test["Weight"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)

plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.legend()


#ARIMA model
ARIMAmodel = ARIMA(y, order = (1, 1, 1))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
plt.legend()

arma_rmse = np.sqrt(mean_squared_error(test["Weight"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)

#SARIMA model
SARIMAXmodel = SARIMAX(y, order = (1,1,1), seasonal_order=(2,2,2,12))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
plt.legend()

plt.show()