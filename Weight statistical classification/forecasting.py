import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Read the data from a CSV file
data = pd.read_csv('individualrecords/982 123766109258.csv',sep=",", index_col=0)

# Convert the 'date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Cleaning 'date' so it only shows once
data = data.groupby('Date')['Weight'].median().reset_index()

# Handle invalid recordings (replace with NaN)
data['Weight'] = np.where(data['Weight'] < 0, np.nan, data['Weight'])  # Assuming negative weights are invalid

# Interpolate missing values
data['Weight'] = data['Weight'].interpolate()

data.index = pd.to_datetime(data['Date'], format='%Y-%m-%d')

#get % of whole dataset to train/test
index = round(len(data.index) * 0.8)
date = data.iloc[index]['Date']

del data['Date']

sns.set()

train = data[data.index < pd.to_datetime(date, format='%Y-%m-%d')]
test = data[data.index > pd.to_datetime(date, format='%Y-%m-%d')]

plt.plot(train, color = "black")
plt.plot(test, color = "red", label = "Test Data")
plt.ylabel('Weight')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Weight Data")
#plt.show()

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
ARIMAmodel = ARIMA(y, order = (2, 1, 1))
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
SARIMAXmodel = SARIMAX(y, order = (2,1,1), seasonal_order=(2,2,2,12))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"] 
plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
plt.legend()

plt.show()