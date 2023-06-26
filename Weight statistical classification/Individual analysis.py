#import dependencies
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

'''file = open("individualrecords/982 123724743222.txt", "r")
datainlist = []
#into list
for i in file:
    fields = i.split(",")
    datainlist.append(fields)

weight_to_list = []
#extract weight into list
for i in datainlist:
    weight_to_list.append(i[2])'''

df = pd.read_csv('individualrecords/982 123724743222.csv',sep=",", index_col=0)
#df = pd.read_csv('Belmont_cows_WOW.csv')
df = df.drop(columns=(['RFID', 'Location','cattle_id','prev_diff','within_standard_dev','padding_remove']))

#df = df.describe()[['Weight']]
#df.sort_values(['Weight'], axis=0, ascending=True,inplace=True,na_position='first')
print(df)

#print(df.head())
#plt.hist(df.Weight)

#sns.displot(df['Weight'])





q1 = np.quantile(df.Weight, 0.25)
q3 = np.quantile(df.Weight, 0.75)
med = np.median(df.Weight)

def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   return outliers

#cleans by weight attribute
def reject_outliers(data, m=2):
    return data[abs(data['Weight'] - np.mean(data['Weight'])) < m * np.std(data['Weight'])]

#takes only 1 attribute
def reject_outliers_(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

'''outliers = find_outliers_IQR(df['Weight'])
print("number of outliers: "+ str(len(outliers)))
print("max outlier value: "+ str(outliers.max()))
print("min outlier value: "+ str(outliers.min()))
print(outliers)'''

cleaned = reject_outliers(df)

#print("Contents in csv file: \n", df)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


####
#double clean
cleaned_2 = reject_outliers(cleaned)

'''#Boxplots
plt.boxplot(cleaned.Weight,vert = False)
plt.show()'''

'''#Histograms
plt.hist(cleaned.Weight)
plt.show()'''

'''#LineChart
x = df['Date'].values
y1 = df['Weight'].values
fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
plt.fill_between(x, y1=y1, alpha=0.5, linewidth=2, color='seagreen')
plt.hlines(y=0, xmin=np.min(df['Date']), xmax=np.max(df['Date']), linewidth=.5)
plt.plot(cleaned.Date, cleaned.Weight, color = "red")
#plt.plot(cleaned_2.Date, cleaned_2.Weight, color="blue")
plt.show()'''


'''data = df[['Weight']]
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(n_cluster, scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
'''


'''# Convert the timestamp column to a datetime object
#cleaned['Date'] = pd.to_datetime(cleaned['Date'])

# Calculate the moving average of the temperature readings
window_size = 200 # MODIFICATION, original was 50
ma = cleaned['Weight'].rolling(window_size).mean()

# Calculate the deviation from the moving average
deviation = cleaned['Weight'] - ma

# Calculate the standard deviation of the deviation
std_deviation = deviation.rolling(window_size).std()

# Calculate the threshold for anomaly detection
threshold = 3 * std_deviation

# Detect anomalies based on deviations from the moving average
anomalies = cleaned[deviation.abs() > threshold]

print(ma)
print(deviation)
print(std_deviation)
print(threshold)
print(anomalies)

# Plot the temperature readings and the anomalies
plt.subplots(figsize=(14, 10)) # MODIFICATION, inserted
plt.plot(cleaned['Date'], cleaned['Weight'], color='blue', label='Weight')
plt.scatter(anomalies['Date'], anomalies['Weight'], color='red', label='Anomalies')
plt.plot(cleaned['Date'], ma, color='green', label='Moving Average')
plt.fill_between(cleaned['Date'], ma-threshold, ma+threshold, color='gray', alpha=0.2, label='Threshold')
plt.legend()
plt.title('Machine Weight Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.grid() # MODIFICATION, inserted
plt.show()'''


'''# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv')

print(df)

# Convert the timestamp column to a datetime object
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate the moving average of the temperature readings
window_size = 200 # MODIFICATION, original was 50
ma = df['value'].rolling(window_size).mean()

# Calculate the deviation from the moving average
deviation = df['value'] - ma

# Calculate the standard deviation of the deviation
std_deviation = deviation.rolling(window_size).std()

# Calculate the threshold for anomaly detection
threshold = 3 * std_deviation

# Detect anomalies based on deviations from the moving average
anomalies = df[deviation.abs() > threshold]

# Plot the temperature readings and the anomalies
plt.subplots(figsize=(14, 10)) # MODIFICATION, inserted
plt.plot(df['timestamp'], df['value'], color='blue', label='Temperature Readings')
plt.scatter(anomalies['timestamp'], anomalies['value'], color='red', label='Anomalies')
plt.plot(df['timestamp'], ma, color='green', label='Moving Average')
plt.fill_between(df['timestamp'], ma-threshold, ma+threshold, color='gray', alpha=0.2, label='Threshold')
plt.legend()
plt.title('Machine Temperature Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Temperature (Celsius)')
plt.grid() # MODIFICATION, inserted
plt.show()'''

data = pd.read_csv('individualrecords/982 123766109290.csv',sep=",", index_col=0)
data = data.drop(columns=(['RFID', 'Location','prev_diff','within_standard_dev','padding_remove']))

# Convert the 'date' column to datetime format
cleaned['Date'] = pd.to_datetime(data['Date'])

# Calculate the weight changes
cleaned['weight_change'] = cleaned['Weight'].diff()

# Calculate the mean and standard deviation of weight changes
mean = cleaned['weight_change'].mean()
std_dev = cleaned['weight_change'].std()

# Define a threshold for anomaly detection
threshold = 2 * std_dev  # Adjust this value based on your data and requirements

# Detect anomalies
anomalies = cleaned[cleaned['weight_change'].abs() > threshold]

'''# Print the anomalies
print("Anomalies:")
print(anomalies)'''

# Plot the weight changes over time
plt.figure(figsize=(10, 6))
plt.plot(cleaned['Date'], cleaned['weight_change'], label='Weight Change')
#plt.axhline(y=threshold, color='r', linestyle='--', label='Anomaly Threshold')
plt.scatter(anomalies['Date'], anomalies['weight_change'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Weight Change')
plt.title('Weight Change Over Time with Anomalies')
plt.legend()
plt.xticks(rotation=45)
plt.show()

'''# Plot the weight data with anomalies highlighted
plt.figure(figsize=(10, 6))
plt.plot(cleaned['Date'], cleaned['Weight'], label='Weight')
plt.scatter(anomalies['Date'], anomalies['Weight'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.title('Weight Data with Anomalies')
plt.legend()
plt.xticks(rotation=45)
plt.show()'''


'''# Apply moving average for smoothing
window_size = 3  # Adjust this value based on your preference
cleaned['smooth_weight'] = cleaned['Weight'].rolling(window_size, min_periods=1).mean()

plt.figure(figsize=(10, 6))
plt.plot(cleaned['Date'], cleaned['Weight'], color='green', label='Weight')
plt.plot(cleaned['Date'], cleaned['smooth_weight'], color='blue' ,label='Smooth Weight')
plt.scatter(anomalies['Date'], anomalies['Weight'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.title('Weight Data with Smooth Line and Anomalies')
plt.legend()
plt.xticks(rotation=45)
plt.show()'''




'''cleaned['threshold'] = mean + 2 * std_dev
anomalies = cleaned[cleaned['weight_change'].abs() > cleaned['threshold']]
# Plot the weight data with anomalies and threshold
plt.figure(figsize=(10, 6))
plt.plot(cleaned['Date'], cleaned['Weight'], label='Weight')
plt.scatter(anomalies['Date'], anomalies['Weight'], color='red', label='Anomalies')
#plt.plot(cleaned['Date'], cleaned['threshold'], color='orange', linestyle='--', label='Anomaly Threshold')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.title('Weight Data with Anomalies')
plt.legend()
plt.xticks(rotation=45)
plt.show()'''



# Read the data from a CSV file
data = pd.read_csv('individualrecords/982 123766109258.csv',sep=",", index_col=0)

# Convert the 'date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])
# Cleaning 'date' so it only shows once
data = data.groupby('Date')['Weight'].median().reset_index()

# Scaling the weight data
scaler = StandardScaler()
data['scaled_weight'] = scaler.fit_transform(data[['Weight']])

'''# Apply moving average for smoothing
window_size = 3  # Adjust
data['smooth_weight'] = data['Weight'].rolling(window_size, min_periods=1).mean()'''


# Detect anomalies using DBSCAN
eps = 0.25  # Adjust
min_samples = 3  # Adjust
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
data['anomaly'] = dbscan.fit_predict(data[['scaled_weight']])

# Filter data points with low anomaly cluster
low_anomaly_data = data[data['anomaly'] == 0]  # Adjust 
average_weight = low_anomaly_data['Weight'].mean()


# Plot the weight data with anomalies
plt.figure(figsize=(10, 6))
plt.scatter(data['Date'], data['Weight'], c=data['anomaly'], cmap='viridis')
plt.plot(low_anomaly_data['Date'], low_anomaly_data['Weight'].rolling(window=3, min_periods=1, center=True).mean(), color='red', linestyle='--', label='Average Weight')
#plt.plot(low_anomaly_data['Date'], [average_weight] * len(low_anomaly_data), color='red', linestyle='--', label='Average Weight')
#plt.scatter(anomalies['Date'], anomalies['Weight'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.title('Weight Data with Anomalies (Clustered DBSCAN)')
plt.colorbar(label='Anomaly Cluster')
plt.xticks(rotation=45)
plt.show()


# Train the anomaly detection model
model = IsolationForest(contamination=0.08)  # Adjust
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


# Calculate the weight changes
low_anomaly_data['weight_change'] = low_anomaly_data['Weight'].diff()

mean = low_anomaly_data['weight_change'].mean()
std_dev = low_anomaly_data['weight_change'].std()

# Define a threshold for anomaly detection
threshold = 2 * std_dev  # Adjust this value based on your data and requirements

# Detect anomalies
new_anomaly_test = low_anomaly_data[low_anomaly_data['weight_change'].abs() > threshold]

# Plot the weight data with anomalies highlighted
'''plt.figure(figsize=(10, 6))
plt.plot(low_anomaly_data['Date'], low_anomaly_data['Weight'], label='Weight')
plt.scatter(new_anomaly_test['Date'], new_anomaly_test['Weight'], color='red', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.title('Weight Data with Anomalies (statistical apporach)')
plt.legend()
plt.xticks(rotation=45)
plt.show()'''

