#import dependencies
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('allcattle.csv',sep=",", index_col=0)
#df = pd.read_csv('Belmont_cows_WOW.csv')
df = df.drop(columns=(['RFID', 'Location','cattle_id','prev_diff','within_standard_dev','padding_remove']))

#df = df.describe()[['Weight']]
df.sort_values(['Weight'], axis=0, ascending=True,inplace=True,na_position='first')
#df.sort_values(['Weight'], axis=0, ascending=True,inplace=True,na_position='first')

print(df.describe()[['Weight']])

#print(df.head())
#plt.hist(df.Weight)

#sns.displot(df['Weight'])






'''x = df['Date'].values
y1 = df['Weight'].values
fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
plt.fill_between(x, y1=y1, alpha=0.5, linewidth=2, color='seagreen')
plt.hlines(y=0, xmin=np.min(df['Date']), xmax=np.max(df['Date']), linewidth=.5)
plt.show()'''

#plt.hist(df.Weight)
#plt.show()


q1 = np.quantile(df.Weight, 0.25)
q3 = np.quantile(df.Weight, 0.75)
med = np.median(df.Weight)

def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   return outliers

def drop_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   not_outliers = df[~((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   outliers_dropped = not_outliers.dropna().reset_index()
   return outliers_dropped

outliers = find_outliers_IQR(df['Weight'])
print("number of outliers: "+ str(len(outliers)))
print("max outlier value: "+ str(outliers.max()))
print("min outlier value: "+ str(outliers.min()))

#cleans by weight attribute
def reject_outliers(data, m=2):
    return data[abs(data['Weight'] - np.mean(data['Weight'])) < m * np.std(data['Weight'])]

#print(outliers)
#cleaned = df[(np.abs(stats.zscore(df['Weight'])) < 3)]
cleaned2 = reject_outliers(df)


'''plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.boxplot(df.Weight,vert=False)
plt.show()'''


#df = uncleaned / cleaned2 = cleaned
plt.scatter(cleaned2.Date,cleaned2.Weight)
plt.show()

plt.hist(cleaned2.Weight, bins=100)
plt.show()


cleaned2.sort_values(['Date'], axis=0, ascending=True,inplace=True,na_position='first')
x = cleaned2['Date'].values
y1 = cleaned2['Weight'].values
fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
plt.fill_between(x, y1=y1, alpha=0.5, linewidth=2, color='seagreen')
plt.hlines(y=0, xmin=np.min(cleaned2['Date']), xmax=np.max(cleaned2['Date']), linewidth=.5)
plt.show()

