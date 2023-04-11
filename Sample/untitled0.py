# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 08:56:45 2023

@author: Bui Son Trang
"""

import numpy as np
import pandas as pd
from statistics import mode

#nrows is the number of rows to be read to the variable. The original file can be 1 million rows so limit it will take less time to run 
measure = pd.read_csv('accel-05.csv',header=0,parse_dates=[0],nrows=15000)

result = pd.read_csv('halter-05.csv',header=0,parse_dates=[0],nrows=15000)

#merging the measure with result that has the highest matching timestamp. the timestamp between two files doesn't match, so I have to do it this way
data = pd.merge_asof(result,measure,on='timestamp',direction='nearest')

#get x,y,z
measure = data.iloc[:,2:]

#prediction
result = data.iloc[:,1]

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#scale x,y,z
scaler = StandardScaler()
measure = scaler.fit_transform(measure)

#dummy variable
encoder = OneHotEncoder()

#window_size is the number of rows you want to consider. For example, the data is recorded at 10Hz, meaning 10 records per seconds.
#window_size of 100 means that every time we consider the data, we consider it over the span of 100/10 = 10 seconds
#step_size is the number of records you will move from the previous reading. For example, after reading the first 100 records as a whole, the next reading will be from position 2 to 102
window_size = 100
step_size = 2

segments = []
arr = []
for i in range(0, measure.shape[0] - window_size + 1, step_size):
    segment = measure[i:i+window_size]
    segments.append(segment)
    arr.append(mode(result[i:i+window_size]))

#reshape the result to fit in the model
X = np.vstack(segments)

X = X.reshape((-1, window_size, 3))

y = encoder.fit_transform(np.array(arr).reshape(-1,1)).toarray()

#split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 188)

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten

#building the model
model = Sequential()

#you can change the number of filters and kernel_size. filters is the number of nodes at this layer. kernel_size is a thing of Convolutional Layer. Don't make it large.
model.add(Conv1D(filters=100, kernel_size=3, activation='relu', input_shape=(window_size,3)))

model.add(Conv1D(filters=50, kernel_size=3, activation='relu',padding='same'))

model.add(Conv1D(filters=20, kernel_size=3, activation='relu',padding='same'))

model.add(Flatten())

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#epochs is the number of times you will run over the dataset. batch_size is the number of records as inputs each time

model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=20)

#the first number is the loss number. The second number is the accuracy
print(model.evaluate(X_test, y_test))
