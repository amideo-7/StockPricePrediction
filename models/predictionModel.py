# Importing Libraries
import math

import yfinance as yf
import numpy as np
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM

import pickle


# Reading the data
start = dt.datetime(2012,1,1)
end = dt.date.today()

company1 = 'AAPL'
company2 = 'GOOGL'
company3 = 'TSLA'

df1 = yf.download(company1, start, end)
df2 = yf.download(company2, start, end)
df3 = yf.download(company3, start, end)

# Filtering closing stock prices
data1 = df1.filter(['Close'])
data2 = df2.filter(['Close'])
data3 = df3.filter(['Close'])

# Converting the dataframes to numpy arrays
dataset1 = data1.values
dataset2 = data2.values
dataset3 = data3.values

# Getting the length of the datasets
trainDataLen1 = math.ceil(len(dataset1)*0.8)
trainDataLen2 = math.ceil(len(dataset2)*0.8)
trainDataLen3 = math.ceil(len(dataset3)*0.8)

# Scaling the Data

# Initializing the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

scaledData1 = scaler.fit_transform(dataset1)
scaledData2 = scaler.fit_transform(dataset2)
scaledData3 = scaler.fit_transform(dataset3)

# Creating training dataset
trainData1 = scaledData1[0:trainDataLen1,:]
trainData2 = scaledData2[0:trainDataLen2,:]
trainData3 = scaledData3[0:trainDataLen3,:]

x_train1, y_train1 = [], []
x_train2, y_train2 = [], []
x_train3, y_train3 = [], []

for i in range(60, len(trainData1)):
    x_train1.append(trainData1[i-60:i,0])
    y_train1.append(trainData1[i,0])

for i in range(60, len(trainData2)):
    x_train2.append(trainData2[i-60:i,0])
    y_train2.append(trainData2[i,0])

for i in range(60, len(trainData3)):
    x_train3.append(trainData3[i-60:i,0])
    y_train3.append(trainData3[i,0])

x_train1, y_train1 = np.array(x_train1), np.array(y_train1)
x_train2, y_train2 = np.array(x_train2), np.array(y_train2)
x_train3, y_train3 = np.array(x_train3), np.array(y_train3)

# Reshaping the training dataset for LSTM model - converting 2-D data to 3-D data
x_train1 = np.reshape(x_train1, (x_train1.shape[0],x_train1.shape[1],1))
x_train2 = np.reshape(x_train2, (x_train2.shape[0],x_train2.shape[1],1))
x_train3 = np.reshape(x_train3, (x_train3.shape[0],x_train3.shape[1],1))

# Building the LSTM model
model1 = Sequential()
model1.add(LSTM(50, return_sequences=True, input_shape=(x_train1.shape[1],1)))
model1.add(LSTM(50, return_sequences=False))
model1.add(Dense(25))
model1.add(Dense(1))

model2 = Sequential()
model2.add(LSTM(50, return_sequences=True, input_shape=(x_train1.shape[1],1)))
model2.add(LSTM(50, return_sequences=False))
model2.add(Dense(25))
model2.add(Dense(1))

model3 = Sequential()
model3.add(LSTM(50, return_sequences=True, input_shape=(x_train1.shape[1],1)))
model3.add(LSTM(50, return_sequences=False))
model3.add(Dense(25))
model3.add(Dense(1))

# Compiling the model with optimizer and loss function
model1.compile(optimizer='adam', loss='mean_squared_error')
model2.compile(optimizer='adam', loss='mean_squared_error')
model3.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model1.fit(x_train1,y_train1, batch_size=1,epochs=1)
model2.fit(x_train2,y_train2, batch_size=1,epochs=1)
model3.fit(x_train3,y_train3, batch_size=1,epochs=1)

# Dumping the models to pickle file
pickle.dump(model1,open('model1.pkl','wb'))
pickle.dump(model2,open('model2.pkl','wb'))
pickle.dump(model3,open('model3.pkl','wb'))

print("Done!")