import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(7)
dataframe = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=2)
dataset = dataframe.values 
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
trainSize = int(len(dataset) * 0.67) # 67% of rows: all columns => 96 rows
trainset = dataset[0:trainSize, :]
testset = dataset[trainSize:, :]

#simple function to convert our single column of data into a two-column 
#dataset: the first column containing this month’s (t) passenger count and 
#the second column containing next month’s (t+1) passenger count, to be predicted.
look_back=1
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
       a = dataset[i:(i+look_back), 0]
       dataX.append(a)
       dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

x_train, y_train = create_dataset(trainset)
x_test, y_test = create_dataset(testset)

# reshape input to be [samples, time steps, features]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

def LSTM_model():
    model = Sequential()
    model.add(LSTM(units=4, input_shape=(1,look_back)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = LSTM_model()    
model.fit(x_train, y_train, epochs=100, batch_size=1)

# make predictions
y_test_predicted = model.predict(x_test)
for i in range(len(x_test)):
    print('X=%s, Predicted=%s' % (x_test[i], y_test_predicted[i]))
# invert predictions
y_test_predicted = scaler.inverse_transform(y_test_predicted)
y_test = scaler.inverse_transform([y_test])
# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(y_test[0], y_test_predicted[:,0]))
print('Test Score: %.2f RMSE' % (testScore))









