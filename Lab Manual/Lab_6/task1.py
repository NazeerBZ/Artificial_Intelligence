from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# fix random seed for reproducibility\
seed = 7
np.random.seed(seed)
dataframe = pd.read_csv('diabetes.csv')
dataset = dataframe.values

X = dataset[:, 0:8]
Y = dataset[:, 8]
#
##Normalizing (rescale data to the range of 0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=seed)

model = Sequential()
#start from the first hidden layer, since the input is not actually a layer   
#but inform the shape of the input, with 8 elements. 
model.add(Dense(units=12, input_shape=(8,), activation='relu')) #hidden layer 1 with input
model.add(Dense(units=8, activation='relu')) #hidden layer 2
model.add(Dense(units=1, activation='sigmoid')) #output layer
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(x_train, y_train, batch_size=10, epochs=150)
# evaluate the model
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(x_test)
y_pred = [round(x[0]) for x in predictions]
