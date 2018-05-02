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
scaler = MinMaxScaler(feature_range=(0, 1)) # only works on array with 2 dimention
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
loss, accuracy = model.evaluate(x_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%% \n" % (loss, accuracy*100))

# make predictions
y_test_predicted = model.predict_classes(x_test)
for i in range(len(x_test)):
    print('Sample=%s, Predicted=%s' % (i,y_test_predicted[i]))
    
  