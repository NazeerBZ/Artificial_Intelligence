from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

dataframe = pd.read_csv('diabetes.csv')
dataset = dataframe.values

X = dataset[:,:8] #input values
Y = dataset[:,8:] #output values

#Normalizing (rescale data to the range of 0 to 1)
scaler = MinMaxScaler(feature_range = (0, 1)) #only works on array with 2 dimention
X = scaler.fit_transform(X)

#Separating Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

#Develop Model using Functional API
visible = Input(shape=(8,))
hidden1 = Dense(units=12, activation='relu')(visible) # it will receive input from input layer
hidden2 = Dense(units=8, activation='relu')(hidden1)
output = Dense(units=1, activation='sigmoid')(hidden2)

model = Model(inputs=visible, outputs=output)
model.summary()

#Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit Model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=10)

#Evaluate Model
loss, accuracy = model.evaluate(X_test, Y_test)
print('\nloss: %.2f, acc: %.2f \n' % (loss, accuracy*100))

# Make Predictions
Y_test_proba = model.predict(X_test)
Y_test_classes = [round(y[0]) for y in Y_test_proba]
for i in range(len(Y_test_classes)):
    print('Sample:%s, Predicted:%s' % (i, Y_test_classes[i]))







