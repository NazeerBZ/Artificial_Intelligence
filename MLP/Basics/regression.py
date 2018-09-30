from keras.datasets import boston_housing
from keras.models import Model
from keras.layers import Dense, Input

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

visible = Input(shape=(13,))
hidden1 = Dense(units=10, activation='sigmoid')(visible)
hidden2 = Dense(units=10, activation='sigmoid')(hidden1)
hidden3 = Dense(units=10, activation='sigmoid')(hidden2)
output = Dense(units=1)(hidden3)

model = Model(inputs=visible, outputs=output)

 # stochastic gradient descent optimizer
 # for regression problems, mean squared error (MSE) is often employed
model.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['mse'])

model.fit(X_train, Y_train, epochs=100, batch_size=50)

loss, accuracy = model.evaluate(X_test, Y_test)
print('loss: %.2f, acc: %.2f' % (loss, accuracy))
