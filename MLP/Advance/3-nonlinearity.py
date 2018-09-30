from keras.models import  Model
from keras.layers import Input, Dense
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist # Handwritten digits dataset
from sklearn.model_selection import train_test_split
import numpy as np


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# reduce 60000 training set and
# use only 33% of training data to expedite the training process
X_train, _ , Y_train, _ = train_test_split(X_train, Y_train, test_size=0.67, random_state=7)

# reshaping dataset for multilayer perceptron
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# converting y data into categorical (one-hot encoding)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


def mlp_model():
    visible = Input(shape=(784,))
    hidden1 = Dense(units=50, activation='relu')(visible)
    hidden2 = Dense(units=50, activation='relu')(hidden1)
    hidden3 = Dense(units=50, activation='relu')(hidden2)
    output = Dense(units=10, activation='softmax')(hidden3)
    model = Model(inputs=visible, outputs=output)
    
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model

model = mlp_model()
model.fit(X_train, Y_train, epochs=30, batch_size=100)

loss, accuracy = model.evaluate(X_test, Y_test)
print('loss: %.2f, acc: %.2f' % (loss, accuracy*100))

Y_test_proba = model.predict(X_test)
Y_test_classes = [round(np.argmax(y, axis=None, out=None)) for y in Y_test_proba]

#for i in range(len(Y_test_classes)):
#    print('Sample:%s, Predicted:%s' % (i, Y_test_classes[i]))



