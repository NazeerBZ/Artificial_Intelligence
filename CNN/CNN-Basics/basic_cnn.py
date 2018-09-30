import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# reduce 60000 training set and use only 33% of training data 
# to expedite the training process
X_train, _ ,  Y_train, _ = train_test_split(X_train, Y_train, test_size=0.67, random_state=7)

# CNN input layer(s) expacts inputs as image and it's depth must defined
# reshaping X dataset for CNN.
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# converting y data into categorical (one-hot encoding)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

def cnn_model():
    visible = Input(shape=(1,28,28))
    conv1 = Conv2D(filters=50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(visible)
    maxpool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    flat1 = Flatten()(maxpool1)
    hidden1 = Dense(units=50, activation='relu')(flat1)
    output = Dense(units=10, activation='softmax')(hidden1)

    model = Model(inputs=visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       
    return model


model = cnn_model()
model.fit(X_train, Y_train, epochs=10, batch_size=100)

loss, accuracy = model.evaluate(X_test, Y_test)
print('loss: %.2f, acc: %.2f' % (loss, accuracy*100))

Y_test_proba = model.predict(X_test)
Y_test_classes = [round(np.argmax(y, axis=None, out=None)) for y in Y_test_proba]
