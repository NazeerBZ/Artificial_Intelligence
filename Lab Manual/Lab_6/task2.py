import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th') 

# fix random seed for reproducibility
np.random.seed(7)

(x_train, y_train), (x_test,y_test) = mnist.load_data()
#plt.imshow(x_train[5])
#print(y_train[5])
# just to make large dataset to small
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:2000]
y_test = y_test[:2000]

x_train = x_train.reshape(x_train.shape[0],1,28,28).astype('float32')
x_test = x_test.reshape(x_test.shape[0],1,28,28).astype('float32')
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255
#one hot encoded
y_train = np_utils.to_categorical(y_train) # np.unique
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape=(1,28,28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) # picks max value in 2x2 region of feature map
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2)) # randomly exclude 20% of neurons in the layer in order to reduce overfitting.
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) # it's gonna give multiple classes as an output
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(x_train, y_train, validation_split=0.22, epochs=5, batch_size=200)

loss, accuracy = model.evaluate(x_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%% \n" % (loss, accuracy*100))

y_test_predicted = model.predict_classes(x_test)
for i in range(len(x_test)):
	print("Sample=%s, Predicted=%s" % (i, y_test_predicted[i]))

