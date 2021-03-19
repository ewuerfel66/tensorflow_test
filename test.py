from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#Load the data and split it into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Reshape the data to fit the model
X_train = X_train.reshape(60000, 28 ,28 ,1)
X_test = X_test.reshape(10000, 28, 28, 1)

#One-Hot Encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Create Model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compile the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from datetime import datetime
start = datetime.now()
hist = model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=3)
end = datetime.now()

print(f"Train Time: {(end - start).seconds}s")