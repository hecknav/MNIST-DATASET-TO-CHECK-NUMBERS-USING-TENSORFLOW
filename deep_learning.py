import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy,pandas

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

x_train=x_train/255
x_test=x_test/255
model=tf.keras.Sequential([tf.keras.layers.Input(shape=(28,28)),tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dense(10,activation="softmax") ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()


his=model.fit(x_train,y_train,epochs=5,validation_split=0.1)

test_loss,test_accuracy=model.evaluate(x_test,y_test)

print("accuracy %",test_accuracy*100)