import tensorflow as tf
from keras.src.layers import Conv2D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import layers, models

class DeepANN():
    def simple_model(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(128,activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="leaky_relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def CNN_MODEL(self):
        model = Sequential()
        model.add(Conv2D(32,(3,3) , activation='relu' , input_shape=(28,28,3)))
        model.add(layers.MaxPooling2D(2,2))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model
