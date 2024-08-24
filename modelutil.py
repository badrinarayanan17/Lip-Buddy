# Contains the Model Architecture

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, LSTM, Dense, Dropout, 
                                     Bidirectional, MaxPool3D, Activation, 
                                     TimeDistributed, Flatten)
from utils import char_to_num

def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # Ensuring the output units match the vocabulary size + 1 for the blank token used in CTC (Beam Search Decoding)
    model.add(Dense(len(char_to_num.get_vocabulary()) + 1, kernel_initializer='he_normal', activation='softmax'))

    # Loading the model weights in .h5 format
    # Otherwise we will face error. Keras will not support models without it's format.
    model.load_weights(os.path.join('models', 'checkpoint.weights.h5'))

    return model
