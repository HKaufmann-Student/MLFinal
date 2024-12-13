import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def create_classification_model(input_shape):
    inputs = Input(shape=input_shape)

    # L2 regularization
    l2_reg = regularizers.l2(0.001)

    x = LSTM(50, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg, dropout=0.3)(inputs)
    x = BatchNormalization()(x)

    x = LSTM(50, return_sequences=True, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg, dropout=0.3)(x)
    x = BatchNormalization()(x)

    x = LSTM(50, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg, dropout=0.3)(x)
    x = BatchNormalization()(x)

    x = Dense(50, activation='relu', kernel_regularizer=l2_reg)(x)
    x = Dropout(0.3)(x)

    x = Dense(25, activation='relu', kernel_regularizer=l2_reg)(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
