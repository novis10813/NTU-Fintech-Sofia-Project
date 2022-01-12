import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

def RNN_Block(x, layer_norm=False):
    if layer_norm:
        x = layers.SimpleRNN(64, activation='relu', return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.SimpleRNN(128, activation='relu', return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.SimpleRNN(256, activation='relu', return_sequences=False)(x)
        x = layers.LayerNormalization()(x)
    else:
        x = layers.SimpleRNN(64, activation='relu', return_sequences=True)(x)
        x = layers.SimpleRNN(128, activation='relu', return_sequences=True)(x)
        x = layers.SimpleRNN(256, activation='relu', return_sequences=False)(x)
    x = layers.Flatten()(x)
    return x

def LSTM_Block(x, layer_norm=False):
    if layer_norm:
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.LSTM(256, return_sequences=False)(x)    
        x = layers.LayerNormalization()(x)
    else:
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.Flatten()(x)
    return x

def GRU_Block(x, layer_norm=False):
    if layer_norm:
        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.GRU(128, return_sequences=True)(x)
        x = layers.LayerNormalization()(x)
        x = layers.GRU(256, return_sequences=False)(x)
        x = layers.LayerNormalization()(x)
    else:
        x = layers.GRU(64, return_sequences=True)(x)
        x = layers.GRU(128, return_sequences=True)(x)
        x = layers.GRU(256, return_sequences=False)(x)
    x = layers.Flatten()(x)
    return x

def Net(state_shape, n_action, lr, type='RNN', layer_norm=False):
    input = Input(shape=state_shape)
    if type == 'RNN':
        x = RNN_Block(input, layer_norm)
    elif type == 'LSTM':
        x = LSTM_Block(input, layer_norm)
    else:
        x = GRU_Block(input, layer_norm)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(n_action, activation='linear')(x)
    model = Model(input, output)
    model.compile(optimizer=RMSprop(learning_rate=lr), loss=Huber)
    return model

def DuelNet(state_shape, n_action, lr, type='RNN', layer_norm=False):
    input = Input(shape=state_shape)
    if type == 'RNN':
        x = RNN_Block(input, layer_norm)
    elif type == 'LSTM':
        x = LSTM_Block(input, layer_norm)
    else:
        x = GRU_Block(input, layer_norm)
    a = layers.Dense(512, activation='relu')(x)
    a = layers.Dense(n_action, activation='linear')(a)
    v = layers.Dense(512, activation='relu')(x)
    v = layers.Dense(1, activation='linear')(v)
    output = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
    model = Model(input, output)
    model.compile(optimizer=RMSProp(learning_rate=lr), loss=Huber)
    return model