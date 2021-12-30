import tensorflow as tf
from tensorflow.keras import layers
from parameters import params

class Net:
    def __init__(self, env):
        super(Net, self).__init__()
        self.layer_1 = layers.Dense(32, activation='relu')
        self.layer_2 = layers.Dense(64, activation='relu')
        self.layer_3 = layers.Dense(128, activation='relu')
        self.output = layers.Dense(len(env.action_space), activation='linear')
    
    def call(self, x):
        x = layers.Flatten()(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.output(x)
        return x