import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
import keras
import random
import os

class FqlAgent:
	def __init__(self,hidden_units,learning_rate):
		self.learning_rate=learning_rate
		self.max_treward=0
		self.trewards=list()
		self.treward=0
		self.model=self.build_model(hidden_units,learning_rate)
		self.epsilon=0.01
		self.window_size=50
	def build_model(self,hu,learning_rate):
		if os.path.isdir("fqlmodel"):
			print("using existed nn model")
			model = keras.models.load_model("fqlmodel")
		else:
			model=Sequential()
			model.add(Dense(hu,input_shape=(500,),activation='relu'))
			#model.add(Dropout(0.3,seed=100))
			model.add(Dense(hu,activation='relu'))
			#model.add(Dropout(0.3,seed=100))
			model.add(Dense(2,activation='linear'))
			model.compile(loss='mse',optimizer=RMSprop(learning_rate=learning_rate))
			print(model.summary())
		return model
	def act(self,state,env):
		if random.random()<= self.epsilon:
			return np.random.choice(env.action_space,1)
		action=self.model.predict(state)
		return np.argmax(action)+1
	def learn(self,episodes,learn_env):
		for e in range(1,episodes+1):
			state=learn_env.reset()
			self.treward=0
			state=state.reshape(1,500)
			action_sum=0
			n=len(learn_env.df)
			for _ in range(n-100):
				if _ % 1000==0:
					print("_:",_,"treward:",self.treward,"action_sum",action_sum)
					action_sum=0
				action = self.act(state,learn_env)
				action_sum =action_sum+action
				obs,reward,done= learn_env.step(action)
				self.treward+=reward
				obs=obs.reshape(1,500)
				state=obs
				if done:
					print("done")
					break
			self.modelsave()
	def valid(self,episodes,valid_env):
		self.epsilon=0
		for e in range(1,episodes+1):
			state=valid_env.reset()
			self.treward=0
			state=state.reshape(1,500)
			action_sum=0
			n=len(valid_env.df)
			for _ in range(n):
				if _ % 1000==0:
					print("_:",_,"treward:",self.treward,"action_sum",action_sum)
					action_sum=0
				action = self.act(state,valid_env)
				action_sum =action_sum+action
				obs,reward,done= valid_env.step(action)
				self.treward+=reward
				obs=obs.reshape(1,500)
				state=obs
		print("e:",e," total treward:",self.treward)
	def modelsave(self):
		self.model.save("fqlmodel")