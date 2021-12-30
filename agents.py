import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam, RMSprop

import numpy as np
import random
import os

from collections import deque
from parameters import Hyperparams
from models import Net
from utils import LinearAnneal

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
			model = tf.keras.models.load_model("fqlmodel")
		else:
			model=Sequential()
			model.add(layers.Dense(hu,input_shape=(500,),activation='relu'))
			#model.add(Dropout(0.3,seed=100))
			model.add(layers.Dense(hu,activation='relu'))
			#model.add(Dropout(0.3,seed=100))
			model.add(layers.Dense(2,activation='linear'))
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

class DQN(Hyperparams):
    def __init__(self, env):
        self.env = env
        self.n_action = len(self.env.action_space)
        self.observation = self.env.state_size
        
        # initialize model
        self._model_init()
        
        # initialize memory
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)
        
        # greedy policy
        self.epsilon = LinearAnneal(self.EPSILON, self.MIN_EPSILON, self.EPISODES)
    
    def _model_init(self):
        self.policy_model = Net(self.env)
        self.target_model = Net(self.env)
        self.policy_model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.LR))
        self.target_model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.LR))
        self._update_model()
        
    def _update_model(self):
        self.target_model.set_weights(self.target_model.get_weights())
    
    def _update_memory(self, transitions):
        self.replay_memory.append(transitions)
    
    def _choose_action(self, state):
        
        if random.random() < self.epsilon.anneal():
            return np.argmax(self.policy_model.predict(state)[0])
        return random.randint(0, 2)
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        
        # pick sample batch from replay memory
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])
        next_states = np.array([transition[2] for transition in batch])
        
        q = self.policy_model.predict(states)
        next_q = self.target_model.predict(next_states)
        
        for index, (state, action, next_state, reward) in enumerate(batch):
            q[index][action] = (1 - self.GAMMA) * q[index][action] + self.GAMMA * (reward + np.amax(next_q[index])*self.DISCOUNT)
        
        self.policy_model.fit(states, q, verbose=0)
        
    def train(self):
        for episode in range(1, self.EPISODES+1):
            self.env.reset()
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self._choose_action(state)
                next_state, reward, done = self.env.step(action)
                self._update_memory((state, action, next_state, reward))
                self._optimize()
                total_reward += reward
                state = next_state
            
            if episode % self.MODEL_UPDATE == 0:
                self._update_model()
            
            print(f'episode: {episode} | reward {total_reward}')