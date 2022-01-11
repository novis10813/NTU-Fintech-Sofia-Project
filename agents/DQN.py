import tensorflow as tf
import numpy as np
import logging
import random
import itertools
import os

from collections import deque
from parameters import DQNparams
from models import Net, DuelNet
from utils import LinearAnneal

class DQN(DQNparams):
    def __init__(self, env, type='RNN', layer_norm=False, name='DQN'):
        self.env = env
        self.n_action = len(self.env.action_space)
        self.state_shape = self.env.state_size
        self.type = type
        self.layer_norm = layer_norm
        self.name = name
        
        # initialize model
        self._init_model()
        
        # initialize memory
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)
        
        # greedy policy
        self.epsilon = LinearAnneal(self.EPSILON, self.MIN_EPSILON, self.EPISODES)
        
        # initialize logging
        self.log = self._logging()
    
    def _init_model(self):
        self.policy_model = Net(self.state_shape, self.n_action, self.LR, self.type, self.layer_norm)
        self.target_model = Net(self.state_shape, self.n_action, self.LR, self.type, self.layer_norm)
        self._update_model()
        
    def _update_model(self):
        self.target_model.set_weights(self.target_model.get_weights())
    
    def _update_memory(self, transitions):
        self.replay_memory.append(transitions)
    
    def _choose_action(self, state):
        if random.random() < self.epsilon.anneal():
            return np.argmax(self.policy_model.predict(np.expand_dims(state, axis=0)))
        return random.randint(0, self.n_action)
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE * 3:
            return
        
        # pick sample batch from replay memory
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        next_states = np.array([transition[2] for transition in batch])
        rewards = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        # calculate Q value
        Qs = self.policy_model.predict(states)
        Qs_next = self.target_model.predict(next_states)
        
        Qs_next_max = tf.reduce_max(Qs_next, axis=1, keepdims=True).numpy()
        Qs_target = np.copy(Qs)
        
        for i in range(self.BATCH_SIZE):
            if not dones[i]:
                Qs_target[i, actions[i]] = rewards[i] + self.GAMMA * Qs_next_max[i]
            else:
                Qs_target[i, actions[i]] = rewards[i]
            
        self.policy_model.train_on_batch(states, Qs_target)
    
    def _save(self, path):
        self.policy_model.save(path)
    
    def _logging(self):
        formatter = logging.Formatter(r'"%(asctime)s",%(message)s')
        logger = logging.getLogger('Trade-Agents')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(f'./logs/{self.name}_{self.type}.csv')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
    
    def train(self):
        reward_list = []
        optm_cnt = 0
        for episode in range(1, self.EPISODES+1):
            self.env.reset()
            state = self.env.reset()
            done = False
            total_reward = 0
            for t in itertools():
                action = self._choose_action(state)
                next_state, reward, done = self.env.step(action)
                self._update_memory((state, action, next_state, reward, done))
                self._optimize()
                total_reward += reward
                state = next_state
                if done:
                    break

            optm_cnt += t
            reward_list.append(total_reward)
            if total_reward > max(reward_list):
                self._save(path=f'./models/{self.name}_{self.type}')
                
            if episode % self.MODEL_UPDATE == 0:
                self._update_model()
            
            self.log.info(f'{episode},{optm_cnt},{total_reward},{self.epsilon.p:.6f}')
            
class DuelDQN(DQN):
    def __init__(self, env, type='RNN', layer_norm=False, name='DuelDQN'):
        super().__init__(env, type=type, layer_norm=layer_norm)
        self.name = name
        
    def _init_model(self):
        self.policy_model = DuelNet(self.state_shape, self.n_action, self.LR, self.type, self.layer_norm)
        self.target_model = DuelNet(self.state_shape, self.n_action, self.LR, self.type, self.layer_norm)
        self._update_model()
        
class DoubleDQN(DQN):
    def __init__(self, env, type='RNN', layer_norm=False):
        super().__init__(env, type=type, layer_norm=layer_norm)
        
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE * 3:
            return
        
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        next_states = np.array([transition[2] for transition in batch])
        rewards = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        # calculate Q value
        Qs = self.policy_model.predict(states)
        Qs_next = self.target_model.predict(next_states)
        
        evaluated_action = np.argmax(self.policy_model.predict(next_states), axis=1)
        Qs_target = np.copy(Qs)
        
        for i in range(self.BATCH_SIZE):
            if not dones[i]:
                Qs_target[i, actions[i]] = rewards[i] + self.GAMMA * Qs_next[i, evaluated_action[i]]
            else:
                Qs_target[i, actions[i]] = rewards[i]
            
        self.policy_model.train_on_batch(states, Qs_target)

class CERDQN(DQN):
    def __init__(self, env, type='RNN', layer_norm=False, name='CERDQN'):
        super().__init__(env, type=type, layer_norm=layer_norm)
        self.name = name
    
    def _optimize(self):
        if len(self.replay_memory) < self.BATCH_SIZE * 3:
            return
        
        # pick sample batch from replay memory
        batch = random.sample(self.replay_memory, self.BATCH_SIZE-1)
        batch.append(self.new_transition)
        
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        next_states = np.array([transition[2] for transition in batch])
        rewards = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        # calculate Q value
        Qs = self.policy_model.predict(states)
        Qs_next = self.target_model.predict(next_states)
        
        Qs_next_max = tf.reduce_max(Qs_next, axis=1, keepdims=True).numpy()
        Qs_target = np.copy(Qs)
        
        for i in range(self.BATCH_SIZE):
            if not dones[i]:
                Qs_target[i, actions[i]] = rewards[i] + self.GAMMA * Qs_next_max[i]
            else:
                Qs_target[i, actions[i]] = rewards[i]
            
        self.policy_model.train_on_batch(states, Qs_target)
    
    def train(self):
        optm_cnt = 0
        reward_list = []
        for episode in range(1, self.EPISODES+1):
            self.env.reset()
            state = self.env.reset()
            done = False
            total_reward = 0
            for t in itertools.count():
                action = self._choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.new_transition = (state, action, next_state, reward, done)
                self._update_memory(self.new_transition)
                self._optimize()
                total_reward += reward
                state = next_state
                if done:
                    break
            
            optm_cnt += t
            reward_list.append(total_reward)
            if total_reward > max(reward_list):
                self._save(path=f'./models/{self.name}_{self.type}')
            
            # Update model every 5 episode    
            if episode % self.MODEL_UPDATE == 0:
                self._update_model()
            
            # record info
            self.log.info(f'{episode},{optm_cnt},{total_reward},{self.epsilon.p:.6f}')
    
    def test(self, model):
        '''load model'''
        while True:
            state = self.env.reset()
            self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.policy_model.predict(np.expand_dims(state, axis=0)))
                next_state, _, done = self.env.step(action)
                state = next_state