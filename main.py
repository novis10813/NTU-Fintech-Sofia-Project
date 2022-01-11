import numpy as np
import pandas as pd
from agents.DQN import DQN, DuelDQN, DoubleDQN, CERDQN
from Env import CustomEnv

'''
df = pd.read_csv()
'''

train_df,valid_df=preproccessing("ETHBTC","5m",1)

env = CustomEnv()

### DQN Agent ###
agent = DQN(env)
agent.train()
#################

### Duel DQN ###
# agent = DuelDQN(env)
# agent.train()
################

### Double DQN ###
# agent = DoubleDQN(env)
# agent.train()
################

### CER DQN ###
# agent = CERDQN(env)
# agent.train()
################