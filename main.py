import numpy as np
import pandas as pd
from agents.Baseline import LogisticPolicy,LP_train
from agents.DQN import DQN, DuelDQN, DoubleDQN, CERDQN
from Env import CustomEnv
from parameters import DQNparams
from preprocessing import preprocessing

param=DQNparams()

train_df,valid_df=preprocessing("ETHUSD","1h",1)
print("preprocessing is done")

window_list={50,100,500,1000}
for window_size in window_list:
	train_env = CustomEnv(train_df,1000,window_size,100)
	valid_env = CustomEnv(valid_df,1000,window_size,100)

	episode_rewards, policy = LP_train(θ=np.random.rand(window_size*10),
									α=0.01,
									γ=0.99,
									window_size=window_size,
									Policy=LogisticPolicy,
									learn_env=train_env,
									valid_env=valid_env,
									MAX_EPISODES=5,
									evaluate=True)
	print("window_size:",window_size,"episode_rewards:",episode_rewards)
	### DQN Agent ###
	agent = DQN(train_env)
	agent.train()
	print(agent.log)
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