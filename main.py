import numpy as np
import pandas as pd
import logging

from agents.Baseline import LogisticPolicy,LP_train
from agents.DQN import DQN, DuelDQN, DoubleDQN, CERDQN
from data.data_bitmex import get_all_bitmex
from Env import CustomEnv
from preprocessing import preprocessing

df = get_all_bitmex("ETHUSD","1h",1)

df.columns=['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Trades',
			'Volume', 'vwap', 'lastSize', 'turnover', 'homeNotional','foreignNotional']

train_df,valid_df=preprocessing(df,1)
print("preprocessing is done")

window_list={50,100,500,1000}
for window_size in window_list:
	train_env = CustomEnv(train_df,1000,window_size,100)
	valid_env = CustomEnv(valid_df,1000,window_size,100)

	np.random.seed(window_size)
	episode_rewards, policy = LP_train(θ=np.random.rand(window_size*10),
												α=0.1,
												γ=0.99,
												window_size=window_size,
												Policy=LogisticPolicy,
												learn_env=train_env,
												valid_env=valid_env,
												MAX_EPISODES=1,
												evaluate=False)
	print("window_size:",window_size,"episode_rewards:",episode_rewards)

	### DQN Agent ###
	agent = DQN(train_env)
	agent.train()
	#################

	### Duel DQN ###
	# agent = DuelDQN(train_env)
	# agent.train(logger)
	################

	### Double DQN ###
	# agent = DoubleDQN(train_env)
	# agent.train(logger)
	################

	### CER DQN ###
	# agent = CERDQN(train_env)
	# agent.train(logger)
	################