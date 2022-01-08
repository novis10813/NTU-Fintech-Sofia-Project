import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random

class param:
	def __init__(self,symbol,kline_size,α,γ,episodes,lr):
		self.symbol = symbol #examples:{XBTUSD,ETHUSD}
		self.kline_size = kline_size #examples:{1m,5m,1h,1d}
		self.file_name = '%s-%s-data.csv' % (symbol,kline_size)
		self.α=α
		self.γ=γ
		self.model_name = '%s-%s-pg.txt' % (α,γ)
		self.episodes=episodes
		self.lr=lr#learning rate between old model and new model

case1=param("XBTUSD","1h",0.01,0.99,5,0.8)

import os
from data.data_bitmex import get_all_bitmex

df = get_all_bitmex(case1.symbol,case1.kline_size,save=True)
df.columns=['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Trades',
			'Volume', 'vwap', 'lastSize', 'turnover', 'homeNotional','foreignNotional']

#rl
from utils import TradingGraph
from Env import CustomEnv
from agent1 import LogisticPolicy
from agent1 import train

divider=int(8*len(df)/10)
learn_env=CustomEnv(df[:divider],1000,50,100)
valid_env=CustomEnv(df[divider:],1000,50,100)

episode_rewards, policy = train(θ=np.random.rand(500),
								α=case1.α,
								γ=case1.γ,
								Policy=LogisticPolicy,
								learn_env=learn_env,
								valid_env=valid_env,
								MAX_EPISODES=case1.episodes,
								evaluate=True)
if os.path.isfile(case1.model_name)==0:
	with open(case1.model_name, 'w') as f:
		for item in policy.θ :
			print(item)
			f.write("%s\n" % item)

old_θ=[]
with open(case1.model_name,'r') as f:
	for item in f:
		temp=item.replace('[','').replace(']\n','')
		old_θ.append(float(temp))
sum_list=[]
for (item1,item2) in zip(policy.θ,old_θ):
	sum_list.append(item1*(1-case1.lr)+item2*case1.lr)
episode_rewards, policy = train(θ=sum_list,
								α=case1.α,
								γ=case1.γ,
								Policy=LogisticPolicy,
								learn_env=learn_env,
								valid_env=valid_env,
								MAX_EPISODES=case1.episodes,
								evaluate=True)

with open(case1.model_name, 'w') as f:
	for item in policy.θ:
		f.write("%s\n" % item)