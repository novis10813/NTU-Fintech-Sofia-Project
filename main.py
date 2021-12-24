import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
#from tensorflow.keras.optimizers import Adam,RMSprop
import random

#data proccessing(checking and downloading raw data,change columns to feat env.py's requirement)
import os
from data_bitmex import get_all_bitmex
#symbol=input("examples:{XBTUSD,ETHUSD}\nsymbol:")
#kline_size=input("examples:{1m,5m,1h,1d}\nkline_size:")
symbol="ETHUSD"
kline_size="1h"
filename = '%s-%s-data.csv' % (symbol, kline_size)

if os.path.isfile(filename)==0:
	data=get_all_bitmex(symbol,kline_size,save=True)
df=pd.read_csv(filename)
df.columns=['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Trades',
       		'Volume', 'vwap', 'lastSize', 'turnover', 'homeNotional','foreignNotional']
for col in df.columns:
	if (col !='Date')& (col!='symbol'):
		df[col]=df[col].astype(float)

print(df[:5])

"""from data_twstock import get_tw_stock_data
start_year,start_month,end_year,end_month,stock_code=2010,1,2021,11,2350
filename = '%s-%s-%s-%s-%s-data.csv' % (stock_code, start_year,start_month,end_year,end_month)
if os.path.isfile(filename)==0:
	get_tw_stock_data(start_year,start_month,end_year,end_month,stock_code)
df=pd.read_csv(filename)
print(df[:5])"""


#rl
from utils import TradingGraph
from Env import CustomEnv
from agent import FqlAgent
#from agent1 import LogisticPolicy

divider=int(8*len(df)/10)
learn_env=CustomEnv(df[:divider],1000,50,100)
valid_env=CustomEnv(df[divider:],1000,50,100)
agent=FqlAgent(24,0.0001)
episodes=50
agent.learn(5,learn_env)
agent.valid(5,valid_env)
agent.save()
