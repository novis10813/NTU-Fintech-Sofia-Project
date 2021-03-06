# IMPORTS
import pandas as pd
import math
import os.path
import time
from bitmex import *
#from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)

### API
bitmex_api_key = 'RP1ba9HvFTp0W7jGHGCk2yRb'	#Enter your own API-key here
bitmex_api_secret = 'HWBcJsQXzCof-JCy4SKwMMXH78SnwMqYoBnTlso0V_Uyt_zj' #Enter your own API-secret here

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
batch_size = 750
bitmex_client = bitmex(test=False, api_key=bitmex_api_key, api_secret=bitmex_api_secret)
#

### FUNCTIONS
def minutes_of_new_data(symbol, kline_size, data, source):
	if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
	elif source == "bitmex": old = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
	if source == "bitmex": new = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
	return old, new

def get_all_bitmex(symbol, kline_size, save = False):
	filename = '%s-%s-data.csv' % (symbol, kline_size)
	if os.path.isfile("./data/"+filename):
		return pd.read_csv("./data/"+filename)
	data_df = pd.DataFrame()
	oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "bitmex")
	delta_min = (newest_point - oldest_point).total_seconds()/60
	available_data = math.ceil(delta_min/binsizes[kline_size])
	rounds = math.ceil(available_data / batch_size)
	if rounds > 0:
		print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data in %d rounds.' % (delta_min, symbol, available_data, kline_size, rounds))
		for round_num in tqdm_notebook(range(rounds)):
			time.sleep(1)
			new_time = (oldest_point + timedelta(minutes = round_num * batch_size * binsizes[kline_size]))
			data = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=batch_size, startTime = new_time).result()[0]
			temp_df = pd.DataFrame(data)
			data_df = data_df.append(temp_df)
	data_df.set_index('timestamp', inplace=True)
	data_df.columns=['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Trades',
					'Volume', 'vwap', 'lastSize', 'turnover', 'homeNotional','foreignNotional']
	for col in data_df.columns:
		if (col !='Date')& (col!='symbol'):
			data_df[col]=data_df[col].astype(float)
	if save and rounds > 0:
		print('all caught up..!')
	return data_df 
