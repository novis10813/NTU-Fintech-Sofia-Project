from data.data_bitmex import get_all_bitmex

def preprocessing(symbol,kline_size,process_class=1):
	df = get_all_bitmex(symbol,kline_size,save=True)
	df.columns=['Date', 'symbol', 'Open', 'High', 'Low', 'Close', 'Trades',
				'Volume', 'vwap', 'lastSize', 'turnover', 'homeNotional','foreignNotional']
	
	train_ratio = 0.8
	df = df[:int(train_ratio * len(df))]
	train_df = df[:int(train_ratio * len(df))]
	valid_df = df[int(train_ratio * len(df)):]
	normalized_train_df=train_df
	normalized_valid_df=valid_df
	normalized_train_close = train_df['Close']
	normalized_valid_close = valid_df['Close']

	if process_class==1:
		train_mean = train_df.mean()
		train_std = train_df.std()
		normalized_train_df = (normalized_train_df - train_mean)/train_std
		normalized_valid_df = (normalized_valid_df - train_mean)/train_std
		normalized_train_close = normalized_train_df['Close']
		normalized_valid_close = normalized_valid_df['Close']

	return normalized_train_df,normalized_valid_df