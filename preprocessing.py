import pandas as pd

def preprocessing(df,process_class):
	train_ratio = 0.8
	df=df[['Date', 'Open', 'High', 'Low', 'Close','Volume']]
	train_df = df[:int(train_ratio * len(df))]
	valid_df = df[int(train_ratio * len(df)):]
	normalized_train_df=train_df
	normalized_valid_df=valid_df
	normalized_train_close = train_df['Close']
	normalized_valid_close = valid_df['Close']

	if process_class==0:
		pass

	if process_class==1:
		train_mean = train_df.mean()
		train_std = train_df.std()
		normalized_train_df = (normalized_train_df - train_mean)/train_std
		normalized_valid_df = (normalized_valid_df - train_mean)/train_std
		col=['s_Close','s_Date','s_High','s_Low','s_Open','s_Volume']
		normalized_train_df.columns=col
		normalized_valid_df.columns=col
		normalized_train_df[['Date', 'Open', 'High', 'Low', 'Close','Volume']]=train_df[['Date', 'Open', 'High', 'Low', 'Close','Volume']]
		normalized_valid_df[['Date', 'Open', 'High', 'Low', 'Close','Volume']]=valid_df[['Date', 'Open', 'High', 'Low', 'Close','Volume']]
		normalized_train_df=normalized_train_df.drop(['s_Date'],axis=1)
		normalized_valid_df=normalized_valid_df.drop(['s_Date'],axis=1)
	return normalized_train_df,normalized_valid_df