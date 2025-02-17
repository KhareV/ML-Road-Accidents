import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



def transform(dataframe):
    df=np.log(dataframe)
    df.insert(1, "t-1", 1) 
    df['t-1'] = df['t'].shift(1)
    return df

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	agg = pd.concat(cols, axis=1)
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]
 
# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	train = np.asarray(train)
	trainX, trainy = train[:, :-1], train[:, -1]
	#fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	yhat = model.predict([testX])
	return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	train, test = train_test_split(data, n_test)
	history = [x for x in train]
	expected=[]
	predicted=[]
	for i in range(len(test)):
		testX, testy = test[i, :-1], test[i, -1]
		yhat = random_forest_forecast(history, testX)
		predictions.append(yhat)
		history.append(test[i])
		print('>expected=%.1f, predicted=%.1f' % (np.exp(testy), np.exp(yhat)))
		expected.append(np.exp(testy))
		predicted.append(np.exp(yhat))
	error = mean_absolute_error(test[:, -1], predictions)
	return error, expected,predicted

def check_error(df,mae):
    print('Mean Absolute Error: %.3f' % mae)
    avg = df['t'].mean()
    percent_error = (mae/avg)*100
    print('Percent error: %0.3f' %percent_error)
    avg = df['t'].mean()
    percent_error = (mae/avg)*100
    print(percent_error)


def rearranging(d1,i):
		d1=d1.rename(columns={i:"t"})
		#print("d1",d1)
		d1['Year']=d1.index
		d1.reset_index()
		i='t'
		df=d1[["Year","t"]].set_index(["Year"])   
		return df              #original data of a state

def general_func(df):
		data_transform=transform(df)
		values = data_transform.values
		data1 = series_to_supervised(values)

		mae, y, yhat = walk_forward_validation(data1, 1)
		#check_error(df,mae)
		x = df.index.values[-1].split(" ")
		expected=pd.DataFrame(y,index=[x[0]+" "+str(int(x[1])+i) for i in range(1,2)])
		predicted=pd.DataFrame(yhat,index=[x[0]+" "+str(int(x[1])+i) for i in range(1,2)])
		return predicted


def findings(d1,i):
	df=rearranging(d1,i)
	print("df",df)
	#1st year prediction
	for j in range(6): 
		d1=general_func(df)
		d1=d1=d1.rename(columns={0:"t"})
		d1=rearranging(d1,i)
		df=pd.concat([df,d1])
	df=df.tail(6)
	df=df.transpose()
	index = pd.Index([i])
	df = df.set_index(index)
	print(df)
	return df


data=pd.read_excel("D:/College/Sem 4/DAVL/Project/Accident Dataset/2018-2021 data.xlsx")
col=data.columns.to_list()
cols=[]
#injured
for i in col:
    if 'Injured' in i:
	    cols.append(i)

injured=data[cols]
state_injured=pd.DataFrame()
for i in range(len(data)):
	d1=injured.iloc[i]
	d1.transpose()
	try:
		print(d1.to_frame())
		df=findings(d1.to_frame(),i)
		state_injured = pd.concat([state_injured,df])
	except:
		pass


print(state_injured.columns.tolist())
state=data['State/UT']
state_injured=state_injured.join(state)
state_injured=state_injured.set_index(state_injured['State/UT'])
del state_injured['State/UT']
print("final",state_injured)
state_injured.to_excel("statewise_injured.xlsx", index=True)


#killed
for i in col:
    if 'Killed' in i:
	    cols.append(i)

Killed=data[cols]
state_Killed=pd.DataFrame()
for i in range(len(data)):
	d1=Killed.iloc[i]
	d1.transpose()
	try:
		print(d1.to_frame())
		df=findings(d1.to_frame(),i)
		state_Killed = pd.concat([state_Killed,df])
	except:
		pass


print(state_Killed.columns.tolist())
state=data['State/UT']
state_Killed=state_Killed.join(state)
state_Killed=state_Killed.set_index(state_Killed['State/UT'])
del state_Killed['State/UT']
print("final",state_Killed)
state_Killed.to_excel("statewise_Killed.xlsx", index=True)


#accidents
for i in col:
    if 'Accidents' in i:
	    cols.append(i)

Accidents=data[cols]
state_Accidents=pd.DataFrame()
for i in range(len(data)):
	d1=Accidents.iloc[i]
	d1.transpose()
	try:
		print(d1.to_frame())
		df=findings(d1.to_frame(),i)
		state_Accidents = pd.concat([state_Accidents,df])
	except:
		pass


print(state_Accidents.columns.tolist())
state=data['State/UT']
state_Accidents=state_Accidents.join(state)
state_Accidents=state_Accidents.set_index(state_Accidents['State/UT'])
del state_Accidents['State/UT']
print("final",state_Accidents)
state_Accidents.to_excel("statewise_Accidents.xlsx", index=True)