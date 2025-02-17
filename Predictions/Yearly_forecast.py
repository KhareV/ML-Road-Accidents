import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


data=pd.read_excel("D:/College/Sem 4/DAVL/Project/Accident Dataset/1970-2021 data.xlsx")

#data preprocessing
data.dropna(inplace=True)

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
		#print('>expected=%.1f, predicted=%.1f' % (np.exp(testy), np.exp(yhat)))
		expected.append(np.exp(testy))
		predicted.append(np.exp(yhat))
	error = mean_absolute_error(test[:, -1], predictions)
	return error, expected,predicted

def check_error(df,mae):
    print('Mean Absolute Error: %.3f' % mae)
    avg = df['t'].mean()
    percent_error = (mae/avg)*100
    print('Percent error: ',percent_error)



col=data.columns.to_list()
col.remove('Year')
pre_list=[]
predictions=[]
totals=['Total Number of Road Accidents (in numbers)','Total Number of Persons Killed (in numbers)','Total Number of Persons Injured (in numbers)']
for i in col:
	if i in totals:
		df=data[["Year",i]].set_index(["Year"])
		df=df.rename(columns={i:"t"})
		data_transform=transform(df)
		values = data_transform.values
		data1 = series_to_supervised(values)
		mae, y, yhat = walk_forward_validation(data1, 10)
		check_error(df,mae)
		expected=pd.DataFrame(y,index=[2021+i for i in range(1,11)])
		predicted=pd.DataFrame(yhat,index=[2021+i for i in range(1,11)])
		pre_list.append(predicted.reset_index().values.tolist())
		predictions.append(predicted)
		# plot expected vs predicted
		"""df.plot()
		plt.plot(expected, label='Expected',marker="o")
		plt.plot(predicted, label='Predicted',marker="o")
		plt.legend()
		plt.show()"""

predictions_df = pd.concat(predictions, axis=1)
predictions_df.columns = totals
predictions_df.to_excel("predicted_values.xlsx", index=True)


#plot
"""df1=data[['Year','Total Number of Road Accidents (in numbers)']]
    df2=data1[['Year','Total Number of Road Accidents (in numbers)']]
    merged_df = pd.merge(df1, df2)
    df = merged_df.set_index('Year')
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Total Number of Road Accidents (in numbers)'])
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Number of Road Accidents (in numbers)')
    st.pyplot(fig)"""



