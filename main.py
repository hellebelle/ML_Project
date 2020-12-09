import pandas as pd 
import numpy as np 
import matplotlib as pyplot
from matplotlib import pyplot
import datetime as dt
from sklearn.preprocessing import Normalizer

#load dataset
df = pd.read_csv("./dataset.csv",comment='#',header=None,parse_dates=[0])
dates = df.iloc[:,0]
dates = np.array(dates) + '01' #Concatenate with fixed day for all dates

#Convert the time in first column from type string to type date.
d= np.array([dt.datetime.strptime(date,"%Y%m%d").date() for date in dates],dtype='datetime64')

#Check for stationarity of multivariate time series (Visual Test)
#Visualise the trends in data
#Specify columns to plot
def dataset_visualisation() :
    groups=[1,2,3,4,5]
    groups_name = ["average_house_prices", "mortgage_interest_rate", "consumer_price_index", "yearly_GDP", "household_income"]
    i=1 
    #plot each column
    pyplot.figure()
    for group in groups :
        pyplot.subplot(len(groups),1,i)
        pyplot.plot(df.iloc[:,group])
        pyplot.title(groups_name[i-1],y=0.5,loc='right')
        i += 1
    pyplot.show()

#Standard in time-series to use variable depicting date-time information as index
df.iloc[:,0] = d
#Use 'data' from this point forward
data = df.drop([0],axis=1)
data.index = df.iloc[:,0]

# #Indexing techniques
# #1. Specific the index as a string constant:
#     data['2005-01']
# #2. Specify the entire range:
#     data['2005-01':'2005-05']
# #3. Specify values for a whole year:
#     data['2005']

#TODO : Check for stationarity of multivariate time series (Statistical Test-Dickie Fuller) (Helen)
#https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/#:~:text=A%20Multivariate%20time%20series%20has,used%20for%20forecasting%20future%20values.&text=In%20this%20case%2C%20there%20are,considered%20to%20optimally%20predict%20temperature.
#https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries, i):
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()

    #Plot rolling statistics:
    orig = pyplot.plot(timeseries, color='blue',label='Original')
    mean = pyplot.plot(rolmean, color='red', label='Rolling Mean')
    std = pyplot.plot(rolstd, color='black', label = 'Rolling Std')
    pyplot.legend(loc='best')
    pyplot.title('Rolling Mean & Standard Deviationat for Column = {}:'.format(i))
    pyplot.show()

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test for Column at {}:'.format(i))
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#check stationarity
# for i in range(1,6):
#     test_stationarity(data[i],i)

#From testing of stationarity concluded that data[2:] are non-stationary
def simple_trend_reduction(ts,i):
    
    #Take average of ‘k’ consecutive values depending on the frequency of time series. 
    #Here we can take the average over the past 1 year, i.e. last 12 values
    ts_log = np.log(ts)
    #moving_avg = ts_log.rolling(12).mean()
    # pyplot.plot(ts_log)
    # pyplot.plot(moving_avg, color='red')
    # pyplot.show()

    # #Subtract rolling mean from original series
    #ts_log_moving_avg_diff = ts_log - moving_avg
    # print(ts_log_moving_avg_diff.head(12))

    # #Check for stationarity
    #ts_log_moving_avg_diff.dropna(inplace=True)
    #test_stationarity(ts_log_moving_avg_diff,i)

    #------Another Way which I think should be used------#
    #We take a 'weighted moving average' where more recent values are given a higher weight
    #Exponentially Weighted Moving Average(EWMA) - Assigning weights 
    expwighted_avg = ts_log.ewm(halflife=12).mean()
    # pyplot.plot(ts_log)
    # pyplot.plot(expwighted_avg, color='red')

    ts_log_ewma_diff = ts_log - expwighted_avg
    # test_stationarity(ts_log_ewma_diff, i)
    return ts_log_ewma_diff

for i in range(2,6):
    data[i] = simple_trend_reduction(data[i],i)

#Simple trend reduction techniques discussed before don’t work in all cases, particularly the ones with high seasonality. 
#For data[2] we need to try remove trend and seasonality
def differencing(ts,i):
    #We take the difference of the observation at a particular instant with that at the previous instant.
    ts_diff = ts - ts.shift(1)
    # plt.plot(ts_log_diff)
    ts_diff.dropna(inplace=True)
    test_stationarity((ts_diff),i)

    return ts_diff

data[2] = differencing(data[2],2)
print(data)

def normalizeDataframe(dataFrame):
    df_num = dataFrame.select_dtypes(include=[np.number])
    df_norm = (df_num- df_num.min()) / (df_num.max() - df_num.min())
    dataFrame[df_norm.columns] = df_norm
    

def crossValidationTimeSeries():
    # All info from here: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
    
    ####### Method 1 #######
    
    # Start with small subset of the data
    # Forecast for later data points and then check accuracy of the forecasted data points
    # Use the same forecasted points as part of te next training dataset and forecast subsequent points again.
    
    # EXAMPLE: 5 observations  in cross validation set and want 4 fold cross validation
    # Let dataset be [1,2,3,4,5]
    # Need to create 4 pairs of training/test sets that follow these two rules
    # 1. Every test set contains unique observations
    # 2. Observations from training set occur before their corresponding test set
    # We get the following pairs of training/test sets:
    # Training: [1] Test: [2]
    # Training: [1,2] Test: [3]
    # Training: [1,2,3] Test: [4]
    # Training: [1,2,3,4] Test: [5]
    # Compute average of accuracies of the 4 test fold
    
    ######## Sample code for METHOD 1 ########
    # import numpy as np
    # from sklearn.model_selection import TimeSeriesSplit
    # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    # y = np.array([1, 2, 3, 4, 5, 6])
    # tscv = TimeSeriesSplit()
    # print(tscv)
    # TimeSeriesSplit(max_train_size=None, n_splits=3)
    # for train_index, test_index in tscv.split(X):
    # print(“TRAIN:”, train_index, “TEST:”, test_index) X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    
    # This gives the following output:
    # TRAIN: [0 1 2] TEST: [3]
    # TRAIN: [0 1 2 3] TEST: [4]
    # TRAIN: [0 1 2 3 4] TEST: [5]
    # Now train several models on each of these and average the accruacies for each model then pick the best one
	pass
    
    
#TODO : Drop GDP and Consumer Price Index (or whichever we should after checking for stationarity) (Helen)

#TODO : Normalise data (0 to 1) (Cian)

#TODO : 5-fold cross-validation (Cian)
#Refer to : https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4

#TODO : Cross validation for Model 1 (VAR) to select alpha value

#TODO : Cross validation for Model 2 (ARIMAX) to select weight value 

#TODO : Model 1(VAR) Function

    
#TODO : Model 2(ARIMAX) Function


#TODO : Both model's + baseline model Prediction Error(MSE) + Errorbar Function

