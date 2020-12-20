import pandas as pd 
import numpy as np 
import matplotlib as pyplot
from matplotlib import pyplot
import datetime as dt
from sklearn.preprocessing import Normalizer

#load dataset
df = pd.read_csv("./dataset.csv",parse_dates=[0])
dates = df['Month']
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
#Use 'data' from this point forward
data = df.drop(['Month'],axis=1)
data.index = d
# #Indexing techniques
# #1. Specific the index as a string constant:
#     data['2005-01']
# #2. Specify the entire range:
#     data['2005-01':'2005-05']
# #3. Specify values for a whole year:
#     data['2005']

#Check for stationarity of multivariate time series (Statistical Test-Dickie Fuller) (Helen)
#https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/#:~:text=A%20Multivariate%20time%20series%20has,used%20for%20forecasting%20future%20values.&text=In%20this%20case%2C%20there%20are,considered%20to%20optimally%20predict%20temperature.
#https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/

from statsmodels.tsa.stattools import adfuller
def stationarity(timeseries,name):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test for :' + name)
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#Check for stationarity for all features
def stationarity_test() :
    stationarity(data['Average Housing Price'],"Average Housing Price")   #Result : Stationary
    stationarity(data['Mortgage Interest Rates'],"Mortgage Interest Rates") #Result : Non-stationary
    stationarity(data['Consumer Price Index'],"Consumer Price Index") #Result : Non-stationary
    stationarity(data['Yearly GDP Per Capita'],"Yearly GDP Per Capita") #Result: Non-stationary
    stationarity(data['Net Yearly Household Income'],"Net Yearly Household Income") #Result : Non-stationary

#Differencing method on non-stationary data to convert them to stationary
def differencing() :
    data['diffMIR'] = data['Mortgage Interest Rates']-data['Mortgage Interest Rates'].shift(1)
    data['diffCPI'] = data['Consumer Price Index'] - data['Consumer Price Index'].shift(1)
    data['diffGDP'] = data['Yearly GDP Per Capita']-data['Yearly GDP Per Capita'].shift(1)
    data['diffYHI'] = data['Net Yearly Household Income']-data['Net Yearly Household Income'].shift(1)

#Check for stationarity after differencing
def diff_stationarity_test() :
    differencing()
    stationarity(data['diffMIR'].dropna(),"Differenced Mortgage Interest Rates")
    stationarity(data['diffCPI'].dropna(),"Differenced Consumer Price Index")
    stationarity(data['diffGDP'].dropna(),"Differenced Yearly GDP Per Capita") #Result after first differencing is data remains non-stationary
    stationarity(data['diffYHI'].dropna(),"Differenced Net Yearly Household Income") #Result after first differencing is data remains non-stationary
    
diff_stationarity_test()

#Second round of differencing
def second_differencing() :
    data['Second_diffMIR'] = data['diffMIR']-data['diffMIR'].shift(1)
    data['Second_diffCPI'] = data['diffCPI']-data['diffCPI'].shift(1)
    data['Second_diffGDP'] = data['diffGDP']-data['diffGDP'].shift(1)
    data['Second_diffYHI'] = data['diffYHI']-data['diffYHI'].shift(1)

def second_diff_stationarity_test() :
    second_differencing()
    stationarity(data['Second_diffMIR'].dropna(),"Second Differenced Mortgage Interest Rates")
    stationarity(data['Second_diffCPI'].dropna(),"Second Differenced Consumer Price Index")
    stationarity(data['Second_diffGDP'].dropna(),"Second Differenced Yearly GDP Per Capita")
    stationarity(data['Second_diffYHI'].dropna(),"Second Differenced Net Yearly Household Income")

second_diff_stationarity_test()

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
    

#TODO : Cross validation for Model 1 (VAR) to select alpha value

#TODO : Cross validation for Model 2 (ARIMAX) to select weight value 

#TODO : Model 1(VAR) Function

    
#TODO : Model 2(ARIMAX) Function


#TODO : Both model's + baseline model Prediction Error(MSE) + Errorbar Function

