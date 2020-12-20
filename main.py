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

from statsmodels.tsa.stattools import grangercausalitytests
def granger_causality_matrix() :
    test = 'ssr_chi2test'; maxlag=12; verbose=False 
    variables = ["Average Housing Price","Mortgage Interest Rates","Consumer Price Index","Yearly GDP Per Capita","Net Yearly Household Income"]
    matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in matrix.columns :
        for r in matrix.index :
            test_result = grangercausalitytests(data[[r,c]],maxlag=maxlag,verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            matrix.loc[r, c] = min_p_value
    matrix.columns = [var + '_x' for var in variables]
    matrix.index = [var + '_y' for var in variables]
    print(matrix.loc[['Average Housing Price_y']])
    
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
    stationarity(data['Average Housing Price'].dropna(),"Average Housing Price")   
    stationarity(data['Mortgage Interest Rates'].dropna(),"Mortgage Interest Rates")
    stationarity(data['Consumer Price Index'].dropna(),"Consumer Price Index") 
    stationarity(data['Yearly GDP Per Capita'].dropna(),"Yearly GDP Per Capita") 
    stationarity(data['Net Yearly Household Income'].dropna(),"Net Yearly Household Income") 

#Differencing method on non-stationary data to convert them to stationary
def differencing() :
    data['Average Housing Price'] = data['Average Housing Price'].diff()
    data['Mortgage Interest Rates'] = data['Mortgage Interest Rates'].diff()
    data['Consumer Price Index'] = data['Consumer Price Index'].diff()
    data['Yearly GDP Per Capita'] = data['Yearly GDP Per Capita'].diff()
    data['Net Yearly Household Income'] = data['Net Yearly Household Income'].diff()

#First stationarity test shows that :
#Average Housing Price = stationary ; Mortgage Interest Rates = non-stationary ; Consumer Price Index = non-stationary ; Yearly GDP Per Capita = non-stationary ; Net Yearly Household Income = non-stationary
stationarity_test()  
differencing()
#Second stationarity test shows that :
#Average Housing Price = non-stationary ; Mortgage Interest Rates = stationary ; Consumer Price Index = stationary ; Yearly GDP Per Capita = non-stationary ; Net Yearly Household Income = non-stationary
stationarity_test() 
differencing()
#Second stationarity test shows that :
#Average Housing Price = stationary ; Mortgage Interest Rates = stationary ; Consumer Price Index = stationary ; Yearly GDP Per Capita = stationary ; Net Yearly Household Income = stationary
stationarity_test()

print(data.head())

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
    
#granger_causality_matrix()


#TODO : Cross validation for Model 1 (VAR) to select alpha value


#TODO : Cross validation for Model 2 (ARIMAX) to select weight value 


#TODO : Model 1(VAR) Function

    
#TODO : Model 2(ARIMAX) Function


#TODO : Both model's + baseline model Prediction Error(MSE) + Errorbar Function


