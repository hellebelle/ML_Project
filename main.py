import pandas as pd 
import numpy as np 
import matplotlib as pyplot
from matplotlib import pyplot
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import Normalizer
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

#load dataset
df = pd.read_csv("./dataset.csv",parse_dates=[0])
dates = df['Month']
dates = np.array(dates) + '01' #Concatenate with fixed day for all dates

#Convert the time in first column from type string to type date.
d= np.array([dt.datetime.strptime(date,"%Y%m%d").date() for date in dates],dtype='datetime64')
data = df.drop(['Month'],axis=1) # data = Used for VAR
df.drop(['Month'],axis=1) # df = Used for ARIMAX
df.index = d
data.index = d 

#Check for stationarity of multivariate time series (Visual Test)
#Visualise the trends in data
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

def baseLineFunction(arr):
    output = [0] * len(arr)
    output[0] = arr[0]
    for i in range(1,len(arr)):
        output[i] = arr[i-1]
    return output
#granger_causality_matrix()

#Check for stationarity of multivariate time series (Statistical Test-Dickie Fuller) (Helen)
from statsmodels.tsa.stattools import adfuller
def stationarity(timeseries,name):
    #Perform Dickey-Fuller test:
    #print('Results of Dickey-Fuller Test for :' + name)
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print(dfoutput)

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

#Making copy of train/test data
train_orig = data[0:-12].copy()
test_orig = data[-12:].copy()

def stationarize():
    # #First stationarity test shows that :
    # #Average Housing Price = stationary ; Mortgage Interest Rates = non-stationary ; Consumer Price Index = non-stationary ; Yearly GDP Per Capita = non-stationary ; Net Yearly Household Income = non-stationary
    stationarity_test()  
    differencing()
    # #Second stationarity test shows that :
    # #Average Housing Price = non-stationary ; Mortgage Interest Rates = stationary ; Consumer Price Index = stationary ; Yearly GDP Per Capita = non-stationary ; Net Yearly Household Income = non-stationary
    stationarity_test() 
    differencing()
    # #Second stationarity test shows that :
    # #Average Housing Price = stationary ; Mortgage Interest Rates = stationary ; Consumer Price Index = stationary ; Yearly GDP Per Capita = stationary ; Net Yearly Household Income = stationary

#Cross validation for Model 1 (VAR) to select alpha value
#The selected order(p) should be the order that gives the lowest ‘AIC’, ‘BIC’, ‘FPE’ and ‘HQIC’ scores.
def cross_validation_VAR(train):
    model_VAR = VAR(train)
    x = model_VAR.select_order(maxlags=24)
    # print(x.summary())


# inverting transformation
def invert_transformation(df_forecast,second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = train_orig.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (train_orig[col].iloc[-1]-train_orig[col].iloc[-2]) + df_fc[str(col)+'_pred'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = train_orig[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

#Model 1(VAR) Function
def build_var(data):
    stationarize()
    #train-test split
    data = data.dropna()
    
    nobs = 12
    train, test = data[0:-nobs], data[-nobs:]

    ##From cross validation determined that lag order of 24 is optimal value
    # cross_validation_VAR(train)
    
    #Train the VAR Model of lag-order 24
    model_VAR = VAR(train)
    model_fitted_VAR = model_VAR.fit(24)
    # print(model_fitted_VAR.summary())

    #Check for Serial Correlation of Residuals (Errors) using Durbin Watson Statistic
    from statsmodels.stats.stattools import durbin_watson
    out = durbin_watson(model_fitted_VAR.resid)

    for col, val in zip(data.columns, out):
        print(col, ':', round(val, 2))

    #Forecasting
    # Get the lag order
    lag_order = model_fitted_VAR.k_ar
    # print(lag_order)  #> should be 24

    # Input data for forecasting
    forecast_input = data.values[-lag_order:] #shape (24,5)
    # print(forecast_input) #array of values to forecast

    # forecast
    pred = model_fitted_VAR.forecast(y=forecast_input, steps=nobs)
    pred = (pd.DataFrame(pred, index=test.index, columns= data.columns + '_pred'))
    
    output = invert_transformation(pred, second_diff=True)
    print(output)
    # print(output.loc[:, ['Average Housing Price_forecast']])

    fig, axes = pyplot.subplots(nrows=int(len(data.columns)/2), ncols=2, dpi=150, figsize=(10,10))
    for i, (col,ax) in enumerate(zip(data.columns, axes.flatten())):
        output[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        test_orig[col][-nobs:].plot(legend=True, ax=ax)
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    pyplot.show()

    #Putting values into list for comparison
    VAR_results = output.loc[:, ['Average Housing Price_forecast']]
    new_indexes = np.arange(0, 12, 1).tolist()
    print(new_indexes)
    VAR_results.reindex(new_indexes)
    results = []
    for i in range(12):
        results.append(VAR_results['Average Housing Price_forecast'][i])
    print(results)
    return results

forecast_VAR = build_var(data) #Forecast of Average Houseing Price for 12 months

#Cross validation for Model 2 (ARIMAX) to select order of differencing(d), order of AR(p) and order of MA(q)
from statsmodels.tsa.arima_model import ARIMA
def build_arimax() :
    #Use autocorrelation and partial autocorrelation plots to help us select a range for the p,d,q hyperparameters.
    #First differencing
    diff = df['Average Housing Price'].diff()
    #To identify if the model requires any MA terms
    sm.graphics.tsa.plot_acf(diff.dropna(), lags=80)
    #To identify if the model requires any AR terms
    sm.graphics.tsa.plot_pacf(diff.dropna(), lags=80)
    #Second differencing
    diff2 = diff.diff()
    #To identify if the model requires any MA terms
    sm.graphics.tsa.plot_acf(diff2.dropna(), lags=80)
    #To identify if the model requires any AR terms
    sm.graphics.tsa.plot_pacf(diff2.dropna(), lags=80)
    #plt.show()

    #Split data to test and training
    Y = df['Average Housing Price']
    X = df.drop(['Average Housing Price'],axis=1)
    Ytrain = Y[:168] ; Ytest = Y[168:]
    Xtrain = X[:168] ; Xtest = X[168:]
    # #ARIMAX(1,1,1)
    # model = ARIMA(endog=Ytrain,exog=Xtrain[['Mortgage Interest Rates','Consumer Price Index','Yearly GDP Per Capita','Net Yearly Household Income']],order=[1,1,1])
    # fitted = model.fit()
    # print(fitted.summary())
    # #ARIMAX(1,1,13)
    # model = ARIMA(endog=Ytrain,exog=Xtrain[['Mortgage Interest Rates','Consumer Price Index','Yearly GDP Per Capita','Net Yearly Household Income']],order=[1,1,13])
    # fitted = model.fit()
    # print(fitted.summary())
    # #ARIMAX(1,2,1)
    # model = ARIMA(endog=Ytrain,exog=Xtrain[['Mortgage Interest Rates','Consumer Price Index','Yearly GDP Per Capita','Net Yearly Household Income']],order=[1,2,1])
    # fitted = model.fit()
    # print(fitted.summary())
    # #ARIMAX(2,1,1)
    # model = ARIMA(endog=Ytrain,exog=Xtrain[['Mortgage Interest Rates','Consumer Price Index','Yearly GDP Per Capita','Net Yearly Household Income']],order=[2,1,1])
    # fitted = model.fit()
    # print(fitted.summary())
    #ARIMAX(1,2,2)
    model = ARIMA(endog=Ytrain,exog=Xtrain[['Mortgage Interest Rates','Consumer Price Index','Yearly GDP Per Capita','Net Yearly Household Income']],order=[1,2,2])
    fitted = model.fit()
    print(fitted.summary())
    # #ARIMAX(2,2,1)
    # model = ARIMA(endog=Ytrain,exog=Xtrain[['Mortgage Interest Rates','Consumer Price Index','Yearly GDP Per Capita','Net Yearly Household Income']],order=[2,2,1])
    # fitted = model.fit()
    # print(fitted.summary())
                 
#build_arimax()

#TODO : Both model's + baseline model Prediction Error(MSE) + Errorbar Function
def models_performance(arr1,arr2,lengthOfArrays = 12) :
    # MAPE of all model
    actualValues = df["Average Housing Price"][-lengthOfArrays:]
    mabe1 = np.abs((actualValues - arr1))
    mabe2 = np.abs((actualValues - arr2))
    baseLinePredictions = baseLineFunction(actualValues)
    baseLineError = np.abs((actualValues - baseLinePredictions))
    
    return baseLineError,mabe1, mabe2

