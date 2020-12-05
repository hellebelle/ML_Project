import pandas as pd 
import numpy as np 
import matplotlib as pyplot
from matplotlib import pyplot
import datetime as dt

#load dataset
df = pd.read_csv("./dataset.csv",comment='#',header=None,parse_dates=[0])
dates = df.iloc[:,0]
dates = np.array(dates) + '01' #Concatenate with fixed day for all dates

#Convert the time in first column from type string to type date.
d= np.array([dt.datetime.strptime(date,"%Y%m%d").date() for date in dates],dtype='datetime64')

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

#TODO : K-fold Cross validation to select number of folds
#Refer to : https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4


#TODO : Cross validation for Model 1 (Ridge) to select alpha value

#TODO : Cross validation for Model 2 (kNeighborsRegressor) to select weight value 

#TODO : Model 1(Ridge) Function

#TODO : Model 2(kNeighborsRegressor) Function

#TODO : Prediction using short term trend (+Compare prediction error between the two models)

#TODO : Prediction using yearly seasonality

#TODO : Prediction using two-year seasonality

#TODO : Prediction using yearly, two-year and short term trend combined

#TODO : Both model's + baseline model Prediction Error(MSE) + Errorbar Function




