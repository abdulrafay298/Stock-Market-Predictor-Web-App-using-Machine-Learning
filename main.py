import math, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import datetime as dt
import yfinance as yf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from alpha_vantage.timeseries import TimeSeries
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import warnings 
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_data(input):
    end = datetime.now()
    start = datetime(end.year-2, end.month, end.day)#using previous 2 years of data
    
    data = yf.download(input, start=start, end=end)
    df = pd.DataFrame(data=data)
    
    
    
    return


#****LSTM Analysis*****
def LSTM_analysis(df):
    data_train = df.iloc[0:int(len(df)*0.8),:] #using 80% of data for training
    data_test = df.iloc[int(len(df)*0.8):,:] #using 20%  of data for testing
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_train_scale = scaler.fit_transform(df.iloc[:,4:5].values)
    x_train=[]
    y_train=[]
    for i in range(7, len(data_train_scale)):       #using previous 7 days for data to predict the next one
        x_train.append(data_train_scale[i-7:i,0])
        y_train.append(data_train_scale[i,0])
    #convert to np arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
 

    #X_train[-1, 1:] selects the last row in X_train (i.e., the last 7-day sequence used for training).
    #1: slices the array from the second element to the last (i.e., the last 6 elements of the 7-day sequence).
    x_forecast = np.array(x_train[-1,1:])

    #This appends the last value from y_train to X_forecast. y_train[-1] 
    #is the actual stock price corresponding to the last day in the training data.
    x_forecast = np.append(x_forecast,y_train[-1])

    #X_train.shape[0] is the number of samples (rows).
    #X_train.shape[1] is the number of timesteps (columns).
    #1 is the number of features, which is added as the third dimension for compatibility with LSTM models.
    #This converts X_train from a 2D array of shape (samples, timesteps) to a 3D array of shape (samples, timesteps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

    #1 is the number of samples, as X_forecast is a single sequence.
    #X_forecast.shape[0] is the number of timesteps in X_forecast, which is 7.
    #1 is the number of features.
    #This converts X_forecast from a 1D array of shape (timesteps,) to a 3D array of shape (samples, timesteps, features).
    x_forecast = np.reshape(x_forecast, (1,x_forecast.shape[0],1))

    from keras.layers import Dense, Dropout, LSTM
    from keras.models import Sequential

    model = Sequential()
    model.add(LSTM(units=50, return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(Dropout(0.1))

    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(units=50))
    model.add(Dropout(0.1))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train,epochs=30,batch_size=32)

    real_price = data_test.iloc[:,4:5].values

    dataset = pd.concat((data_train['Close'],data_test['Close']), axis=0)
    testing_set = dataset[len(dataset)-len(data_test)-7:].values
    testing_set = testing_set.reshape(-1,1)
    testing_set=scaler.transform(testing_set)

    x_test = []
    for i in range(7,len(testing_set)):
        x_test.append(testing_set[i-7:i,0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    fig = plt.figure(figsize=(7,5))
    plt.plot(real_price, label="Real price")
    plt.plot(predicted_price, label='Predicted price')
    plt.legend()
    plt.show()
    plt.savefig('images/LSTM.png')
    plt.close(fig)

    lstm_error = math.sqrt(mean_squared_error(real_price,predicted_price))
    forecast_price = model.predict(x_forecast)
    forecast_price = scaler.inverse_transform(forecast_price)
    lstm_forecast = forecast_price[0,0]
    print("Tomorrow's " , input, "Closing price predicted by LSMT model is ", lstm_forecast )
    print("LSTM RMSE:",lstm_error)

    return lstm_error,lstm_forecast

    



input = 'AAPL'


get_data(input)

    

