import quandl
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def predictFutureStockPrices(ticker, days):
    quandl.ApiConfig.api_key = 'ekUM3Ut18s6GyHRFqxV9'

    df = quandl.get("WIKI/" + ticker.upper(), paginate = False)
    df = df[['Adj. Close']]
    pred_days = days
    df['Prediction'] = df['Adj. Close'].shift(-pred_days)

    X = np.array(df.drop(['Prediction'], 1))
    X = preprocessing.scale(X)
    X_pred = X[-pred_days:]
    X = X[:-pred_days]

    Y = np.array(df['Prediction'])
    Y = Y[:-pred_days]  

    try:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    except:
        print(X.shape)
        print(Y.shape)

    clf = LinearRegression()
    clf.fit(X_train, Y_train)
    confidence = clf.score(X_test, Y_test)

    forecast = clf.predict(X_pred)
    
    end_date = datetime.date(2018,3,28) + datetime.timedelta(days = pred_days - 1) 
    dates = pd.date_range(start = '2018-03-28', end = end_date.isofformat())
    return dates, forecast

