from django.shortcuts import render, redirect
from django.utils.datastructures import MultiValueDictKeyError
import openpyxl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential

import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
import io
import urllib, base64
from tensorflow.keras import backend as K
import tensorflow as tf
import gc




def forecast(request):

    if "GET" == request.method:
        return render(request, 'forecast.html', {})
    else:
        excel_file = request.FILES["excel_file"]

        # reading the excel file
        wb = openpyxl.load_workbook(excel_file)
        worksheet = wb.active

        # converting worksheet to DataFrame
        df = pd.DataFrame(worksheet.values)
        month = list(df.drop(0, axis=0)[0])
        passeng = list(df.drop(0, axis=0)[1])
        Data = pd.DataFrame({'Month': month, 'Passengers': passeng})
        Data.Month = pd.to_datetime(Data.Month)

                
        n_input = 12
        n_features = 1

        # creating our model
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dropout(0.20))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Scale 
        Data = Data.set_index('Month')
        train = Data
        scaler = MinMaxScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        
        # Generating Time Series format
        generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

        # Fitting our model
        model.fit_generator(generator,epochs=90, verbose= 0)

        # Predection
        pred_list = []  
        batch = train[-n_input:].reshape((1, n_input, n_features))
        for i in range(n_input):
            pred_list.append(model.predict_on_batch(batch)[0])      
            batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
        add_dates = [Data.index[-1] + DateOffset(months=x) for x in range(13)]
        future_dates = pd.DataFrame(index=add_dates[1:], columns=Data.columns)
        data_predict = pd.DataFrame(scaler.inverse_transform(pred_list), index = future_dates[-12:].index, columns= ['Predictions'])
        
        # Preparing for html page
        data_predictN = data_predict.reset_index()
        data_predictN['index']=data_predictN['index'].apply(lambda x: x.strftime("%m/%Y"))        
        dataF = data_predictN.values
        all_data = [{'Month': dataF[i][0], 'Passenger': dataF[i][1]} for i in range(len(dataF))]
        
        # Plotting the prediction
        df_proj = pd.concat([Data,data_predict], axis=1)
        plt.plot(df_proj.index, df_proj['Passengers'])
        plt.plot(df_proj.index, df_proj['Predictions'], color='r')
        plt.legend(loc='best', fontsize='xx-large')
        plt.xticks(fontsize=18, color = "white")
        plt.yticks(fontsize=16, color = "white")
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        url = urllib.parse.quote(string)

        # Deleating our model to prevent data leakage
        del model
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()

        return render(request, 'forecast.html', {"all_data":all_data, 'plot_div': url})

      
        
