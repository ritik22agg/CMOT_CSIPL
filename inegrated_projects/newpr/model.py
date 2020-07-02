import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import numpy as np 
import pandas as pd 
from model_for_percentage import get_percentage

def model_call(client_name, le_name):
    df = pd.read_csv(r'./processed_data.csv').set_index('Payment Date')
    df = df[(df['Client Name']==client_name)&(df['Legal Entity']==le_name)]
    print(df)

    train = df.iloc[:-10, :]
    test = df.iloc[-10:, :]
    pred = test.copy()
    model = ExponentialSmoothing(np.asarray(train['Paid Amount']), trend="add", seasonal="add", seasonal_periods=6)
    model2 = ExponentialSmoothing(np.asarray(train['Paid Amount']), trend="add", seasonal="add", seasonal_periods=6, damped=True)
    fit = model.fit()
    pred = fit.forecast(len(test))
    fit2 = model2.fit()
    pred2 = fit2.forecast(len(test))
    if fit.aic < fit2.aic :
    	pickle.dump(fit, open('model.pkl','wb'))
    else :
    	pickle.dump(fit2, open('model.pkl','wb'))
    return df['Paid Amount']


def get_key(val, my_dict):

    for key, value in my_dict.items():
        if val == value:
            return key 

    return "key doesn't exist"

def logic(index):
    if index % 3 == 0:
       return True
    return False

def predict_top_clients(num,allClients, forecast_num=6):
    
    df = pd.read_csv(r'./processed_data.csv', parse_dates=['Payment Date'], index_col='Payment Date')

    model_dict = {}

    for client in df['Client Name'].unique():
        model1 = ExponentialSmoothing(np.asarray(df[df['Client Name']==client]['Paid Amount']), trend='add', seasonal='add', seasonal_periods=6, damped=True)
        hw_model1 = model1.fit()
        model2 = ExponentialSmoothing(np.asarray(df[df['Client Name']==client]['Paid Amount']), trend='add', seasonal='add', seasonal_periods=6)
        hw_model2 = model2.fit()
        model_dict[client] = hw_model1 if hw_model1.aic < hw_model2.aic else hw_model2

    predicted_amounts = {}
    predicted_range = {}

    for client, model in model_dict.items():
        pred = model.forecast(forecast_num)
        predicted_range[client] = pred
        predicted_amounts[client] = pred.mean()
    percentage=get_percentage(allClients)
    #percentage = {'sbi': 100.0, 'hdfcbankltdmumbaiheadoffice': 97.58575356484164, 'jpmchase': 96.55315330404157, 'bankofbarodamumbaiheadoffice': 96.1396145301158, 'hsbc': 88.85818430732402, 'barclays': 97.40791379423159}

    final_result={}
    for client in predicted_range.keys():
        final_result[client]=[round(predicted_amounts[client], 3)]
        final_result[client].append(round(percentage[client], 3))
        for x in predicted_range[client]:
            final_result[client].append(round(x, 3))

    listofTuples = sorted(final_result.items() ,reverse=True, key=lambda x: (x[1][0], x[1][1]))
    final_result_n={}
    for elem in listofTuples:
        final_result_n[elem[0]]=elem[1]
    print(final_result_n)
    print(type(final_result_n))
    print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')

    return final_result_n
