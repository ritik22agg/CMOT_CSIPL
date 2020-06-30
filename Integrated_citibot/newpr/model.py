import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import numpy as np
import pandas as pd

def model_call(client_name, le_name):
    df = pd.read_csv(r'processed_data.csv').set_index('Payment Date')
    df = df[(df['Client Name']==client_name)&(df['Legal Entity']==le_name)]
    train = df.iloc[:-10, :]
    test = df.iloc[-10:, :]
    pred = test.copy()
    model = ExponentialSmoothing(np.asarray(train['Paid Amount']), trend="add", seasonal="add", seasonal_periods=12)
    model2 = ExponentialSmoothing(np.asarray(train['Paid Amount']), trend="add", seasonal="add", seasonal_periods=12, damped=True)
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

def predict_top_clients(num, forecast_num=6):

    df = pd.read_csv(r'processed_data.csv', parse_dates=['Payment Date'], index_col='Payment Date')

    model_dict = {}

    for client in df['Client Name'].unique():
        model1 = ExponentialSmoothing(np.asarray(df[df['Client Name']==client]['Paid Amount']), trend='add', seasonal='add', seasonal_periods=12, damped=True)
        hw_model1 = model1.fit()
        model2 = ExponentialSmoothing(np.asarray(df[df['Client Name']==client]['Paid Amount']), trend='add', seasonal='add', seasonal_periods=12)
        hw_model2 = model2.fit()
        model_dict[client] = hw_model1 if hw_model1.aic < hw_model2.aic else hw_model2

    predicted_amounts = {}
    predicted_range = {}

    for client, model in model_dict.items():
        pred = model.forecast(forecast_num)
        predicted_range[client] = pred
        predicted_amounts[client] = pred.mean()

    values = list(predicted_amounts.values())
    values.sort()

    result = {}

    for i in range(1,num + 1):
        result[get_key(values[-1*i],predicted_amounts)] = values[-1*i]

    final_result = {}
    for client in result :
        temp_result = list()
        temp_result.append(round(result[client], 3))
        for x in predicted_range[client]:
            temp_result.append(round(x, 3))
        final_result[client] = temp_result

    print(result)
    print(final_result)
    print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')

    return final_result
