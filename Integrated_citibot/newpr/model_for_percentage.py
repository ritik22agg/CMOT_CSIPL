import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import numpy as np
from datetime import date, timedelta
import calendar 


def get_percentage(allClients) :
	df = pd.read_csv(r'./processed_data.csv', parse_dates = ['Payment Date'], index_col = 'Payment Date')
	start_date = date(2018,12,1)
	num_months = (date.today().year - 2019)*12 + date.today().month 
	data = []
	for i in range(num_months):
		for client in df['Client Name'].unique():
			days_in_month = calendar.monthrange(start_date.year, start_date.month)[1]
			date_ = start_date + timedelta(days=days_in_month)
			if date_ < date(2020, 6, 30):
				df1 = df[str(date_)[:-3]]
			else:
				break
			count = len(df1[(df1['Client Name']==client)&(df1['Payment Status']=='resolved')])
			total = len(df1[df1['Client Name']==client])
			row = [date_, client, count, total]
			data.append(row)
		start_date = date_
	df2 = pd.DataFrame(data,columns = ['Payment Date','Client Name', 'Resolved', 'Total']).set_index('Payment Date')
	df=df2
	total_resolved={}
	total_transaction={}
	predicted_resolved={}
	predicted_total={}
	percentage = {}
	for clients in allClients :
		total_resolved[clients]=list()
		total_transaction[clients]=list()
	for i in range(len(df)):
		total_resolved[df.iloc[i,0]].append(df.iloc[i,1])
		total_transaction[df.iloc[i,0]].append(df.iloc[i,2])
	for client in allClients :
		model1 = ExponentialSmoothing(np.asarray(total_resolved[client]), trend='add', seasonal='add', seasonal_periods=6, damped=True)
		hw_model1 = model1.fit()
		model2 = ExponentialSmoothing(np.asarray(total_resolved[client]), trend='add', seasonal='add', seasonal_periods=6)
		hw_model2 = model2.fit()
		model = hw_model1 if hw_model1.aic < hw_model2.aic else hw_model2
		predicted_resolved[client]=model.forecast(6)

		model1 = ExponentialSmoothing(np.asarray(total_transaction[client]), trend='add', seasonal='add', seasonal_periods=6, damped=True)
		hw_model1 = model1.fit()
		model2 = ExponentialSmoothing(np.asarray(total_transaction[client]), trend='add', seasonal='add', seasonal_periods=6)
		hw_model2 = model2.fit()
		model = hw_model1 if hw_model1.aic < hw_model2.aic else hw_model2
		predicted_total[client]=model.forecast(6)

		num=0
		deno=0
		for i in range(6):
			num=abs(predicted_resolved[client][i])+num
			deno=abs(predicted_total[client][i])+deno
		if deno > 0:
			percentage[client]=(num*100)/deno
		else :
			percentage[client]=0

	return percentage 

