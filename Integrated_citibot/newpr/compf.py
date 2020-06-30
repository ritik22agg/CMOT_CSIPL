import pickle
import plotly.express as px
import plotly.io as pio
from datetime import date
import datetime
from dateutil.relativedelta import relativedelta
from model import model_call , predict_top_clients
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
import numpy as np
from helper_functions import give_last_date, take_fields, give_dates, give_clients_and_entities
from forms import OutputForm
import csv
from successfulTransactionCompare import comparareClientsTransaction


def plot_com(client1,client2,legal1,legal2,from_d):
	start = from_d

	df1 = model_call(str(client1),str(legal1))
	model = pickle.load(open('model.pkl', 'rb'))
	lastdate_1=give_last_date(client1, legal1)
	pred1=model.forecast(6)
	show_predict1=np.array(df1[str(datetime.datetime.strptime(str(lastdate_1),'%Y%m%d').date())])
	show_predict1=np.append(show_predict1, pred1)


	df2 = model_call(str(client2),str(legal2))
	model = pickle.load(open('model.pkl', 'rb'))
	lastdate_2=give_last_date(client2, legal2)
	pred2=model.forecast(6)
	print(lastdate_2)
	print('.............................................................')
	show_predict2=np.array(df2[str(datetime.datetime.strptime(str(lastdate_2),'%Y%m%d').date())])
	show_predict2=np.append(show_predict2, pred2)

	print(type(show_predict2))
	#fig = make_subplots(rows=1, cols=2)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x =df1[start:].index,y=df1[start:],mode='lines',name='Recorded trend 1'))
	fig.add_trace(go.Scatter(x=give_dates(lastdate_1),y=show_predict1,mode='lines',name='Predicted trend 1',line=dict(width=4, dash='dot')))
	fig.add_trace(go.Scatter(x = df2[start:].index, y=df2[start:], mode='lines', name='Recorded Trend 2'))
	fig.add_trace(go.Scatter(x=give_dates(lastdate_2), y=show_predict2, mode='lines', name='Predicted trend 2', line=dict(width=4, dash='dot')))
	fig.update_yaxes(title_text="Paid Amount")
	fig.update_xaxes(title_text='Dates')

	pio.write_html(fig, file='templates/output.html', auto_open=False)





def plot_pred(cname,lename,from_d):
	df = model_call(str(cname),str(lename))
	model = pickle.load(open('model.pkl', 'rb'))
	lastdate_=give_last_date(cname, lename)
	pred=model.forecast(6)
	start = from_d
	show_predict=np.array(df[str(datetime.datetime.strptime(str(lastdate_),'%Y%m%d').date())])
	show_predict=np.append(show_predict, pred)



	fig = go.Figure()
	fig.add_trace(go.Scatter(x = df[start:].index, y=df[start:], mode='lines', name='Recorded'))
	fig.add_trace(go.Scatter(x=give_dates(lastdate_), y=show_predict, mode='lines', name='Predicted',line=dict(width=4, dash='dot')))
	slope = pred[-1] - df.iloc[-1]
	final_verdict=''
	if slope>0:
		final_verdict="Beneficial to work with this client"
	else :
		final_verdict="Not beneficial to work with this client"
	fig.update_layout(title_text=final_verdict)
	fig.update_yaxes(title_text="Paid Amount")
	fig.update_xaxes(title_text='Dates')
	pio.write_html(fig, file='templates/predict.html', auto_open=False)
