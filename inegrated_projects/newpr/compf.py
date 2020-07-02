import pickle
import plotly.express as px
import plotly.io as pio
from datetime import date
import datetime
from model import model_call , predict_top_clients
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
import numpy as np
from helper_functions import give_last_date, take_fields, give_dates, give_clients_and_entities
from forms import OutputForm
import csv


def plot_com(client1,client2,legal1,legal2,from_d):
	start = from_d


	df1 = model_call(str(client1),str(legal1))
	model = pickle.load(open('model.pkl', 'rb'))
	lastdate_1=give_last_date(client1, legal1)
	pred1=model.forecast(6)
	pred1_mean=round(pred1.mean(), 3)
	show_predict1=np.array(df1[str(datetime.datetime.strptime(str(lastdate_1),'%Y%m%d').date())])
	show_predict1=np.append(show_predict1, pred1)


	df2 = model_call(str(client2),str(legal2))
	model = pickle.load(open('model.pkl', 'rb'))
	lastdate_2=give_last_date(client2, legal2)
	pred2=model.forecast(6)
	pred2_mean=round(pred2.mean(), 3)
	print(lastdate_2)
	print('.............................................................')
	show_predict2=np.array(df2[str(datetime.datetime.strptime(str(lastdate_2),'%Y%m%d').date())])
	show_predict2=np.append(show_predict2, pred2)
	
	print(type(show_predict2))
	#fig = make_subplots(rows=1, cols=2)
	table_col=['Clients', 'Legal Entity', 'Mean of Predicted Paid Amount (USD)']
	table_title=""
	if pred1_mean > pred2_mean :
		table_row=[[client1, client2], [legal1, legal2], [pred1_mean, pred2_mean]]
		table_title="{m} and {n} are expected to do better<br>business based on predicted mean amount".format(m=client1,n=legal1)
	else :
		table_title="{m} and {n} are expected to do better<br>business based on predicted mean amount ".format(m=client2, n=legal2)
		table_row=[[client2, client1], [legal2, legal1], [pred2_mean, pred1_mean]]

	fig = make_subplots(rows=2, cols=1,  vertical_spacing=0.03,specs=[ [{"type": "table"}],[{"type": "scatter"}] ] )
	fig.add_trace(go.Table(header=dict(values=table_col,font=dict(size=10),align="left"), cells=dict(values=table_row,  height=40,align="left")), row=1, col=1)
	fig.add_trace(go.Scatter(x =df1[start:].index,y=df1[start:],mode='lines',name='Recorded trend 1'), row=2, col=1)
	fig.add_trace(go.Scatter(x=give_dates(lastdate_1),y=show_predict1,mode='lines',name='Predicted trend 1',line=dict(width=4, dash='dot')), row=2, col=1)
	fig.add_trace(go.Scatter(x = df2[start:].index, y=df2[start:], mode='lines', name='Recorded Trend 2'),  row=2, col=1)
	fig.add_trace(go.Scatter(x=give_dates(lastdate_2), y=show_predict2, mode='lines', name='Predicted trend 2', line=dict(width=4, dash='dot')), row=2, col=1)
	fig.update_yaxes(title_text="Paid Amount", row=2, col=1)
	fig.update_xaxes(title_text='Dates', row=2, col=1)
	fig.update_layout(title_text=table_title)

	pio.write_html(fig, file='templates/output.html', auto_open=False)





def plot_pred(cname,lename,from_d):
	df = model_call(str(cname),str(lename))
	model = pickle.load(open('model.pkl', 'rb'))
	lastdate_=give_last_date(cname, lename)
	pred=model.forecast(6)
	pred_mean=round(pred.mean(), 3)
	start = from_d
	show_predict=np.array(df[str(datetime.datetime.strptime(str(lastdate_),'%Y%m%d').date())])
	show_predict=np.append(show_predict, pred)
	


	fig = go.Figure()
	fig.add_trace(go.Scatter(x = df[start:].index, y=df[start:], mode='lines', name='Recorded'))
	fig.add_trace(go.Scatter(x=give_dates(lastdate_), y=show_predict, mode='lines', name='Predicted',line=dict(width=4, dash='dot')))
	
	final_verdict=''
	if pred_mean>0:
		final_verdict="Mean paid amount of next 6 months is {m}.<br>So, considering this it is beneficial to work with this client.".format(m=pred_mean)
	else :
		final_verdict="Mean paid amount of next 6 months is {m}.<br>So, considering this it is not beneficial to work with this client.".format(m=pred_mean)
	fig.update_layout(title_text=final_verdict)
	fig.update_yaxes(title_text="Paid Amount")
	fig.update_xaxes(title_text='Dates')
	pio.write_html(fig, file='templates/predict.html', auto_open=False)