import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import date
import datetime
from dateutil.relativedelta import relativedelta
import datetime
import pandas as pd
from helper_functions import give_last_date, take_fields, give_dates, give_clients_and_entities

			
def get_Ans(number,final_result):
	x_bar = list()
	name_scatter =[]
	now_date=str(date.today())
	new_now_date=""
	for w in now_date :
		if(w!='-') :
			new_now_date+=w
	x_scatter_temp = give_dates(new_now_date, 6)
	x_scatter=x_scatter_temp[1:]
	y_bar = list()
	y_scatter = list()
	table_col = list()
	table_col.append('Clients')
	for var in x_scatter :
		table_col.append(str(var))
	table_col.append('Predicted Paid Amt Mean(USD)')
	table_row = list()

	i=0
	for key in final_result.keys():
		x_bar.append(key)
		y_bar.append(int(final_result[key][0]))
		table_row_temp=list()
		table_row_temp.append(key)
		y_scatter_temp = list()
		j=1
		while j < len(final_result[key]):
			y_scatter_temp.append(final_result[key][j])
			table_row_temp.append(final_result[key][j])
			j=j+1
		table_row_temp.append(final_result[key][0])
		y_scatter.append(y_scatter_temp)
		table_row.append(table_row_temp)
		i=i+1
		if i ==int(number) :
			break

	new_table_row = []
	for i in range(len(table_row[0])):
		table_row_temp_new = []
		for elem in table_row :
			table_row_temp_new.append(elem[i])
		new_table_row.append(table_row_temp_new)

	for word in x_bar:
		if len(word) < 12:
			name_scatter.append(word)
		else :
			small_word=""
			for i in range(12) :
				small_word+=word[i]
			name_scatter.append(small_word+"...")

	fig = make_subplots(rows=3, cols=1,  vertical_spacing=0.09,specs=[ [{"type": "table"}],[{"type": "bar"}],[{"type": "scatter"}] ] )
	fig.add_trace(go.Table(header=dict(values=table_col,font=dict(size=10),align="left"), cells=dict(values=new_table_row,  height=40,align="left")), row=1, col=1)
	fig.add_trace(go.Bar(x=name_scatter, y=y_bar, text=x_bar,textposition='outside'), row=2, col=1)
	fig.update_yaxes(title_text="Mean Prdicted Amt(USD)", row=2, col=1)
	fig.update_yaxes(title_text="Predicted Paid Amt ", row=3, col=1)
	fig.update_xaxes(title_text="Dates", row=3, col=1)

	i=0
	for element_y in y_scatter:
		fig.add_trace(go.Scatter(x=x_scatter, y=element_y, name=name_scatter[i]), row=3, col=1)
		i=i+1
	pio.write_html(fig, file='templates/topNClientsGraph.html')