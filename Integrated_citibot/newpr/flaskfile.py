
from flask import Flask, render_template, url_for, request, jsonify
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
from forms import OutputForm, CompareForm
import csv
from successfulTransactionCompare import comparareClientsTransaction
from dict import get_dict
from file1 import get_Ans
from compf import plot_com, plot_pred


app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

allClients=[]
allLegalEntities=[]
allData=give_clients_and_entities()
allClients=allData[0]
allLegalEntities=allData[1]
listofatt=take_fields()
final_result = predict_top_clients(len(allClients))

d = get_dict()

print(final_result)
print('..................')
print(allLegalEntities)
print(',..............................................start......................................................')


@app.route('/get_food/<cl>')
def get_food(cl):
	if cl not in d:
		return jsonify([])
	else:
		return jsonify(d[cl])

		
@app.route('/')
@app.route('/home')
def home():
	form = OutputForm()
	return render_template('index.html', form = form)


@app.route("/compare")
def compare():
	form = CompareForm()
	return render_template('compare.html', form = form)


@app.route('/predict',methods=['POST'])
def predict():
	form = OutputForm()
	from_d = request.form['from'];
	cname = form.clientName.data
	lename = form.Legal.data;
	#attribute_value = request.form['attribute_value'];
	plot_pred(cname,lename,from_d)
	return render_template('predict.html')




@app.route("/script", methods = ["POST"])
def script():
	form = CompareForm()
	client1 = form.client1.data
	client2 = form.client2.data
	legal1 = form.Legal1.data
	legal2 = form.Legal2.data
	from_d = request.form['from'];
	#attribute_value = request.form['attribute_value'];
	plot_com(client1,client2,legal1,legal2,from_d)
	return render_template('output.html')


@app.route("/topNClients.html",  methods = ["POST", "GET"])
def topNClients() :

	if request.method == 'POST' :
		number = request.form['number']
		criteria= request.form['criteria']

		if criteria == 'Mean' :
			get_Ans(number,final_result)

		else :
			comparareClientsTransaction(number, allClients)


		return render_template('topNClientsGraph.html')


	return render_template('topNClients.html', allClientsNumber=len(allClients))


@app.route("/alter.html")
def alter() :
	return render_template('alter.html')



if __name__ == '__main__':
      app.run(debug=True)
