import csv
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
readed_data = {}

#number = 3
#allClients=['sbi', 'jpmchase', 'hdfcbankltdmumbaiheadoffice','bankofbarodamumbaiheadoffice','barclays','hsbc']
def comparareClientsTransaction(number, allClients):
    print(number)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for cli in allClients:
        temp = {}
        temp['RESOLVED']=0.0
        temp['DISCARDED']=0.0
        temp['PENDING_COLLATERAL']=0.0
        readed_data[cli]=temp

    with open("processed_data.csv", 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields=next(csvreader)
        #test=0
        for row in csvreader:
            readed_data[row[1]][row[4]]=readed_data[row[1]][row[4]]+1.0

    for cli in readed_data.keys():
        total = 0.0
        for ty in readed_data[cli].keys():
            total=total+readed_data[cli][ty]
        for ty in readed_data[cli].keys():
            if total >0:
                readed_data[cli][ty]=round((readed_data[cli][ty]*100)/total, 3)

    print(readed_data)
    sorted_clients = {}
    x_bar = []
    y_bar_resolved=[]
    y_bar_discarded=[]
    y_bar_pendingcollateral=[]
    for cli in readed_data.keys():
        sorted_clients[cli]=-1*readed_data[cli]['RESOLVED']
    s_clients={k: -1*v for k, v in sorted(sorted_clients.items(), key=lambda item: item[1])}
    print(s_clients)
    i=0
    for cli in s_clients:
        x_bar.append(cli)
        y_bar_resolved.append(readed_data[cli]['RESOLVED'])
        y_bar_discarded.append(readed_data[cli]['DISCARDED'])
        y_bar_pendingcollateral.append(readed_data[cli]['PENDING_COLLATERAL'])
        i=i+1
        print(i)
        if i == int(number):
            break
    print('.................................................................................')
    print(x_bar)
    print(y_bar_resolved)
    print(y_bar_discarded)
    print(y_bar_pendingcollateral)

    table_col=['Clients', 'Resolved Payments', 'Pending Collateral', 'Discarded']
    table_row=[]
    table_row.append(x_bar)
    table_row.append(y_bar_resolved)
    table_row.append(y_bar_pendingcollateral)
    table_row.append(y_bar_discarded)
    fig1=go.Figure()
    fig1 = make_subplots(rows=2, cols=1,  vertical_spacing=0.09,specs=[ [{"type": "table"}],[{"type": "bar"}] ] )
    fig1.add_trace(go.Table(header=dict(values=table_col,font=dict(size=10),align="left"), cells=dict(values=table_row,  height=40,align="left")), row=1, col=1)
    fig1.add_trace(go.Bar(x=x_bar, y=y_bar_resolved, name='Resolved', hovertext='Resolved Payments %',text=y_bar_resolved, textposition='auto'), row=2, col=1)
    fig1.add_trace(go.Bar(x=x_bar, y=y_bar_pendingcollateral, name='Pending Col..',hovertext='Pending Collateral %',text=y_bar_pendingcollateral, textposition='auto'), row=2, col=1)
    fig1.add_trace(go.Bar(x=x_bar, y=y_bar_discarded, name='Discarded',hovertext='Discarded %',text=y_bar_discarded, textposition='auto'), row=2, col=1)
    pio.write_html(fig1, file='templates/topNClientsGraph.html')
