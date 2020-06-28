import traceback
import mysql.connector
#systemctl status mysql.service
#sudo mysql -u root -ppassword
def query_formation(entities, table:str):

    query = "SELECT * FROM {} WHERE".format(table)
    for e in entities:
        if e['extractor'] == 'CRFEntityExtractor':
            query += " {} = {} and".format(e['entity'], e['value'])
    print(query)
    return query[:-3]

def getData(query:str, table):

        try:
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="Uchihas123",
                database="rasadatabse"
                )
            cursor = mydb.cursor()
            cursor.execute(query)

            results = cursor.fetchall()
            cursor.execute("show columns from {}".format(table))
            features = cursor.fetchall()
            return results, features
        except:
            print("Error occured while connecting to database or fetching data from database. Error Trace: {}".format(traceback.format_exc()))
            return []
