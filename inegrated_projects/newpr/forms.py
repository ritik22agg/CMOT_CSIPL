from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField
from wtforms.validators import DataRequired, Length , EqualTo
import pandas as pd

lis = ['Please Select Client Name']
df = pd.read_csv(r'processed_data.csv')
cl = df['Client Name'].unique()

lis.extend(cl)

class OutputForm(FlaskForm):
    clientName = SelectField('clientName' , choices = lis)
    Legal = SelectField('Le')


class CompareForm(FlaskForm):
    client1 = SelectField('client1' , choices = lis)
    Legal1 = SelectField('Le1')
    client2 = SelectField('client2' , choices = lis)
    Legal2 = SelectField('Le2')

