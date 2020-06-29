"""
Flask Application to start Smart Email Tracker
python start.py
on localhost:5000
"""
import csv
import os, json
import time
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
# from listener_xg import handle_new_email
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from flask_login import LoginManager, login_user, UserMixin, current_user, \
login_required, logout_user
from werkzeug.utils import secure_filename


date_ = datetime.today().strftime('%Y-%m-%d')

# from xgb_inp import inp
from file_parser import allowed_ext, extract_text
# from listener_xg import handle_new_email
# from GLOVE_XGBOOST import train
from dataload import loadData
from voicebotfunc import talk


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emails.sqlite3'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///User.db'
app.config['SECRET_KEY'] = "random string"
app.config['UPLOAD_FOLDER'] = "inputEmails"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

strs = []


class User(db.Model, UserMixin):
    """
    user database definition
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}','{self.password}')"



@login_manager.user_loader
def load_user(user_id):
    """
    load user
    """
    return User.query.get(int(user_id))

class RegistrationForm(FlaskForm):
    """
    registration form
    """
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = StringField('Password',
                           validators=[DataRequired(), Length(min=6, max=20)])
    confirm_password = StringField('Confirm Password',
                                   validators=[DataRequired(),
                                               EqualTo('password')])
    submit = SubmitField('SignUp')

    def validate_username(self, username):
        """
        username validation
        """
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('This username is already taken. \
                                   Please choose a different username')

    def validate_email(self, email):
        """
        email validation
        """
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('This email is already taken. \
                                   Please choose a different email-address')


class LoginForm(FlaskForm):
    """
    login form
    """
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = StringField('Password',
                           validators=[DataRequired(), Length(min=6, max=20)])
    submit = SubmitField('SignIn')
    remember = BooleanField('Remember Me')


class mails(db.Model):
    """
        email table definition
    """
    id = db.Column('mail_id', db.Integer, primary_key=True)
    mfrom = db.Column(db.String(50))
    mto = db.Column(db.String(50))
    mdate = db.Column(db.String(20))
    msubject = db.Column(db.String(200))
    mbody = db.Column(db.String(500))
    m_class = db.Column(db.String(50))
    ID = db.Column(db.String(50))


def __init__(self, mfrom, mto, mdate, msubject, ID, mbody, m_class):
    """
        definition of mail object
    """
    self.mfrom = mfrom
    self.mto = mto
    self.mdate = mdate
    self.msubject = msubject
    self.mbody = mbody
    self.m_class = m_class
    self.ID = ID


def write_mail(mail):
    """
        create txt copy of form mail
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f = open("./inputEmails/email_" + str(timestr) + '.txt', "w+")
    f.write("To: %s \n" % mail.mto)
    f.write("From: %s \n" % mail.mfrom)
    f.write("Subject: %s \n\n" % mail.msubject)
    f.write("Email_Body: %s \n" % mail.mbody)
    f.write("class: %s \n" % mail.m_class)
    f.write("Date: %s \n" % mail.mdate)
    f.close()


@app.route('/upload')
def upload_file():
   return render_template('upload.html')


def retrain():
    newDict1 = {'Complete', 'Failed', 'Request', 'General', 'Pending', 'Processing',
               'Request'}
    newDict2 = {}
    ct = 0
    for mail in mails.query.all():
        if mail.m_class not in newDict1:
            newDict2.add(mail.m_class)
    for mail in mails.query.all():
        if mail.m_class in newDict2 and mails.query.filter(mails.m_class == mail.m_class).count() > 39:
            newDict1.add(mail.m_class)
    with open('emaildataset.csv', 'a') as f:
        f.write('\n')
        f.close()
    with open('emaildataset.csv', 'a') as f:
        out = csv.writer(f)
        for mail in mails.query.all():
            if mail.m_class in newDict1:
                ct += 1
                out.writerow([mail.mfrom, mail.mto, mail.msubject, mail.mbody, mail.ID, mail.mdate, mail.m_class])
                db.session.delete(mail)
                db.session.commit()
        f.close()
    msg = '' + str(ct) + ' mail(s) sent for retraining successfully!'
    train()
    return render_template('retrained.html', message=msg)



@app.route('/retrain')
def retrain():
    """
    retrain model only if at least 40 mails of new classes
    """
    org_classes = {'Complete', 'Failed', 'Request', 'General', 'Pending', 'Processing'}
    new_classes = set()
    new_classes_40 = set()
    ctr = 0
    for mail in mails.query.all():
        curclass = str(mail.m_class).capitalize()
        if curclass not in org_classes:
            new_classes.add(mail.m_class)
    for mail in mails.query.all():
        if mail.m_class in new_classes and mails.query.filter(mails.m_class == mail.m_class).count() > 39:
            new_classes_40.add(mail.m_class)
    with open('emaildataset.csv', 'a') as f:
        f.write('\n')
        f.close()
    with open('emaildataset.csv', 'a') as f:
        out = csv.writer(f)
        for mail in mails.query.all():
            if str(mail.m_class).capitalize() in org_classes or mail.m_class in new_classes_40:
                ctr += 1
                out.writerow([mail.mfrom, mail.mto, mail.msubject, mail.mbody, mail.ID, mail.mdate, mail.m_class])
                db.session.delete(mail)
                db.session.commit()
        f.close()
    if ctr > 0:
        msg = '' + str(ctr) + ' mail(s) sent for retraining successfully!'
        train()
    else:
        msg = 'Less than 40 emails of new classes - model not retrained.'
    return render_template('retrained.html', message=msg)



@app.route('/show')
@login_required
def show_all():
    """
        display all emails currently in DB
    """
    #return render_template('show_all.html', mails=mails.query.filter(mails.mto == current_user.email).all())
    return render_template('show_all.html', mails=mails.query.order_by(mails.id.desc()).all())


@app.route("/", methods=["GET", "POST"])
def welcome():
    """
        homepage
        accept email with attachment from UI form
        predict and display output class
    """
    global myDict, mclass, tid
    if request.method == "POST":
        myDict=inputvalues = {
            "From": request.form['From'],
            "To": request.form['To'],
            "Subject": request.form['Subject'],
            "Message": request.form['Message'],
            "Date": date_
        }

        #handle file attachment in email from form
        body1=""
        if not request.files.get('file', None):
            print('No file uploaded')
        else:
            f = request.files['file']
            print('File Uploaded')

            sfname = (secure_filename(f.filename))
            timestr = time.strftime("%Y%m%d-%H%M%S")
            fullname = str(timestr) + sfname
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], fullname))

            #extract text from txt or pdf AND IMAGE, else "" returned
            body1 = extract_text(app.config['UPLOAD_FOLDER'] + '/' + fullname)

            if (body1!=""):
                body1 = '\n-----------Extracted from Attachment-----------\n' + body1

        #append attchment txt to message body for prediction
        inputvalues['Message'] = inputvalues['Message'] + str(body1)
        print(inputvalues['Message'])


        if (inputvalues['Subject']=="" and inputvalues['Message']==""):
            msg = 'Empty email or invalid attachment - no prediction!'
            return render_template('retrained.html', message=msg)

        #m_class, ID = inputfunc(inputvalues['To'], inputvalues['From'], inputvalues['Subject'], inputvalues['Message'])
        m_class, ID = inp(inputvalues['To'], inputvalues['From'], inputvalues['Subject'], inputvalues['Message'])
        mclass = m_class
        tid = ID
        return render_template("index1.html", m=m_class)

    else:
        return render_template("index1.html")


@app.route('/submit', methods = ['POST'])
def submit():
    if request.method == 'POST':
        file = request.files['data']
        path = '../botfiles/' + file.filename
        file.save(path)
        #loadData(path)
        q = 'python Backend/dynamictrain.py ' + path
        #os.system(q)
        #os.system('python Backend/train.py')
        print("TRAINED !!!")

    return redirect('/')


@app.route('/mike', methods=['POST'])
def mike():
    if request.method == 'POST':
        message, botmessage = talk()

        strs.append(message)
        strs.append(botmessage)

        return redirect('/')
        # loadtheModule(file.filename, file.filename)

    return redirect('/')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file_1():
    """
        upload email as a file
        pdf, txt, image
    """
    global myDict, mclass, tid
    if request.method == 'POST':

      if not request.files.get('file', None):
        msg = 'No file uploaded'
        return render_template('retrained.html', message=msg)

      f = request.files['file']
      print('File Uploaded=====')
      f = request.files['file']
      sfname = (secure_filename(f.filename))
      timestr = time.strftime("%Y%m%d-%H%M%S")
      fullname = str(timestr) + "_" +  sfname
      f.save("./inputEmails/" + fullname)

      if not allowed_ext('inputEmails/' + fullname):
          msg = 'Invalid file type - no prediction!'
          return render_template('retrained.html', message=msg)


      to_add, from_add, receivedDate, sub, id, body, m_class = handle_new_email('inputEmails/' + fullname)
      myDict = {
          "From": from_add,
          "To": to_add,
          "Subject": sub,
          "Message": body,
          "Date": receivedDate
      }
      mclass=m_class
      tid=id
      return render_template("index1.html", m=m_class)

    else:
      return render_template("index1.html")


@app.route('/newclass', methods=['Get', 'POST'])
def new_class():
    """
        manually correct predicted class
        can enter a new class also
    """
    if request.method == "POST":
        mclass = str(request.form['NewClass'])
        mail = mails()
        __init__(mail, myDict['From'], myDict['To'], myDict['Date'],
                 myDict['Subject'], tid, myDict['Message'], mclass)
        write_mail(mail)
        db.session.add(mail)
        db.session.commit()
        # time.sleep(4)
        flash('Record was successfully added')
        return render_template("index1.html")
    else:
        return render_template("new_class.html")


@app.route('/wrong', methods=['GET', 'POST'])
def wrong():
    """
        if predicted class is wrong
        input new class
        add to DB
    """
    if request.method == "POST":
        mclass = str(request.form.get("class", None))
        if mclass == 'newclass':
            return redirect(url_for('new_class'))
        mail = mails()
        __init__(mail, myDict['From'], myDict['To'], myDict['Date'],
                 myDict['Subject'], tid, myDict['Message'], mclass)
        write_mail(mail)
        db.session.add(mail)
        db.session.commit()
        #time.sleep(4)
        flash('Record was successfully added')
        return redirect(url_for('show_all'))
    else:
        return render_template("dropdown.html")


@app.route('/right')
def right():
    """
        if predicted class is right, add mail to DB
    """
    mail = mails()
    __init__(mail, myDict['From'], myDict['To'], myDict['Date'],
             myDict['Subject'], tid, myDict['Message'], mclass)
    write_mail(mail)
    db.session.add(mail)
    db.session.commit()
    time.sleep(4)
    flash('Record was successfully added')
    return redirect(url_for('welcome'))


@app.route('/thread', methods = ['GET','POST'])
@login_required
def findthread():
    """
        find all emails with particular transaction id
    """
    if request.method == "POST":
        id = str(request.form['transacid'])
        return render_template('show_all.html', mails=mails.query.filter(mails.mdate == id and mails.mto == current_user.email).all())
    else :
        return render_template('findthread.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
        user login
    """
    if current_user.is_authenticated:
        return redirect(url_for('welcome'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('welcome'))
        else:
            flash('Login unsuccessful. You have entered incorrect email or password', 'danger')
    return render_template('login.html', title='login', form=form)


@app.route('/logout')
def logout():
    """
        logout current signed in user, redirect to homepage
    """
    logout_user()
    return redirect(url_for('welcome'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """
        Register - create a new account
    """
    if current_user.is_authenticated:
        return redirect(url_for('welcome'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


        
@app.route("/details", methods = ['GET'])
def details():
    lisp = []
    with open('../botfiles/records.json') as json_file: 
        data = json.load(json_file) 
        for dic in data.values():
            lisp.append(dic)
    return render_template('details.html', lis = lisp)

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
