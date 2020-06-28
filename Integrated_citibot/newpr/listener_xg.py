"""
Created on Mon Jun 15 00:04:39 2020
@author: divya
renamed from listenerpdfXG.py to listener_xg.py
PyLint Score : 9.89/10
"""
# Tuesday 23rd JUNE 2020 10PM
#loads XG boost model
import os
import shutil
import time
import sqlite3

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from xgb_inp import inp
#from Glove_XGBoost import inp
from file_parser import parse
from parseimage import ocr




def handle_new_email(mail_path):
    """
    Extract text from email depending on file type
    Call model to get output class
    Return to_add, from_add, received_date, sub, tid, body, outputclass
    """
    #print(type(mail_path))
    #added this for image remove if problematic
    if (mail_path.endswith('.png') or mail_path.endswith('.jpg') or
            mail_path.endswith('.jpeg')):
        #print('Email Image uploaded!!')
        extracted_text = ocr(mail_path)
        email = extracted_text.split('\n')

    elif mail_path.endswith('.pdf'):
        #print('PDF received')
        extracted_text = parse(mail_path)
        email = extracted_text.split('\n')
        #print(email)

    elif mail_path.endswith('.txt'):
        #print('txt received')
        email = open(mail_path, "r")
    else:
        #Unhandled file type
        return '', '', '', '', '', '', ''

    to_add = ''
    from_add = ''
    sub = ''
    body = ''
    for line in email:
        #print(f'**{line}**')
        if line.startswith('To: ') or line.startswith('to: '):
            pieces = line.split()
            to_add = pieces[1]
            #print(f'To---->Found!!!{to_add}')
            continue
        if line.startswith('From: ') or line.startswith('from: '):
            pieces = line.split()
            from_add = pieces[1]
            continue
        if line.startswith('Subject: ') or line.startswith('subject: '):
            pieces = line.split()
            subject = pieces[1:]#remove word subject
            for word in subject:
                sub = sub + " " + word
            # text=text+" "
            continue
        body = body + line
    #text = sub + " " + body
    received_date = time.ctime(os.path.getctime(mail_path))

    if mail_path.endswith('.txt'):
        email.close()
        #print(encoded_mail)

    outputclass, tid = inp(to_add, from_add, sub, body)

    #add_to_db(to_add, from_add, received_date, sub, tid, body, outputclass)
    #move_email(mail_path, outputclass)
    print(f'====>Event Processing completed')

    return to_add, from_add, received_date, sub, tid, body, outputclass

def add_to_db(to_add, from_add, received_date, sub, tid, body, outputclass):
    """
    add email fields and  output class to DB
    """
    #conn = sqlite3.connect('mails.sqlite3')
    conn = sqlite3.connect('User.db')
    cur = conn.cursor()
    cur.execute('''INSERT INTO mails (mto, mfrom, mdate, msubject, ID, mbody,
               m_class)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (to_add, from_add, received_date, sub, tid, body, outputclass))
    print("Inserted in DB\n")
    conn.commit()
    cur.close()

def move_email(mail_path, outputdir):
    """
    move email to output class folder
    """
    #Check if output class directory exists, if not, create it
    check_folder = os.path.isdir(outputdir)
    if not check_folder:
        os.makedirs(outputdir)
        print("created folder : ", outputdir)
    #move email to class output directory
    #shutil.move(mail_path, outputdir)
    shutil.copy(mail_path, outputdir)
    os.remove(mail_path)
    print("moved to folder : ", outputdir)

def verify_class(outputclass):
    """
    console interface
    manually verify predicted class before inserting in DB and moving to folder
    """
    print(f'Class predicted is {outputclass} ')
    ans = (str(input('Is the class correct (yes/no) ? ')))
    if ans.lower() == "no":
        newclass = (str(input('Enter new class name : ')))
        newclass = newclass.capitalize()
        print(newclass)
        outputclass = newclass
    return outputclass

# ## Directory Watcher

def on_created(event):
    """
    if a email file is created in inputEmails folder
    process it
    """
    print(f"====>Event Received: {event.src_path} received!")
    to_add, from_add, received_date, sub, tid, body, outputclass = handle_new_email(event.src_path)
    if outputclass != '':
        #uncomment following line to allow manual verification of class on cmd
        #outputclass = verify_class(outputclass)
        add_to_db(to_add, from_add, received_date, sub, tid, body, outputclass)
        move_email(event.src_path, outputclass)

if __name__ == "__main__":
    PATTERNS = "*"
    IGNORE_PATTERNS = ""
    IGNORE_DIRECTORIES = False
    CASE_SENSITIVE = True
    MY_EVENT_HANDLER = PatternMatchingEventHandler(PATTERNS, IGNORE_PATTERNS,
                                                   IGNORE_DIRECTORIES, CASE_SENSITIVE)
    MY_EVENT_HANDLER.on_created = on_created
    #new emails will be created in inputEmails directory
    PATH = "inputEmails"
    #path = sys.argv[1] if len(sys.argv) > 1 else 'inputEmails'
    GO_RECURSIVELY = False
    MY_OBSERVER = Observer()
    MY_OBSERVER.schedule(MY_EVENT_HANDLER, PATH, recursive=GO_RECURSIVELY)
    MY_OBSERVER.start()
    print('====> Observer Started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        MY_OBSERVER.stop()
        print('====> Observer Stopped')
        MY_OBSERVER.join()

#handle_new_email("inputEmails/email.txt")