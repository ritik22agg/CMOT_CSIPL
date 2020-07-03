# CMOT_CSIPL

# ⚡️CitiBot - SmartEmailTracker - Investment predictor

## Description:

Project 1: Build an interactive system which can train itself from various sources and answer contextual questions

Project 2: Automate the process of classification of incoming emails based on their content

Project 3: Predict future business with clients based on past data for better decision making

<p align="center">
  For details on all the ML models tried and tested, model comparisons, UI features and better understanding of system workflow, please refer to <a href="https://docs.google.com/presentation/d/1kLh_WCi8Xh1EninU81dXtK7xs1VSrnS2v5qASRCsdik/edit#slide=id.g8af3e4b249_0_5">this document</a>.
</p>

## 🛠 Installation and quickstart:

Clone the repository in your terminal:
```sh
git clone https://github.com/ritik22agg/CMOT_CSIPL.git
```
Create a virtual environment for this project using the following command:
```sh
virtualenv venv
```
Activate Your Virtual Environment
```sh
venv/bin/activate
```
Project installations can be done with `pip`:
```sh
pip install -r requirements.txt
```
Install glove.6B.300d.txt (1 GB file) from: 
```sh
```

Open up localhost at http://127.0.0.1:5000/ for live demo of the app.

Email input can given from the form or uploaded as a PDF, text file or an image.
 - NOTE: If any file is found not to be showing, please write the absolute path for that file.
 - Abbreviations, industry specific keywords and definitions can be specificied as per need in wordfile.py (Merged UI_Listener directory).

## Sample email format assumed for PDF or txt file:

To: Rahul@CitiBankPune.com 

From: Mike@BNYMellon.com 

Subject: Transaction 608234 Complete 

Hi,
Hope you are well.

Wanted to inform you that transaction has been completed successfully.
Thanks for your assistance!

Mike
