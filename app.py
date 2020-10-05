#app.py
from flask import Flask, render_template, request, redirect, session
import pandas as pd
import numpy as np
from models import dbMgr

app = Flask(__name__)

# Main Page
@app.route('/')
def index():
        return render_template("index.html")

###########################################################################

# Data Model Page
@app.route('/data_model_form')
def data_model():
        return render_template('data_model_form.html')


# Model Score Page
@app.route('/model_score_form')
def model_score():
        return render_template('model_score_form.html')


# Data Predict & Insert Data Base
@app.route('/modeling',methods=['POST'])
def predict_data():
        name = request.form['name']
        gender = request.form['gender']
        age = request.form['age']
        age_group = age[0] + '0대'
        weight = request.form['weight']
        runtime = request.form['runtime']
        runpulse = request.form['runpulse']
        rstpulse = request.form['rstpulse']
        maxpulse = request.form['maxpulse']
        
        # Input Data 
        data = [[age,weight,runtime,runpulse,rstpulse,maxpulse]]

        # Predict Y (Lasso Model)
        result = dbMgr.load_model_Lasso()
        data = pd.DataFrame(data=data)
        predict = result.predict(data)

        # Insert Data Base 
        data = (name,gender,age,age_group,weight,str(predict[0]),runtime,runpulse,rstpulse,maxpulse)
        dbMgr.insert_customer(data)

        return render_template('data_model_form.html', predict=predict.round(3), names=name)


# Model Performance & Importance Score
@app.route('/model_score',methods=['POST'])
def score():
        result = dbMgr.model_Score_Lasso()
        print(result)
        return render_template('model_score_form.html',result1=result[0],result2=result[1],result3=result[2],result4=result[3],\
                result5=result[4],result6=result[5],result7=result[6],result8=result[7])
                
###########################################################################

# Finished Code 
if __name__ == '__main__' :
    app. debug = True
    app.run()