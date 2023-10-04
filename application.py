from flask import Flask, request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import pandas as pd
import numpy as np
from src.logger import logging

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    elif request.method == "POST":
        data = CustomData(
            geography=request.form.get('geography'),
            gender=request.form.get('gender'),
            age=float(request.form.get('age')),
            tenure=float(request.form.get('tenure')),
            balance=float(request.form.get('balance')),
            num_of_products=float(request.form.get('num_of_products')),
            has_cr_card=float(request.form.get('has_cr_card')),
            is_active_member=float(request.form.get('is_active_member')),
            estimated_salary=float(request.form.get('estimated_salary')),
            credit_score=float(request.form.get('credit_score'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0]*100)
    
if __name__=="__main__":
    app.run(host="0.0.0.0")