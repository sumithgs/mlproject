import pickle
from flask import Flask, request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# route for home page
@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
    gender=request.form.get('gender', 'male'),
    race_ethnicity=request.form.get('ethnicity', 'group A'),
    parental_level_of_education=request.form.get(
        'parental_level_of_education', 
        'some college'
    ),
    lunch=request.form.get('lunch', 'standard'),
    test_preparation_course=request.form.get(
        'test_preparation_course', 
        'none'
    ),
    reading_score=int(request.form.get('reading_score', 70)),
    writing_score=int(request.form.get('writing_score', 70))
)

    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('index.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)