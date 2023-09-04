from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            customer_type=request.form.get('customer_type'),
            age=request.form.get('age'),
            type_of_travel=request.form.get('type_of_travel'),
            Class=request.form.get('Class'),
            flight_distance=request.form.get('flight_distance'),
            inflight_wifi_service=request.form.get('inflight_wifi_service'),
            departure_arrival_time_convenient=request.form.get('departure_arrival_time_convenient'),
            ease_of_online_booking=request.form.get('ease_of_online_booking'),
            gate_location=request.form.get('gate_location'),
            food_and_drink=request.form.get('food_and_drink'),
            online_boarding=request.form.get('online_boarding'),
            seat_comfort=request.form.get('seat_comfort'),
            inflight_entertainment=request.form.get('inflight_entertainment'),
            on_board_service=request.form.get('on_board_service'),
            leg_room_service=request.form.get('leg_room_service'),
            baggage_handling=request.form.get('baggage_handling'),
            checkin_service=request.form.get('checkin_service'),
            inflight_service=request.form.get('inflight_service'),
            cleanliness=request.form.get('cleanliness'),
            departure_delay_in_minutes=request.form.get('departure_delay_in_minutes'),
            arrival_delay_in_minutes=request.form.get('arrival_delay_in_minutes'),
       )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        

