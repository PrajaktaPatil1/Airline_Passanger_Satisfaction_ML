import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join(os.getcwd(),"artifacts","model.pkl")
            preprocessor_path=os.path.join(os.getcwd(),'artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        gender: str,
        customer_type: str,
        age: int,
        type_of_travel: str,
        Class: str,
        flight_distance: int,
        inflight_wifi_service: int,
        departure_arrival_time_convenient: int,
        ease_of_online_booking: int,
        gate_location: int,
        food_and_drink: int,
        online_boarding: int,
        seat_comfort: int,
        inflight_entertainment: int,
        on_board_service: int,
        leg_room_service: int,
        baggage_handling: int,
        checkin_service: int,
        inflight_service: int,
        cleanliness: int,
        departure_delay_in_minutes: int,
        arrival_delay_in_minutes: int,):
    
        self.gender=gender
        self.customer_type=customer_type
        self.age=age
        self.type_of_travel=type_of_travel
        self.Class=Class
        self.flight_distance=flight_distance
        self.inflight_wifi_service=inflight_wifi_service
        self.departure_arrival_time_convenient=departure_arrival_time_convenient
        self.ease_of_online_booking=ease_of_online_booking
        self.gate_location=gate_location
        self.food_and_drink=food_and_drink
        self.online_boarding=online_boarding
        self.seat_comfort=seat_comfort
        self.inflight_entertainment=inflight_entertainment
        self.on_board_service=on_board_service
        self.leg_room_service=leg_room_service
        self.baggage_handling=baggage_handling
        self.checkin_service=checkin_service
        self.inflight_service=inflight_service
        self.cleanliness=cleanliness
        self.departure_delay_in_minutes=departure_delay_in_minutes
        self.arrival_delay_in_minutes=arrival_delay_in_minutes
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "customer_type": [self.customer_type],
                "age": [self.age],
                "type_of_travel": [self.type_of_travel],
                "Class": [self.Class],
                "flight_distance": [self.flight_distance],
                "inflight_wifi_service": [self.inflight_wifi_service],
                "departure_arrival_time_convenient": [self.departure_arrival_time_convenient],
                "ease_of_online_booking": [self.ease_of_online_booking],
                "gate_location": [self.gate_location],
                "food_and_drink": [self.food_and_drink],
                "online_boarding": [self.online_boarding],
                "seat_comfort": [self.seat_comfort],
                "inflight_entertainment": [self.inflight_entertainment],
                "on_board_service": [self.on_board_service],
                "leg_room_service": [self.leg_room_service],
                "baggage_handling": [self.baggage_handling],
                "checkin_service": [self.checkin_service],
                "inflight_service": [self.inflight_service],
                "cleanliness": [self.cleanliness],
                "departure_delay_in_minutes": [self.departure_delay_in_minutes],
                "arrival_delay_in_minutes": [self.arrival_delay_in_minutes],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
    