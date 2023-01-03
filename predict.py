import numpy as np
from tensorflow.keras.models import load_model


class heart_disease_predictor:
    def __init__(self,attributes):
        self.attributes =attributes


    def prediction(self):
        # load model
        model = load_model('heart_model.h5')
        


        #model.summary()
        attributes = self.attributes

        result = model.predict(attributes)
        print(result)
        if result > 0.45:
            prediction = 'Severe'
            print(prediction)
           
        else:
            prediction = 'Normal'
            print(prediction)

val=[[66,1,1,170,120,0,1,120,1,0,1,3,1]]#150=1 249,336=0



a=heart_disease_predictor(val)
a.prediction()
