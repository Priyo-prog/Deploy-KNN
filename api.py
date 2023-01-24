from classification_model.config.core import Buyer
from classification_model.config.core import TRAINED_MODEL_DIR, config
from classification_model.processing.data_manager import load_model

import uvicorn

from fastapi import FastAPI

import numpy
import pandas
import pickle

#  Create the app object
app = FastAPI()

model_file = f"{config.app_config.model_save_file}.pkl"

# Bring the model pickle file
classifier = load_model(file_name=model_file)

# Index route, opens automatically on https://127.0.0.1:8000
@app.get('/')
def index():
    return {"message": "Hello Deloitte"}

# Route with a single parameter returns the parameter within a message
@app.get('/{name}')
def get_name(name:str):
    return {"Priyo's Page": f"{name}"}

# Expose the prediction method of the model
@app.post('/predict')
def predict_purchase(data:Buyer):
    data = data.dict()
    print(data)
    age = data["Age"]
    salary = data["EstimatedSalary"]
    prediction = classifier.predict([[age,salary]])

    if (prediction[0] == 0):
        prediction = "Not Purchased"
    else:
        prediction = "Purchased"    
    return {"prediction": prediction} 

# Run the api with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)       
