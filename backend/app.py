from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import json
from flask import Flask, jsonify
from flask_pymongo import pymongo
from pymongo import MongoClient
from bson.json_util import dumps
from bson import ObjectId
import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import csv
import pandas as pd

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

#get envỉronment variable
MONGO_URI = os.environ.get('MONGO_URI')

# Define a flask app
app = Flask(__name__)
client = MongoClient(MONGO_URI)
db = client['car-prediction']
car_collection = db['cars']

# Read CSV file contain class name
data_class_index = pd.read_csv('uploads/names.csv')

# Load Model 
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 216)
model_state = torch.load('models/car_classifier.pth', map_location=torch.device('cpu'))
model.load_state_dict(model_state)
model.eval()

# Takes image data in bytes, applies the series of transforms and returns a tensor
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((320, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).float().unsqueeze(0)

# Prediction
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor)
    _, y_hat = torch.max(outputs.data, 1)
    predicted_idx = int(y_hat.item())
    return data_class_index['Name'][predicted_idx]


# Api route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        # Read upload image as bytes
        img_bytes = file.read()
        # Return predicted class_name
        class_name = get_prediction(image_bytes=img_bytes)
        # Using predicted class_name to query database
        car = car_collection.find_one({'name': class_name})
        # Return result as JSON to user
        return jsonify(JSONEncoder().encode(car))
    return None


@app.route('/cars', methods=['GET'])
def cars():
    cars = car_collection.find()
    result = list(cars)
    return jsonify(JSONEncoder().encode(result))

if __name__ == '__main__':
    app.run(debug=True)

