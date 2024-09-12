# Python libraries
from flask import Flask, request, jsonify
import numpy as np
import joblib # type: ignore
import random
from datetime import datetime
from mongo_db import get_database
import base64
from bson.binary import Binary
# Files management
import os
from werkzeug.utils import secure_filename

# Conection to the DB
db = get_database()
collection = db['passengers']
users = collection.find()
print(users)

# Load model
dt = joblib.load('catboost_model.joblib')

# Create Flask App
server = Flask(__name__)

# Function to clean the data, and generate random data to give to the model
def clean_data(data):
    # Transoformar home planet
    home_planet = data['home_planet']
    earth = False
    europa = False
    mars = False
    if home_planet == 'Earth':
        earth = True
    elif home_planet == 'Europa':
        europa = True
    else:
        mars = True

    # Transformar Destination
    destination = data['destination']
    cancri = False
    ps = False
    trappist = False
    if destination == '55 Cancri e':
        cancri = True
    elif destination == 'PS0 J318.5-22':
        ps = True
    else:
        trappist = True
        
    # Sacar la edad con base en el birth date
    print(type(data['birth_date']))
    birth_date = datetime.strptime(data['birth_date'], "%Y-%m-%d")
    today = datetime.now()
    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    # Sacar el Deck y el Side
    cabin = data['cabin']
    side_p = False
    side_s = False

    if cabin[-1] == 'S':
        side_s = True
    else:
        side_p = True
    
    map = {chr(i + 65): str(i) for i in range(7)}
    map['H'] = '7'
    
    deck = map[cabin[0]]

    data_modifed = {
        "CryoSleep": bool(np.random.choice([True, False])),
        "VIP":  bool(np.random.choice([True, False])),
        "Has_family": bool(data["has_family"]),
        "HomePlanet_Earth": earth,
        "HomePlanet_Europa": europa,
        "HomePlanet_Mars": mars,
        "Deck": deck,
        "Side_P": side_p,
        "Side_S": side_s,
        "Destination_55 Cancri e": cancri,
        "Destination_PS0 J318.5-22": ps,
        "Destination_TRAPPIST-1e": trappist,
        "Age": age,
        "RoomService": random.uniform(1, 1000),
        "FoodCourt": random.uniform(1, 1000),
        "ShoppingMall": random.uniform(1, 1000),
        "Spa": random.uniform(1, 1000),
        "VRDeck": random.uniform(1, 1000)
    }

    return data_modifed

@server.route('/users', methods=['GET'])
def show_all_users():
    try:
        # Recuperar todos los documentos de la colección
        users = collection.find()
        
        # Imprimir los documentos
        user_list = []
        for user in users:
            user['_id'] = str(user['_id'])  # Convertir ObjectId a cadena
            
            # Convertir campos binarios a Base64 (si es necesario)
            if 'photo' in user and isinstance(user['photo'], bytes):
                user['photo'] = base64.b64encode(user['photo']).decode('utf-8')

            user_list.append(user)

        if not user_list:
            return jsonify({"message": "No users found."})
        
        return jsonify(user_list)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

@server.route('/uploadDB', methods=['POST'])
def upload_to_database():
    data = request.json
    result = collection.insert_one(data)
    return f"Succesful upload: {result.inserted_id}"

# Define a route to send JSON data
@server.route('/predictjson', methods=['POST'])
def predictjson():
    # Procesar datos de entrada
    data = request.json
    data = clean_data(data)
    print(data)
    
    # Change the data type to a numpy array,and ensure that we use a shape of (1, 18)
    inputData = np.array([
        int(data['CryoSleep']),
        int(data['VIP']),
        int(data['Has_family']),
        int(data['HomePlanet_Earth']),  
        int(data['HomePlanet_Europa']),
        int(data['HomePlanet_Mars']),
        int(data['Deck']),
        int(data['Side_P']),
        int(data['Side_S']),  
        int(data['Destination_55 Cancri e']),
        int(data['Destination_PS0 J318.5-22']),
        int(data['Destination_TRAPPIST-1e']),
        float(data['Age']),
        float(data['RoomService']),  
        float(data['FoodCourt']),
        float(data['ShoppingMall']),
        float(data['Spa']),
        float(data['VRDeck'])
    ]).reshape(1, -1)

    print(inputData.shape)

    # Predict using the input and the model
    result = dt.predict(inputData)
    # Return response
    return jsonify({'Prediction': str(result[0])})

if __name__ == '__main__':
    server.run(debug=False, host='0.0.0.0', port=8080)