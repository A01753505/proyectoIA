# Python libraries
from flask import Flask, request, jsonify, send_from_directory  # type: ignore
import numpy as np
import joblib # type: ignore
import random
from datetime import datetime
from mongo_db import get_database
import base64
from bson.binary import Binary  # type: ignore
# Files management
import os
from werkzeug.utils import secure_filename  # type: ignore
from flask_cors import CORS # type: ignore
import gridfs
from io import BytesIO

# Conection to the DB
db = get_database()
collection = db['passengers']
users = collection.find()
print(users)
fs = gridfs.GridFS(db)

# Load model
dt = joblib.load('model.joblib')

# Create Flask App
server = Flask(__name__)
CORS(server)

@server.route('/')
def index():
    return send_from_directory('../Pagina web', 'index.html')


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
    map['T'] = '7'

    deck = map[cabin[0]]

    valores = []
    for i in range(5):
        if np.random.rand() < 0.7:
            valores.append(0)
        else:
            valores.append(np.random.randint(1, 10001))

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
        "RoomService": valores[0],
        "FoodCourt": valores[1],
        "ShoppingMall": valores[2],
        "Spa": valores[3],
        "VRDeck": valores[4]
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

# Recuperar fotos
@server.route('/imagenes', methods=['GET'])
def obtener_imagenes():
    imagenes = []

    for file in fs.find():
        file_data = fs.get(file._id).read()
        imagen_binaria = BytesIO(file_data)
        imagen_binaria.seek(0)

        imagenes.append({
            'nombre': file.filename,
            'contenido': imagen_binaria.getvalue()
        })

    return jsonify(imagenes)

if __name__ == '__main__':
    server.run(debug=False, host='0.0.0.0', port=8080)