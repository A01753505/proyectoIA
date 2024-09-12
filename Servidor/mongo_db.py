from pymongo import MongoClient

def get_database():
    # Conectar a MongoDB en el puerto 27017
    client = MongoClient('mongodb://localhost:27017/')
    
    # Crear o conectar a la BD
    db = client['SpaceSoulmates']
    return db