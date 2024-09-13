from pymongo import MongoClient #type: ignore
import gridfs #type: ignore
import os

def get_database():
    # Conectar a MongoDB en el puerto 27017
    client = MongoClient('mongodb://localhost:27017/')

    # Crear o conectar a la BD
    db = client['SpaceSoulmates']

    directorio_fotos = './Fotos'
    if not os.path.exists(directorio_fotos):
        raise FileNotFoundError(f"El directorio {directorio_fotos} no existe")


    fs = gridfs.GridFS(db)
    for nombre_archivo in os.listdir(directorio_fotos):
        if nombre_archivo.endswith(('jpg', 'jpeg', 'png')):
            ruta_archivo = os.path.join(directorio_fotos, nombre_archivo)
            with open(ruta_archivo, 'rb') as f:
                contenido = f.read()
                fs.put(contenido, filename=nombre_archivo)
                print(f"Imagen {nombre_archivo} guardada en la base de datos")

    return db