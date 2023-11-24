# This file implements the logic necessary for transfer learning. It is only
# used to create the functions in functions.py that are then used by the streamlit app.


import os
import pickle
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pymongo.database import Database
from pymongo.mongo_client import MongoClient
from gridfs import GridFS

def connect_to_db() -> tuple[Database, GridFS]:
    cluster: MongoClient = MongoClient("mongodb+srv://kendrickj5:james@models.bewjwp9.mongodb.net/?retryWrites=true&w=majority")
    db: Database = cluster['models']
    fs: GridFS = GridFS(db)
    return db, fs

# Returns True if the model exists in the database, False otherwise
def check_if_exists(name):
    db, fs = connect_to_db()
    return fs.exists({"_id": name})


# Saves the model to the database and removes the previous version if it exists
def save_model_to_db(model, name, file_exists, last_updated):
    model.save(name)

    db, fs = connect_to_db()

    if file_exists:
        fs.delete(name)
    
    with open(name, 'rb') as f:
        fs.put(f, _id=name, last_updated=last_updated)
    os.remove(name)


# Returns the model from the database
def get_model_from_db(name):
    db, fs = connect_to_db()
    file = fs.find_one({"_id": name}).read()
    with open(name, 'wb') as f:
        f.write(file)
    model = tf.keras.models.load_model(name)
    os.remove(name)
    return model


# Returns the last_updated field from the database
def get_last_updated(name):
    db, fs = connect_to_db()
    file = fs.find_one({"_id": name})
    return file.last_updated

# Serializes and saves the scaler to the database
def save_scaler_to_db(name: str, scaler: StandardScaler) -> None:
    db, fs = connect_to_db()
    collection = db['scalers']
    serialized_scaler = pickle.dumps(scaler)
    collection.insert_one({"_id": name, "scaler": serialized_scaler})

# Deserializes and returns the scaler from the database
def get_scaler_from_db(name: str) -> StandardScaler:
    db, fs = connect_to_db()
    collection = db['scalers']
    serialized_scaler = collection.find_one({"_id": name})
    scaler = pickle.loads(serialized_scaler['scaler'])
    return scaler