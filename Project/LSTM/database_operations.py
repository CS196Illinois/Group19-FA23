import os
import tensorflow as tf
from pymongo.mongo_client import MongoClient
from gridfs import GridFS

# Returns True if the model exists in the database, False otherwise
def check_if_exists(name):
    cluster = MongoClient("mongodb+srv://<User ID>:<password>@models.bewjwp9.mongodb.net/?retryWrites=true&w=majority")
    db = cluster['models']
    fs = GridFS(db)

    return fs.exists({"_id": name})


# Saves the model to the database and removes the previous version if it exists
def save_model_to_db(model, name, file_exists, last_updated):
    model.save(name)

    cluster = MongoClient("mongodb+srv://<User ID>:<password>@models.bewjwp9.mongodb.net/?retryWrites=true&w=majority")
    db = cluster['models']
    fs = GridFS(db)

    if file_exists:
        fs.delete(name)
    
    with open(name, 'rb') as f:
        fs.put(f, _id=name, last_updated=last_updated)
    os.remove(name)


# Returns the model from the database
def get_model_from_db(name):
    cluster = MongoClient("mongodb+srv://<User ID>:<password>@models.bewjwp9.mongodb.net/?retryWrites=true&w=majority")
    db = cluster['models']
    fs = GridFS(db)
    file = fs.find_one({"_id": name}).read()
    with open(name, 'wb') as f:
        f.write(file)
    model = tf.keras.models.load_model(name)
    os.remove(name)
    return model


# Returns the last_updated field from the database
def get_last_updated(name):
    cluster = MongoClient("mongodb+srv://<User ID>:<password>@models.bewjwp9.mongodb.net/?retryWrites=true&w=majority")
    db = cluster['models']
    fs = GridFS(db)
    file = fs.find_one({"_id": name})
    return file.last_updated