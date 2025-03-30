"""tfexample: A Flower / TensorFlow app."""

import os
import shutil
from urllib.parse import quote_plus

import keras
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import pymongo
from bson.binary import Binary

from settings import app_settings

HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 8

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"




def load_model(learning_rate: float = 0.001):

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(HEIGHT, WIDTH, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    #
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])

    return model

fds = None  # Cache FederatedDataset


def load_data(batch_size: int = BATCH_SIZE):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once


    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    client = pymongo.MongoClient("mongodb://%s:%s@%s" % (
            quote_plus(app_settings.mongo_username), quote_plus(app_settings.mongo_password), app_settings.mongo_host))
    db = client[app_settings.mongo_db]
    collection = db[app_settings.mongo_coll]

    def save_image(image_data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as file:
            file.write(image_data)

    for document in collection.find():
        dataset_type = document["dataset_type"]
        category = document["category"]
        filename = document["filename"]
        image_data = document["image"]
        folder_path = os.path.join(app_settings.temp_dir, dataset_type, category)
        file_path = os.path.join(folder_path, filename)
        save_image(image_data, file_path)

    train_generator = train_datagen.flow_from_directory(
        app_settings.train_dir,
        target_size=(HEIGHT, WIDTH),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True,
        classes={'Normal': 0, 'Viral Pneumonia': 1, 'Covid': 2}
    )

    test_generator = test_datagen.flow_from_directory(
        app_settings.test_dir,
        target_size=(HEIGHT, WIDTH),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False,
        classes={'Normal': 0, 'Viral Pneumonia': 1, 'Covid': 2}
    )

    return train_generator, test_generator