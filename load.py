import os
import pymongo
from bson import Binary

# MongoDB connection
client = pymongo.MongoClient("mongodb://testuser:seCretPassword@localhost:27017")
db = client["fed-learning"]
collection = db["xrays"]

data_dir = "../Covid19-dataset"
dataset_types = ["train", "test"]

for dataset_type in dataset_types:
    dataset_path = os.path.join(data_dir, dataset_type)
    for category in ["Covid", "Normal", "Viral Pneumonia"]:
        category_path = os.path.join(dataset_path, category)

        if os.path.exists(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)

                with open(file_path, "rb") as image_file:
                    image_data = image_file.read()

                # Insert into MongoDB
                document = {
                    "filename": filename,
                    "dataset_type": dataset_type,
                    "category": category,
                    "image": Binary(image_data)
                }
                collection.insert_one(document)

print("Images successfully uploaded to MongoDB.")