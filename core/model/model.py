import datetime
from pprint import pprint
import os

import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Model:
    def __init__(self,
                 df: pd.DataFrame):
        self.df = df
        self.model = self._create_model()
        self.dataset = self._setup_dataset()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_model(self, size=(416, 416), depth=3):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(size[0], size[1], depth)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(len(os.listdir("image_db")), activation="softmax")
            #tf.keras.layers.Dense(57, activation="softmax")
        ])

        return model

    def _setup_dataset(self):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory="image_db", 
            labels="inferred", 
            label_mode="categorical", 
            image_size=(416, 416), 
            )
        return dataset

    def print_dataset_info(self):
        num_batches = self.dataset.cardinality()
        print(f"Number of batches in the dataset: {num_batches}")

        total_images = 0
        total_labels = 0

        for images, labels in self.dataset:
            total_images += images.shape[0]
            total_labels += labels.shape[0]

        print(f"Total images in the dataset: {total_images}")
        print(f"Total labels in the dataset: {total_labels}")

    def save(self, model_path="latest.keras"):
        self.model.save(model_path)

    def load(self, model_path="latest.keras"):
        self.model = tf.keras.models.load_model(model_path)

    def split_dataset(self, validation_split=0.15, test_split=0.15, shuffle=True):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory="image_db",
            labels="inferred",
            label_mode="categorical",
            image_size=(416, 416),
            validation_split=validation_split,
            subset="training",
            seed=1337 if shuffle else None,
            shuffle=shuffle
        )

        val_size = int(validation_split * len(dataset))
        test_size = int(test_split * len(dataset))
        train_size = len(dataset) - val_size - test_size

        self.train_dataset = dataset.take(train_size)
        remaining_dataset = dataset.skip(train_size)
        self.val_dataset = remaining_dataset.take(val_size)
        self.test_dataset = remaining_dataset.skip(val_size)

    def compile(self, lr=0.001):
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            custom_optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def fit(self, epochs=10):
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs
        )
        return history

    def evaluate(self):
        res = self.model.evaluate(self.test_dataset, verbose=2)
        pprint(res)

    def evaluate_and_print_predictions(self):
        species_labels = self.df["species"].tolist()

        true_labels = []
        predicted_labels = []

        for images, labels in self.test_dataset:
            predictions = self.model.predict(images)
            true_labels.extend([species_labels[label.numpy().argmax()] for label in labels])
            predicted_labels.extend([species_labels[pred.argmax()] for pred in predictions])

        print("Evaluation Summary:")
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            print(f"True Label: {true_label}\tPredicted Label: {predicted_label}")

    def evaluate_and_show_predictions(self, num_samples=5):
        species_labels = os.listdir("image_db")
    
        for images, _ in self.test_dataset:
            for i in range(images.shape[0]):
                img = images[i]
                img_array = tf.expand_dims(img, 0)
                prediction = self.model.predict(img_array)
                
                sorted_indices = np.argsort(prediction[0])[::-1]
                sorted_labels = [species_labels[i] for i in sorted_indices]
                sorted_confidences = [prediction[0][i] * 100 for i in sorted_indices]

                plt.figure(figsize=(8, 6))
                plt.imshow(img.numpy().astype("uint8"))

                label_str = "Predictions:\n"
                for label, confidence in zip(sorted_labels[:5], sorted_confidences[:5]):
                    label_str += f"{label}: {confidence:.2f}%\n"

                plt.title(label_str)
                plt.axis("off")
                plt.show()

    def predict_and_show(self, image_path):
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(416, 416)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            prediction = self.model.predict(img_array)
            species_labels = os.listdir("image_db")
            sorted_indices = np.argsort(prediction[0])[::-1]
            sorted_labels = [species_labels[i] for i in sorted_indices]
            sorted_confidences = [prediction[0][i] * 100 for i in sorted_indices]

            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            
            label_str = "Predictions:\n"
            for label, confidence in zip(sorted_labels[:5], sorted_confidences[:5]):
                label_str += f"{label}: {confidence:.2f}%\n"
            
            plt.title(label_str)
            plt.axis("off")
            plt.show()

