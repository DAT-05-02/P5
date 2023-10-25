import datetime
from pprint import pprint

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
            #tf.keras.layers.Dense(len(self.df["species"].unique()), activation="softmax")
            tf.keras.layers.Dense(55, activation="softmax")
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
        evaluation_results = self.model.evaluate(self.test_dataset, verbose=2)
        print("Test Accuracy: {:.2f}%".format(evaluation_results[1] * 100))

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
        evaluation_results = self.model.evaluate(self.test_dataset, verbose=2)
        print("Test Accuracy: {:.2f}%".format(evaluation_results[1] * 100))

        species_labels = self.df["species"].tolist()

        for images, labels in self.test_dataset.take(num_samples):
            predictions = self.model.predict(images)

            plt.figure(figsize=(15, 6))

            for i in range(num_samples):
                true_label = species_labels[labels[i].numpy().argmax()]
                predicted_label = species_labels[predictions[i].argmax()]
                plt.subplot(1, num_samples, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
                plt.axis("off")

            plt.show() 

    def predict_and_show(self, image_path):
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(416, 416)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        prediction = self.model.predict(img_array)
        species_labels = self.df["species"].tolist()
        predicted_label = species_labels[np.argmax(prediction)]

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis("off")
        plt.show()
