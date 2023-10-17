import tensorflow as tf
import pandas as pd
import numpy as np


class Model:
    def __init__(self, df: pd.DataFrame):
        self.model = None
        self.df = df

    def create_model(self, outputs, size=(416, 416), depth=1):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], depth)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(outputs)
        ])
        self.model = model
        # for summary use {objectname}.model.summary

    def model_compile_fit_evaluate(self, images, image_labels, lr=0.001):
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            custom_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
            run_eagerly=True
        )

        imageLabels = np.array(image_labels)

        imageArr = np.array(images)

        # We enumerate over the butterfly species and get the labels out, which we put into label_to_index
        label_to_index = {label: index for index, label in enumerate(set(imageLabels))}
        # We map every label to their corresponding value in integer_labels
        integer_labels = [label_to_index[label] for label in imageLabels]

        # converts the labels to int32, so they can be used for model.fit
        integer_labels = np.array(integer_labels, dtype=np.int32)

        # 15% of data used for testing
        rows = imageArr.shape[0]
        train_size = int(rows * 0.85)
        image_arr_train = imageArr[0: train_size]
        image_arr_val = imageArr[train_size:]

        labelArrTrain = integer_labels[0: train_size]
        labelArrVal = integer_labels[train_size:]

        # mangler labels
        self.model.fit(image_arr_train, labelArrTrain, epochs=10, shuffle=True)

        self.model.evaluate(image_arr_val, labelArrVal, verbose=2)