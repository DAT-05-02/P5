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

    def _create_model(self, size=(256, 256), depth=3):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(size[0], size[1], depth)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(len(self.df["species"].unique()))
        ])

        return model
        # for summary use {objectname}.model.summary

    def img_with_labels(self):
        # df = self.df.assign(image=lambda x: Image.open(str(x["path"])))
        self.df['image'] = ""
        arr = []
        for index, row in self.df.iterrows():
            if row['lbp'] and row['lbp'] != "":
                arr.append(np.array(Image.open(row['lbp']).convert("L")).tolist())

        return np.array(arr), self.df['species']

    def model_compile_fit_evaluate(self, lr=0.001, epochs=10):
        #img_arr, lbl = self.img_with_labels()
        dataset = tf.keras.utils.image_dataset_from_directory("image_db", labels="inferred")
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            custom_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Print the model print("creating image of model: ") tf.keras.utils.plot_model(self.model, 'C:/Users/My
        # dude/Pictures/Saved Pictures/model.png', show_shapes=True, show_layer_names=True) print("created ")

        # 15% of data used for testing
        train_ds, test_ds = tf.keras.utils.split_dataset(dataset, left_size=0.85, shuffle=True)


        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # mangler labels
        history = self.model.fit(
            train_ds,
            epochs=epochs)

        self.model.save("latest.keras")
        res = self.model.evaluate(test_ds, verbose=2)
        pprint(res)

        # sumarize history for accuracy
        """
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        """
