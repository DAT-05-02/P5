import datetime
from pprint import pprint

import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# TEMP IMPORTS
from PIL import Image
from matplotlib import cm

modelPath: str = "modelcheckpoint/"
fullModelPath: str = modelPath + "model.ckpt"

class Model:
    def __init__(self,
                 df: pd.DataFrame):
        self.df = df
        self.model = self._create_model()

    def _create_model(self, size=(416, 416), depth=1):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size[0], size[1], depth)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
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
        img_arr, lbl = self.img_with_labels()
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            custom_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
            #run_eagerly=True
        )
        
        # Checking if model allready exist.
        if os.path.exists(modelPath):
            # Loading the existing weights.
            self.model.load_weights(fullModelPath)

        # Print the model print("creating image of model: ") tf.keras.utils.plot_model(self.model, 'C:/Users/My
        # dude/Pictures/Saved Pictures/model.png', show_shapes=True, show_layer_names=True) print("created ")

        # We enumerate over the butterfly species and get the labels out, which we put into label_to_index
        label_to_index = {label: index for index, label in enumerate(set(lbl))}
        # We map every label to their corresponding value in integer_labels
        integer_labels = [label_to_index[label] for label in lbl]

        # converts the labels to int32, so they can be used for model.fit
        integer_labels = np.array(integer_labels, dtype=np.int32)

        # 15% of data used for testing
        rows = img_arr.shape[0]
        train_size = int(rows * 0.85)
        image_arr_train = img_arr[0: train_size]
        image_arr_val = img_arr[train_size:]

        label_arr_train = integer_labels[0: train_size]
        label_arr_val = integer_labels[train_size:]
        
        # logging training data
        tensorboard_training_image = np.reshape(image_arr_train/255, (-1, 416, 416, 1))
        
        data_log = "logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(data_log)
        
        with file_writer.as_default():
            tf.summary.image("Training data", tensorboard_training_image, step=0)
            
        print("SAVED TRAINBING IMAGE!!!")
            
        return
        
        # creating callbacks
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = fullModelPath, 
            save_weights_only = True, 
            verbose = 1)

        # mangler labels
        history = self.model.fit(
            image_arr_train,
            label_arr_train,
            validation_split=0.33,
            epochs=epochs,
            shuffle=True,
            batch_size=2,
            callbacks=[tensorboard_callback, cp_callback])

        res = self.model.evaluate(image_arr_val, label_arr_val, verbose=2)
        pprint(res)

        # sumarize history for accuracy
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
