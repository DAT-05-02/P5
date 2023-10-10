import tensorflow as tf
import csv
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os
from core.data.feature import lbp, make_square_with_bb
import re

dir = os.path.dirname(__file__)
filePath = dir + "core\leopidotera-dk.csv"
filePath = (filePath.replace('\\', "/")).replace("train", "")

imageWidth = 416
imageHeight = 416


#Variables for storing images and labels
imageArr = []
imageLabels = []
labels = []

#important for the training output
butterflySpecies = []


#open csv file C:/Users/My dude/PycharmProjects/P5/core/leopidotera-dk.csv
file = open(filePath, 'r')


#row[2] is the picture url row[7] is species
#csvfile = csv.reader(file)
csvfile = pd.read_csv(file)

#skip the first row of the csv file
#itercsv = iter(csvfile)
#next(itercsv)


imgPath = (dir.replace("train", "")).replace("\\", "/") + "core/image_db"
images = os.listdir(imgPath)

print(len(images))

def asyncImageProcessing(row):
    image = Image.open(imgPath + "/" + images[index])
    print("hello ", index, "\n")

    image = Image.fromarray(lbp(image, radius=17))
    image = make_square_with_bb(image, mode="L", fill_color=0)
    image = np.array(image)
    #Hver pixl får en værdi mellem 0 og 1
    image = image/255.0

    imageArr.append(image)
    imageLabels.append(row[7])
    if row[7] not in butterflySpecies:
        butterflySpecies.append(row[7])

    index += 1


print(len(csvfile))

with ThreadPoolExecutor(4) as executer:
    _






model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imageHeight, imageWidth, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(butterflySpecies))


])

model.summary()

customOptimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(
    customOptimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#Prints the fucking ImageNames and imageLabels

imageLabels = np.array(imageLabels)




#convert standard python image list to numpy array
imageArr = np.array(imageArr)


#We enumerate over the butterfly species and get the labels out, which we put into label_to_index
label_to_index = {label: index for index, label in enumerate(butterflySpecies)}
#We map every label to their corresponding value in interger_labels
integer_labels = [label_to_index[label] for label in imageLabels]


#converts the lables to int32, so they can be used for model.fit
integer_labels = np.array(integer_labels, dtype=np.int32)


#15% data til test of validering
rows = imageArr.shape[0]
trainSize = int(rows*0.85)
imageArrTrain = imageArr[0:trainSize]
imageArrVal = imageArr[trainSize:]

rows = integer_labels.shape[0]
labelArrTrain = integer_labels[0:trainSize]
labelArrVal = integer_labels[trainSize:]



#mangler labels
model.fit(imageArrTrain, labelArrTrain, epochs=10, shuffle=True)


model.evaluate(imageArrVal, labelArrVal, verbose=2)