import tensorflow as tf
import pandas as pd
import csv
import numpy as np
import PIL
import PIL.Image as Image
import os

dir = os.path.dirname(__file__)
filePath = dir + "core\leopidotera-dk.csv"
filePath = (filePath.replace('\\', "/")).replace("Train", "")

imageWidth = 128
imageHeight = 128

imagesDone = 0

#Variables for storing images and labels
imageArr = []
imageLabels = []
labels = []

#important for the training output
butterflySpecies = []


#open csv file C:/Users/My dude/PycharmProjects/P5/core/leopidotera-dk.csv
file = open(filePath, 'r')


#row[2] is the picture url row[7] is species
csvfile = csv.reader(file)


#skip the first row of the csv file
itercsv = iter(csvfile)


next(itercsv)


for row in itercsv:
    #Gett the images
    imgPath = (dir.replace("Train", "")).replace("\\", "/") + "core/image_db/" + str(imagesDone) + ".jpg"
    image = PIL.Image.open(imgPath)
    image = image.convert("L")
    image = image.resize((imageWidth, imageHeight))
    #Billederne laves til numpy arrays
    image = np.array(image)
    #Hver pixl får en værdi mellem 0 og 1
    image = image/255.0
    #Billedet reshpapes til den shape som forventes af modellen
    image = image.reshape(imageWidth, imageHeight, 1)
    imageArr.append(image)
    imageLabels.append(row[7])
    if row[7] not in butterflySpecies:
        butterflySpecies.append(row[7])
    imagesDone += 1



model = tf.keras.models.Sequential([
    #input layer is image width, height, 1 for grayscale
    #tf.keras.layers.Input(shape=(imageWidth, imageHeight, 1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(imageHeight, imageWidth, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.7),
    #Output layer, the different species:
    tf.keras.layers.Dense(len(butterflySpecies))
])

model.summary()

model.compile(
    tf.keras.optimizers.Adam(0.191),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

imageLabels = np.array(imageLabels)

#convert standard python image list to numpy array
imageArr = np.array(imageArr)


#We enumerate over the butterfly species and get the labels out, which we put into label_to_index
label_to_index = {label: index for index, label in enumerate(butterflySpecies)}
#We map every label to their corresponding value in interger_labels
integer_labels = [label_to_index[label] for label in imageLabels]


#converts the lables to int32, so they can be used for model.fit
integer_labels = np.array(integer_labels, dtype=np.int32)


#----------------------------------


#15% data til test of validering
#resten til

rows = imageArr.shape[0]
trainSize = int(rows*0.85)
imageArrTrain = imageArr[0:trainSize]
imageArrVal = imageArr[trainSize:]

rows = integer_labels.shape[0]
labelArrTrain = integer_labels[0:trainSize]
labelArrVal = integer_labels[trainSize:]

#mangler labels
model.fit(imageArrTrain, labelArrTrain, epochs=10)

print(len(imageArrVal))
print("validation testing")
model.evaluate(imageArrVal, labelArrVal, verbose=2)