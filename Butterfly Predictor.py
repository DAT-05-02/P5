import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import pandas as pd


model = tf.keras.models.load_model("core/latest.keras")
species_dataset_path = "core/leopidotera-dk.csv"
df = pd.read_csv(species_dataset_path)
species = df["species"].unique()
print("number of species", len(species))

def predict_image(image):
    image = cv2.resize(image, (128, 128)) # change to match the input_shape of the trained model
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    print("predicted class", predicted_class)
    
    predicted_species = species[predicted_class]
    return predicted_species

def process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        predicted_species = predict_image(image)
        label_result.config(text=f"Predicted species: {predicted_species}")

def use_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predicted_species = predict_image(frame)
        cv2.putText(frame, f"Predicted species: {predicted_species}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("butterfly species recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
# GUI application
root = tk.Tk()
root.title("Butterfly Species Recognition")

btn_image = tk.Button(root, text="Select image", command=process_image)
btn_image.pack(pady=10)

btn_camera = tk.Button(root, text="Use Camera", command=use_camera)
btn_camera.pack(pady=5)

label_result = tk.Label(root, text="")
label_result.pack(pady=5)

root.mainloop()

