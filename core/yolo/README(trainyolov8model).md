## 1. Install Packages
!pip install ultralytics==8.0.134

## 2. Install pretrained model "yolov8n.pt"
#image detection
yolo task=detect mode=predict model=yolov8n.pt source="path/to/image.png" 

## 3. Run roboflow.ipynb

## 4. Train model
#Locate AppData\Roaming\Ultralytics\settings.yaml and change the dataset directory if necessary

#Train new model
yolo task=detect mode=train model=path/to/model.yaml data=path/to/data.yaml epochs=10 imgsz=640

#Or train new model based on pre-trained model
yolo task=detect mode=train model=path/to/model.pt data=path/to/data.yaml epochs=10 imgsz=640

#when training a new model from scratch type in yolov8*.yaml where * is either n/s/m/l/x depending on the desired size of the neural network.

## 5. Do inference with the new model
yolo task=detect mode=predict model="runs/train/weights/best.pt" source="image.png"

## 6. Evaluate the model's performance
yolo val model=path/to/data.yaml.pt data=path/to/config.yaml batch=1 imgsz=640

## 7. Test out the model
yolo detect predict model=path/to/best.pt source='path/to/image.jpg'

## 8. Save the model as another data type.
#replace the format="onnx" to specify another file type if desired
yolo export model=path/to/best.pt format=onnx