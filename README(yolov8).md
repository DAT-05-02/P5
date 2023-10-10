## 1. Navigate to yolo folder and create an environment.
cd yolo
python -m venv env

## 2. Activate virtual environment
#On Windows
env/Scripts/activate

#On Linux / macOS
source env/bin/activate

## 3. Install pretrained model "yolov8n.pt"
#image detection
yolo task=detect mode=predict model=yolov8n.pt source="path/to/image.png" 

## 4. Run roboflow.ipynb

## 5. Insert config file in roboflow folder.
#Copy/paste following into a yaml file


path:
train: train/images
test: test/images
val: valid/images

#Classes
nc: 1 # replace based on your dataset's number of classes

#Class names
#replace all class names with your own classes' names
names: ['Lepidoptera']

## 6. Train model
yolo task=detect mode=train model=yolov8n.pt data=path/to/config.yaml epochs=10 imgsz=640

## 7. Do inference with the new model
yolo task=detect mode=predict model="runs/train/weights/best.pt" source="image.png"

## 8. Evaluate the model's performance(optional)
yolo val model=yolov8n.pt data=path/to/config.yaml batch=1 imgsz=640

## 9. Test out the model
yolo detect predict model=path/to/best.pt source='path/to/image.jpg'
