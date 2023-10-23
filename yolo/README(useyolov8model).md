## Object detection with YOLOv8
Run the following command in cli to use object detection.
yolo detect predict model=yolo/medium250e.pt source='path/to/source'

Where the model and source are the arguments.
Replace model path and source path to specify which model and source to use.

### Object detection with arguments.
To enable GPU add 'device=0' at the end.

To get coordinates for the object detection add 'save_txt' at the end of the command.
This will result in a txt file for every source with the format of 5 values for every object detected.
"Class, x_coordinate, y_coordinate, width, height"
See examplecoordinates.jpg for an example.

https://docs.ultralytics.com/usage/cfg/#predict
The link is an overview of arguments accepted by the predict command, as well as the default arguments.


## Using YOLO autocrop
yolo detect predict model="model" source="path/to/source" save_txt=True boxes=False

The argument save_txt has to be set to "True" and the argument boxes set to "False"

Saved image/s will be saved in "croppedimages" folder and the images will have the same name as the original image name.