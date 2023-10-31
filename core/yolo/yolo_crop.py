from PIL import Image
import os, shutil
from core.yolo.load_img import load_img
from core.yolo.obj_det import obj_det


#def yolo_crop() -> None:
    #if not os.path.exists("runs/detect/predict/croppedimages"):
    #os.mkdir("runs/detect/predict/croppedimages")
    #dest_path = "runs/detect/predict/croppedimages"
    #dest_path = "temptest"
    #img_path = "runs/detect/predict"
    #txt_path = "runs/detect/predict/labels"

    #for path in os.scandir("../image_db"):
        #if path.is_dir():
            #print("0")
            #print(path)
            #temp_path = os.path.splitext(f"{path.name}")[0]
            #print(temp_path)
            #obj_det(temp_path)
            #print("1")
            # Save the image somewhere else where it will not be deleted.
            #for path in os.scandir(img_path):
                #if path.is_file():
                    #name = os.path.splitext(f"{path.name}")[0]
                    #if os.path.exists(f"{txt_path}/{name}.txt"):
                        #with Image.open(f"{path.path}") as img:
                            #corners = load_img(txt_path, img_path, path)
                            # img1 = img.crop((left, top, right, bottom))
                            #img1 = img.crop(corners)

                            #img1.save(fp=f"{dest_path}/{path.name}")
            #shutil.rmtree(f"{img_path}")

def yolo_crop(img, model):

    xywhn = obj_det(img, model)
    corners = load_img(img, xywhn)
    img1 = img.crop(corners)
    return img1

