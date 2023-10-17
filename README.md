# P5 - Endangered butterfly species in Denmark/Europe

## Prerequisites:
- python3.7
- git lfs


## Installing python3.7
Run the following commands:

### Linux
```bash
sudo apt update && sudo apt install software-properties-common && sudo add-apt-repository ppa:deadsnakes/ppa && 
sudo apt install python3.7 && sudo apt install python3.7-distutils
```


### Windows
```powershell
winget install python.python.3.7
```
## Installing Git Large File Storagesystem

### Linux
```bash
sudo apt install git-lfs
# run inside project folder
git lfs fetch
```
### Windows
todo 🤷


## Installing requirements
```bash
pip install -r requirements.txt
```

## Tests
Make sure to enable Pytest as default test tool in your desired IDE.
Naming scripts/directories and their respective tests should match the following pattern:
 - core
   - /data
     - script.py
 - test
   - /test_data
     - test_script.py

Enter test folder
```bash
cd test
```

Run pytest
```bash
pytest
```

## Training a YOLOv8 model
Read README(trainyolov8model).MD

## Using a YOLOv8 model
Read README(useyolov8model).md