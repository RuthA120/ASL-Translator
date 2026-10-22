# ASL Tracker Prototype

## Setup Instructions

Follow these steps to set up the project locally on your machine:

### 1. Clone the Repository

### 2. Set Up the Environment
``` python
# Create the virtual environment
python3 -m venv asl_env

# Download requirments
pip install -r requirements.txt 

# Activate virtual environment (Mac/Linux)
source asl_env/bin/activate

# Activate virtual environment (Windows PowerShell)
# .\asl_env\Scripts\Activate
```

### 3. Run predict_live.py
``` python
python predict_live.py
```
- A window will open up on your computer and to exit out you can click 'q'
- There are currently 4 letter signs the model can detect: H, R, T, U


## Overview on how this works
When running the `collect_data.py` file, Google MediaPipe will take in the video feed and convert the raw pixels of the hand for each corresponding sign and save the precise 3D landmark coordinates into a CSV file `asl_data.csv`.

Taking the coordinates, we can train a classifier model (currently using a RandomForest model but we can change this). The model will recognize the signs and associate them with each label/sign. 

When running the `predict_live.py` file, the webcam will then feed in the live frames into Google MediaPipe which will send the 3D landmark coordinates to the trained model and then output the predicted letter on the screen.
