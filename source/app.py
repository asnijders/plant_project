from flask import Flask, render_template, request, send_from_directory
import os
import requests
from ultralytics import YOLO
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Flask instance
app = Flask(__name__)

# Load pickled YOLO model
# if os.path.isfile('model.pkl'):
#     model = pickle.load(open('model.pkl', 'rb'))
# else:
model = YOLO('yolov8n.pt')

# Set homepage for web-app. Prompts Flask app to run index.html as found under .\templates at localhost:5000
# This page provides functionality for user to upload an image file and send it through an HTTP POST request to the Flask app.
@app.route('/')
def home():
    return render_template('index.html')

# Define predict endpoint. This function will be ran as soon as a user makes a POST request via the upload feature.
@app.route('/predict', methods=['POST'])
def predict():
    print('Running predict')

    if request.method == 'GET':
        return render_template('index.html', msg='')

    image = request.files['file'] # load image from flask user upload
    img = Image.open(image)     # convert to PIL object
    results = model(source=img, stream=False) # do inference on image with YOLO
    
    im_array = results[0].plot() # extract prediction from YOLO results object
    im = Image.fromarray(im_array[..., ::-1])  # load result array into PIL again

    print(os.getcwd())
    f_path = 'dynamic/results.jpg'
    im.save(f_path)  # save image to local folder
    return send_from_directory('dynamic', 'results.jpg') # serve image with added bounding boxes back to user
    # return render_template('image_render.html', image=f_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
