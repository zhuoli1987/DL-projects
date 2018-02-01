from flask import Flask, jsonify, request
import numpy as np
import PIL
from PIL import Image
from utils import *
import sys

app = Flask(__name__)

@app.route('/classify', methods=["POST"])
def classify_image():
    """
    Algorithm  that accepts a file path to an 
    image and first determines whether the image 
    contains a human, dog, or neither. 
    """
    result = ""

    # request image
    image = request.files['file']
    image = Image.open(image)

    # Try to detect whether there is a dog
    # or a human in the given image
    is_dog = dog_detector(image)
    is_human = face_detector(image)

    if is_dog or is_human:
        # Dog breed prediction using ResNet50
        dog_name_prediction = Resnet50_predict_breed(image)
        
        if is_dog:
            result = {'Species':'dog', 'breed':dog_name_prediction}
        else:
            result = {'Species':'human', 'breed':dog_name_prediction}

    return jsonify(result)

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=4444)
