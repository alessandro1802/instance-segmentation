import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import tempfile
import base64
import io

from utils import *

configureGPU()

model = tf.keras.models.load_model('./models/model.h5')

BATCH_SIZE = 1
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
SEED = 123

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index_page():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Create temp dir for input image
    temp_dir = tempfile.TemporaryDirectory()
    
    # Get the input image
    img_data = request.files['img']
    img_byte_string = img_data.read()
    img_array = np.frombuffer(img_byte_string, dtype = np.uint8)
    arr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)[:,:,::-1]
    
    # Save it to a file
    cv2.imwrite(temp_dir.name + "/image.jpg", arr)
    
    # Get the prediction
    DATA_PATH = temp_dir.name
    ds = getDataset(DATA_PATH, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, SEED)
    
    fig = predict(model, ds)
    
    # Convert plot to image
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format = 'JPEG')

    img = Image.open(img_buf)
    
    # Save the image to buffer
    mem_bytes = io.BytesIO()    
    img.save(mem_bytes, 'JPEG')
    
    # Process it to display
    mem_bytes.seek(0)
    img_base64 = base64.b64encode(mem_bytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)
    
    img_buf.close()
    mem_bytes.close()
    temp_dir.cleanup()
    
    return render_template("index.html", image = uri)

if __name__ == '__main__':
    app.run(host = "localhost", port = 8080)