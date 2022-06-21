import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import base64
from flask import jsonify
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
from werkzeug.utils import secure_filename
import numpy as np
import cv2


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
#IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = 'uploads'
model = core.Model.load('model_weights_new.pth', ['matrix'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    image  = cv2.imread(file)
    predictions = model.predict(image)
    labels, boxes, scores = predictions
    filtered_indices = np.where(max(scores))
    filtered_boxes = boxes[filtered_indices]
    num_list = filtered_indices[0].tolist()
    filtered_labels = [labels[i] for i in num_list]
    xmin = filtered_boxes[0, 0].item()
    ymin = filtered_boxes[0, 1].item()
    xmax = filtered_boxes[0, 2].item()
    ymax = filtered_boxes[0, 3].item()
    xmin = round(xmin)
    ymin = round(ymin)
    xmax = round(xmax)
    ymax = round(ymax)
    roi = image[ymin:ymax, xmin:xmax]
    #roi=cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    #destination = "/".join([target, 'matrix.jpg'])
    cv2.imwrite('matrix.jpg', roi)
    with open('matrix.jpg', "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    #destination_1 = "/".join([target, 'encode.txt'])
    with open('encoded.txt', "wb") as file:
        file.write(encoded_string)

    response = {
         "image":encoded_string.decode(),
         "message":"Image is BASE 64 encoded"        
            }

    return response

app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(threaded=False)