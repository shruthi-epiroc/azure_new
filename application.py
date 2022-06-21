import os
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import cv2
from image_class_api import crop
import base64
from app import app
UPLOAD_FOLDER = '/outputs'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/image', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        file.save(secure_filename(file.filename))
        results = crop(file.filename)
        resp = jsonify(
            {'message': 'File successfully uploaded', 'result': results})
        
        resp.status_code = 201
        return results

    else:
        resp = jsonify(
            {'message': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        image = cv2.imread(f)
        results = type(image)
        #f.save(secure_filename(f.filename))
        return results


if __name__ == "__main__":
    app.run()
