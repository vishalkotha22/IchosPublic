from PIL import ImageOps
import flask
from PIL.Image import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import librosa as lb
import numpy as np
import pickle
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
import io
import matplotlib.pyplot as plt
import cv2
from Preprocessing import wav_to_spectrogram, get_sli_features, get_feature_helper

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/alzheimers', methods=['GET', 'POST'])
def alzheimers():
    def load_model():
        pickle_in = open('models/alzheimers_model.pickle', 'rb')
        classifier = pickle.load(pickle_in)
        return classifier

    def inference(file):
        img = wav_to_spectrogram(file)
        model = load_model()
        image = ImageOps.fit(img, (295, 295), Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(295, 295), interpolation=cv2.INTER_CUBIC)) / 255.
        x = img_resize.flatten()
        img_reshape = x[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

    if flask.request.method == 'POST':
        pass
    return render_template('alzheimers.html')


@app.route('/alzhiemersuploader', methods=['GET', 'POST'])
def upload_file1():
    def load_model():
        pickle_in = open('models/alzheimers_model.pickle', 'rb')
        classifier = pickle.load(pickle_in)
        return classifier

    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename('file.wav'))
        img = wav_to_spectrogram(f)
        model = load_model()
        #image = ImageOps.fit(img, (295, 295))
        image = np.asarray(img)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(295, 295), interpolation=cv2.INTER_CUBIC)) / 255.
        x = img_resize.flatten()
        img_reshape = x[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        if prediction[0] < 0.5:
            return 'You do not have Alzheimers'
        else:
            return 'You do have Alzheimers'


@app.route('/respiratoryuploader', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename('respiratoryfile.wav'))

        return prediction


@app.route('/sliuploader', methods=['GET', 'POST'])
def upload_file3():
    def load_model():
        pickle_in = open('models/updated_forest_sli.pkl', 'rb')
        classifier = pickle.load(pickle_in)
        return classifier

    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename('slifile.wav'))
        features = get_sli_features('slifile.wav')
        model = load_model()
        prediction = model.predict(features)
        if prediction < 0.5:
            return "You do not have SLI"
        else:
            return "You may have SLI"



@app.route('/sli')
def sli():
    return render_template('sli.html')


@app.route('/respiratory')
def respiratory():
    return render_template('respiratory.html')


@app.route('/generic')
def generic():
    return render_template('generic.html')


if __name__ == '__main__':
    app.run()
