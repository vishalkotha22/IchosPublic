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
import tensorflow as tf
import cv2

import matplotlib

from website.Preprocessing import *

matplotlib.use('Agg')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/alzhiemersuploader', methods=['GET', 'POST'])
def upload_file1():
    def load_model():
        pickle_in = open('models/alzheimers_model.pickle', 'rb')
        classifier = pickle.load(pickle_in)
        return classifier

    if request.method == 'POST':
        f = request.files['file']
        if not f.filename.endswith('.wav'):
            return 'Wrong File Type'
        f.save(secure_filename('file.wav'))
        # img = wav_to_spectrogram(f)
        model = load_model()
        # img = ImageOps.fit(img, (295, 295))
        '''img = np.asarray(img)
        plt.imshow(img)

        img_resize = (cv2.resize(img, dsize=(295, 295),
                      interpolation=cv2.INTER_CUBIC)) / 255

        plt.show()
        x = img_resize.flatten()
        img_reshape = x[np.newaxis, ...]'''
        img_resize = plotstft('file.wav')
        # img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        img_resize = img_resize[:, :, :3]
        # plt.imshow(img_resize)
        # plt.show()
        x = img_resize.flatten()
        print(img_resize.shape, x.shape)
        img_reshape = x[np.newaxis, ...]
        prediction = model.predict(img_reshape)

        if prediction[0] < 0.5:
            return render_template('results.html', data=[0, 'You do not have Alzheimers'])
        else:
            return render_template('results.html', data=[0, 'You may have Alzheimers'])


@app.route('/respiratoryuploader', methods=['POST'])
def upload_file2():
    if request.method == 'POST':
        f = request.files['file']
        if not f.filename.endswith('.wav'):
            return 'Wrong File Type'
        f.save(secure_filename('respiratoryfile.wav'))
        new_model = tf.keras.models.load_model('models/respiratory_model_v2')
        diag, conf = process_file('respiratoryfile.wav', new_model)
        return render_template('results.html',
                               data=[1, f'{diag} with {int(conf * 100)}% chance' if diag != 'Healthy' else 'Healthy'])


@app.route('/sliuploader', methods=['POST'])
def upload_file3():
    def load_model():
        pickle_in = open('models/updated_forest_sli.pkl', 'rb')
        classifier = pickle.load(pickle_in)
        return classifier

    if request.method == 'POST':
        f = request.files['file']
        if not f.filename.endswith('.wav'):
            return 'Wrong File Type'
        f.save(secure_filename('slifile.wav'))
        features = get_sli_features('slifile.wav')
        model = load_model()
        prediction = model.predict(features)
        print(prediction)
        if prediction[0] < 0.5:
            return render_template('results.html', data=[2, 'You do not have Specific Language Impairment'])
        else:
            return render_template('results.html', data=[2, 'You may have Specific Language Impairment'])


@app.route('/sli')
def sli():
    return render_template('sli.html')


@app.route('/respiratory')
def respiratory():
    return render_template('respiratory.html')


@app.route('/results')
def results():
    return render_template('results.html')


if __name__ == '__main__':
    app.run()
