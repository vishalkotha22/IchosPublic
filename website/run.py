from PIL import ImageOps
import flask
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
from Preprocessing import wav_to_spectrogram, get_sli_features, get_feature_helper
import streamlit as st

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/alzheimers', methods=['GET', 'POST'])
def alzheimers():
   def load_model():
      pickle_in = open('alzheimers_model.pickle', 'rb')
      classifier = pickle.load(pickle_in)
      return classifier

   def inference(file):
      img = wav_to_spectrogram(file)
      model = load_model()
      mage = ImageOps.fit(image_data, (295, 295), Image.ANTIALIAS)
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

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))

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
