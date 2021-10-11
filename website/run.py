from PIL import ImageOps
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

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/alzheimers', methods=['GET', 'POST'])
def alzheimers():
   def wav_to_spectrogram(file):
      sound = AudioSegment.from_wav(file)
      sound = sound.set_channels(1)
      sound.export("path.wav", format="wav")
      sample_rate, samples = wavfile.read("path.wav")
      frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
      # plt.ylabel('Frequency [Hz]')
      # plt.xlabel('Time [sec]')
      # plt.show()
      image = spectrogram
      io_buf = io.BytesIO()
      fig.savefig(io_buf, format='raw', dpi=36)
      io_buf.seek(0)
      img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                           newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
      io_buf.close()
      # print(img_arr.shape)
      return img_arr

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

   if flask.request.method == 'GET':
      return render_template('alzheimers.html')
   if flask.request.method == 'POST':
      

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return "file uploaded succesfully"

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