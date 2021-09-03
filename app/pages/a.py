import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image, ImageOps
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pydub import AudioSegment

def get_model():
    pickle_in = open('alzheimers_model.pickle', 'rb')
    classifier = pickle.load(pickle_in)
    return classifier

def predict(image_data):
    model = get_model()
    image = ImageOps.fit(image_data, (295, 295), Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(295, 295), interpolation=cv2.INTER_CUBIC)) / 255.
    x = img_resize.flatten()
    img_reshape = x[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def app():
    # Temporarily using spectogram images
    st.title('Alzheimers Disease')
    file = st.file_uploader("Please upload an audio file", type=["wav"])

    if file is not None:
        sound = AudioSegment.from_wav(file)
        sound = sound.set_channels(1)
        sound.export("path.wav", format="wav")
        sample_rate, samples = wavfile.read("path.wav")
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

        plt.pcolormesh(times, frequencies, spectrogram)
        plt.imshow(spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        image = spectrogram
        #image = Image.open(spectrogram)
        #st.image(image, use_column_width=True)
        prediction = predict(image)
        num = np.amax(prediction)
        if num >=0.50:
            st.write("Alzheimer's Disease has been diagnosed with: " + str(round(num*100, 3)) + "% Confidence.")
        else:
            st.write("Alzheimer's disease has not been detected with: " + str(round((1.00-num) * 100, 3)) + "% Confidence.")
