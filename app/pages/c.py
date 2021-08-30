import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from pydub import AudioSegment
import wave
from scipy.io import wavfile

@st.cache(allow_output_mutation=True)
def get_model():
    model = tf.keras.models.load_model('models/respiratory_model.pb')
    return model

def app():
    st.title('Respiratory Diseases')
    file = st.file_uploader("Please upload an audio recording", type=["wav"])
    if file is None:
        pass
    else:
        # inference stuff here (display the audio file and run the model)
