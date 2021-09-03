# import sys
#
# sys.path.insert(0, "Congressional-App-Challenge-2021/utils")

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa as lb
from pydub import AudioSegment
import wave
# from utils.Preprocessing import respiratory_preprocess
from scipy.io import wavfile


@st.cache(allow_output_mutation=True)
def get_model():
    model = tf.keras.models.load_model('models/respiratory_model.pb')
    return model


# Helper method for respiratory_preprocess
def get_feature_helper(path):
    soundArr, sample_rate = lb.load(path)
    mfcc = lb.feature.mfcc(y=soundArr, sr=sample_rate)
    cstft = lb.feature.chroma_stft(y=soundArr, sr=sample_rate)
    mSpec = lb.feature.melspectrogram(y=soundArr, sr=sample_rate)
    return mfcc, cstft, mSpec


def respiratory_preprocess(vid_file_path):
    a, b, c = get_feature_helper(vid_file_path)
    mfcc = np.array([a])
    cstft = np.array([b])
    mspec = np.array([c])
    return {"mfcc": mfcc, "croma": cstft, "mspec": mspec}


def app():
    st.title('Respiratory Diseases')
    file = st.file_uploader("Please upload an audio recording", type=["wav"])
    if file is not None:
        data = respiratory_preprocess(file)
        st.write(str(data))
        # inference stuff here (display the audio file and run the model)
        pass
