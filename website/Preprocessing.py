import os

import librosa as lb
import numpy as np
import cv2
import pickle
from PIL import Image, ImageOps
from pydub.silence import split_on_silence
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pydub import AudioSegment
import io
from xgboost import XGBClassifier
import speech_recognition as sr
import re
from scipy import interp

class_names = {0: 'URTI', 1: 'Healthy', 2: 'Asthma', 3: 'COPD', 4: 'LRTI', 5: 'Bronchiectasis',
              6: 'Pneumonia', 7: 'Bronchiolitis'}

class_names_full = {0: 'Upper Respiratory Tract Infection', 1: 'Healthy', 
                    2: 'Asthma', 3: 'Chronic Obstructive Pulmonary Disease', 
                    4: 'Lower Respiratory Tract Infections', 5: 'Bronchiectasis',
                    6: 'Pneumonia', 7: 'Bronchiolitis'}

# Helper method for respiratory_preprocess
def get_feature_helper(path):
    soundArr,sample_rate=lb.load(path)
    mfcc=lb.feature.mfcc(y=soundArr,sr=sample_rate)
    cstft=lb.feature.chroma_stft(y=soundArr,sr=sample_rate)
    mSpec=lb.feature.melspectrogram(y=soundArr,sr=sample_rate)
    return mfcc,cstft,mSpec


def respiratory_preprocess(vid_file_path):
    a, b, c = get_feature_helper(vid_file_path)
    mfcc = np.array([a])
    cstft = np.array([b])
    mspec = np.array([c])
    return {"mfcc":mfcc,"croma":cstft,"mspec":mspec}


def get_sli_features(wav_file):
    r = sr.Recognizer()

    def get_large_audio_transcription(path):
        sound = AudioSegment.from_file(path)
        chunks = split_on_silence(sound,
                                  min_silence_len = 500,
                                  silence_thresh = sound.dBFS-14,
                                  keep_silence=500,
                                  )
        folder_name = "audio-chunks"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        whole_text = ""
        for i, audio_chunk in enumerate(chunks, start=1):
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            with sr.AudioFile(chunk_filename) as source:
                audio_listened = r.record(source)
                try:
                    text = r.recognize_google(audio_listened)
                except sr.UnknownValueError as e:
                    print("Error:", str(e))
                else:
                    text = f"{text.capitalize()}. "
                    print(chunk_filename, ":", text)
                    whole_text += text
        return whole_text

    text = get_large_audio_transcription(wav_file)
    split = text.split(' ')
    child_TNW = text.count(' ')
    repetition = 0
    for i in range(len(split)-1):
        temp = 0
        while split[i] == split[i+1]:
            temp = temp+1
            i = i+1
        repetition = temp+1
    fillers = text.count(' um ') + text.count(' uh ')

    def syllable_count(word):
        if len(word) <= 0:
            return 0
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count
    total_syl = sum([syllable_count(word) for word in split])
    average_syl = total_syl / len(split)
    num_dos = text.count('do')
    return [[child_TNW, repetition, fillers, average_syl, -1, total_syl, num_dos]]


def wav_to_spectrogram(file):
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    sound.export("path.wav", format="wav")
    sample_rate, samples = wavfile.read("path.wav")
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    fig = plt.figure()
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.show()
    image = spectrogram
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=50)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    print(img_arr.shape)
    return img_arr



