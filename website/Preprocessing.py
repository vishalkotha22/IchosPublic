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
import speech_recognition as sr
import librosa.display
import re
import soundfile as sf
from numpy.lib import stride_tricks

class_names = {0: 'URTI', 1: 'Healthy', 2: 'Asthma', 3: 'COPD', 4: 'LRTI', 5: 'Bronchiectasis',
               6: 'Pneumonia', 7: 'Bronchiolitis'}

class_names_full = {0: 'Upper Respiratory Tract Infection', 1: 'Healthy',
                    2: 'Asthma', 3: 'Chronic Obstructive Pulmonary Disease',
                    4: 'Lower Respiratory Tract Infections', 5: 'Bronchiectasis',
                    6: 'Pneumonia', 7: 'Bronchiolitis'}


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


def getPureSample(raw_data, start, end, sr=22050):
    max_ind = len(raw_data)
    start_ind = min(int(start * sr), max_ind)
    end_ind = min(int(end * sr), max_ind)
    return raw_data[start_ind: end_ind]


def process_file(vid_file_path):
    start = 1.
    end = 5.
    audioArr, sampleRate = lb.load(vid_file_path)
    pureSample = getPureSample(audioArr, start, end, sampleRate)
    reqLen = 6 * sampleRate
    padded_data = lb.util.pad_center(pureSample, reqLen)
    sf.write(file='processed.wav', data=padded_data, samplerate=sampleRate)


def get_sli_features(wav_file):
    r = sr.Recognizer()

    def get_large_audio_transcription(path):
        sound = AudioSegment.from_file(path)
        chunks = split_on_silence(sound,
                                  min_silence_len=500,
                                  silence_thresh=sound.dBFS - 14,
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
    for i in range(len(split) - 1):
        temp = 0
        while split[i] == split[i + 1]:
            temp = temp + 1
            i = i + 1
        repetition = temp + 1
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


def wav_to_spectrogram(file): #old version1
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(1)
    sound.export("path.wav", format="wav")
    sample_rate, samples = wavfile.read("path.wav")
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.axis('off')
    # plt.show()
    image = spectrogram
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=50)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    print(img_arr.shape)
    return img_arr


def convert_audio_to_spectogram(filename): #old version2
    x, sr = lb.load(filename, sr=44100)
    X = lb.stft(x)
    Xdb = lb.amplitude_to_db(abs(X))
    fig = plt.figure(figsize=(2.95, 2.95))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')




def stft(sig, frameSize, overlapFac=0.5, window=np.hanning): #curr working version
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


""" plot spectrogram"""


def plotstft(audiopath, binsize=2 ** 10, plotpath=None, colormap="jet"):
    samplerate, samples = wavfile.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=10, sr=44100)

    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    fig = plt.figure(figsize=(2.95, 2.95))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins - 1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins - 1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins - 1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    plt.savefig('spectrogram.png')
    image = Image.open('spectrogram.png')
    image = np.array(image)
    plt.clf()
    return image

