import librosa as lb
import numpy as np


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


