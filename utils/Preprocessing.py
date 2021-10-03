import librosa as lb
import numpy as np


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


