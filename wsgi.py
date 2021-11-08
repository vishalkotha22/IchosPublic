
from website.run import app
from website.Preprocessing import wav_to_spectrogram, get_sli_features, get_feature_helper, respiratory_preprocess, \
    convert_audio_to_spectogram, plotstft, process_file
if __name__ == '__main__':
    app.run()
