import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import warnings
import soundfile as sf
import pandas as pd
import sys
import pyloudnorm as pyln
warnings.simplefilter("ignore")


### functions to compute musical featureas

def get_length(y, sr):
    return {'length' : len(y)/sr}

def get_loudness(filename):
    data, rate = sf.read(filename)
    meter = pyln.Meter(rate) #
    loudness = meter.integrated_loudness(data)
    return {'loudness' : loudness}

def get_temp(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return {'tempo' : tempo}

def pitch_tuning(y, sr):
    pitches, magnitudes = librosa.piptrack(y, sr)
    return {'pitch_tuning' : librosa.pitch_tuning(pitches)}

def get_spectral_properties(y: np.ndarray, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
    result_d = {
        'freq_mean': mean,
        'freq_sd': sd,
        'freq_median': median,
        'freq_mode': mode,
        'freq_Q25': Q25,
        'freq_Q75': Q75,
        'freq_IQR': IQR,
        'freq_skew': skew,
        'freq_kurt': kurt
    }
    return result_d

def get_onset_intensity(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)
    onset_int = np.mean(list(onset_env))
    return {'onset_int' : onset_int}



### analyze all the tracks from the root folder
root    = sys.argv[1]
quality = os.listdir(root)
tracks  = []

for q in quality:

    files = os.listdir(root + '/' + q)

    for fn in files[0:3]:

        print('Processing ' + fn + '...')

        track = {}
        track['filename'] = fn
        track['quality']  = q

        y, sr = librosa.load(root + '/' + q + '/' + fn, duration = 4.2)
        track.update(get_length(y, sr))
        track.update(get_loudness(root + '/' + q + '/' + fn))
        track.update(get_temp(y, sr))
        track.update(pitch_tuning(y, sr))
        track.update(get_spectral_properties(y, sr))
        track.update(get_onset_intensity(y, sr))
        tracks.append(track)


### saving the results in a spreadsheet
df = pd.DataFrame(tracks)
df.index = df.filename
df = df.drop(columns = ['filename'])
folderout = 'MusicalFeatures'
if not os.path.exists(folderout):
    os.makedirs(folderout)
df.to_csv(folderout + '/musical_features_demo.csv')
