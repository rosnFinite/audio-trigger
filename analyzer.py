import functools

import librosa
import numpy as np
import scipy

from recorder import AudioRecorder

def fourier_transform(audio_data):
    len_data = len(audio_data)
    return scipy.fft.fft(audio_data)

def get_note(audio_data):
    return librosa.hz_to_note()