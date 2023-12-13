import time
import plotly.graph_objs as go
import librosa
import numpy as np
import scipy

from webapp.processing.weighting import A_weight
from utility.utility import bisection
from recorder import AudioRecorder

recorder = AudioRecorder(buffer_size=1, rate=44100)

info = recorder.p.get_host_api_info_by_index(0)
data = []
for i in range(0, info.get('deviceCount')):
    if (recorder.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print(recorder.p.get_device_info_by_host_api_device_index(0, i).get('name'))

#
# 70dB(A) = [4400000 - 5300000] mittel = 4800000 (RAZER)

# 50dB(A) = [56000 - 66000]
# 60dB(A) = [210000 - 220000]
# 70dB(A) = [1090000 - 1150000]
# 80dB(A) = [2090000 - 2150000] (Behringer)
# 90dB(A) = [5600000 - 5800000] (Behringer)

# 37dB(A) = 17.8 dB(A)FS = 20
# 45dB(A) = 18.5 dB(A)FS = 28
# 50dB(A) = 20.1 dB(A)FS = 30
# 55dB(A) = 22.7 dB(A)FS = 32.3
# 60dB(A) = 26.7 dB(A)FS = 33.9
# 65dB(A) = 31.9 dB(A)FS = 33.1
# 70dB(A) = 36.2 dB(A)FS = 34
# 75dB(A) = 41.3 dB(A)FS = 34
# 80dB(A) = 46.1 dB(A)FS = 34
# 85dB(A) = 51.7 dB(A)FS = 34
# 90dB(A) = 56.3 dB(A)FS = 34

behringer = {
    36: 45000,
    37: 54000,
    40: 70000,
    50: 105000,
    60: 215000,
    70: 1120000,
    80: 2120000,
    90: 5700000
}

behringer_cor = {
    17.8: 20,
    18.5: 27,
    20.1: 30,
    22.7: 32.3,
    26.7: 33.9,
    31.9: 33.1,
    36.2: 34,
    41.3: 34,
    46.1: 34,
    51.7: 34,
    56.3: 34
}

# z = np.polyfit(list(behringer.values()), list(behringer.keys()), 2)
# poly_interp = np.poly1d(z)

xp = np.linspace(45000, 5700000, 5700000 - 45000 + 1)
poly_interp = np.interp(xp, list(behringer.values()), list(behringer.keys()))
xp = np.linspace(17, 60)
cor_interp = np.interp(xp, list(behringer_cor.keys()), list(behringer_cor.values()))
# pp = poly_interp(xp)

fig = go.Figure(go.Scatter(x=xp, y=cor_interp))
fig.show()


# print(poly_interp(185004))

def check_sound_values():
    recorder.start_stream(input_device_index=2)

    while recorder.stream.is_active():
        time.sleep(1)
        data = recorder.get_audio_data()
        weighted_signal = A_weight(data, fs=41000)
        fourier = scipy.fft.fft(data)
        fourier_to_plot = fourier[0:len(fourier) // 2]
        w = np.linspace(0, recorder.rate, len(fourier))[0:len(fourier) // 2]
        abs_freq = np.abs(fourier_to_plot)
        amp = abs_freq.argmax()
        freq = w[amp]
        note = librosa.hz_to_note(freq)
        rms_value = np.sqrt(np.mean(np.abs(weighted_signal) ** 2))
        print("=" * 24)
        result = 20 * np.log10(rms_value)
        print(f"RESULT: {result:.2f}")
        idx = bisection(xp, result)
        print(f"CORRECTION: {cor_interp[idx]:.2f}")
        result += cor_interp[idx]
        # print(poly_interp[round(abs_freq[amp]) - 45000])
        print(result)
        print(f"Frequency: {note}   FFT-Value: {abs_freq[amp]}   MAX: {np.max(data)}")


check_sound_values()
