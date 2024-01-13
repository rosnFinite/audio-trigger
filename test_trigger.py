import dataclasses
import numpy as np
import pyaudio
import plotly.graph_objs as go

from recorder import AudioRecorder
from webapp.processing.fourier import get_dba_level, get_dominant_freq, calc_quality_score, fft


class Grid:
    def __init__(self, semitone_bin_size: int, dba_bin_size: int, min_q_score: float):
        self.freq_bins_lb = self.__calc_freq_lower_bounds(semitone_bin_size)
        self.dba_bins_lb = self.__calc_dba_lower_bounds(dba_bin_size)
        self.min_q_score = min_q_score
        self.grid = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]

    def __calc_freq_lower_bounds(self, semitone_bin_size: int):
        # arbitrary start point for semitone calculations
        lower_bounds = [55.0]
        while lower_bounds[-1] < 2093:
            lower_bounds.append(np.power(2, semitone_bin_size / 12) * lower_bounds[-1])
        return lower_bounds

    def __calc_dba_lower_bounds(self, dba_bin_size: int):
        lower_bounds = [45]
        while lower_bounds[-1] < 110:
            lower_bounds.append(lower_bounds[-1] + 5)
        return lower_bounds

    def add_trigger(self, freq, dba, q_score):
        # find corresponding freq and db bins
        freq_bin = np.searchsorted(self.freq_bins_lb, freq)
        dba_bin = np.searchsorted(self.dba_bins_lb, dba)
        if freq_bin == 0 or dba_bin == 0:
            # value is smaller than the lowest bound
            return
        if q_score > self.min_q_score:
            return
        old_q_score = self.grid[dba_bin - 1][freq_bin - 1]
        if old_q_score is None:
            self.grid[dba_bin - 1][freq_bin - 1] = q_score
        else:
            if old_q_score > q_score:
                self.grid[dba_bin - 1][freq_bin - 1] = q_score

    def show_grid(self):
        fig = go.Figure(data=go.Heatmap(
            z=self.grid,
            hoverongaps=False
        ))
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(self.freq_bins_lb))),
                ticktext=self.freq_bins_lb
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(self.dba_bins_lb))),
                ticktext=self.dba_bins_lb
            )
        )
        fig.show()


class Trigger(AudioRecorder):
    def __init__(self,
                 rec_destination: str,
                 min_q_score: float = 30,
                 semitone_bin_size: int = 2,
                 dba_bin_size: int = 5,
                 buffer_size: int = 1,
                 rate: int = 44100,
                 chunksize: int = 1024):
        super().__init__(buffer_size, rate, chunksize)
        self.grid = Grid(semitone_bin_size, dba_bin_size, min_q_score)
        self.__rec_destination = rec_destination

    def start_trigger(self, input_device_index: int):
        print("start")
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  input_device_index=input_device_index,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.__trigger_callback)

    def __trigger_callback(self, input_data, frame_count, time_info, flags):
        frame = np.frombuffer(input_data, dtype=np.int16)
        self.frames.append(frame)
        if len(self.frames) == self.frames.maxlen:
            data = self.get_audio_data()
            fourier, fourier_to_plot, abs_freq, w = fft(data, self.rate)
            # print(get_dba_level(data, self.rate))
            self.grid.add_trigger(get_dominant_freq(abs_freq=abs_freq, w=w),
                                  get_dba_level(data, self.rate),
                                  calc_quality_score(abs_freq=abs_freq))
        return input_data, pyaudio.paContinue

    def stop_trigger(self):
        super().stop_stream()


if __name__ == "__main__":
    trigger = Trigger("test")
    # trigger.start(1, 2)
