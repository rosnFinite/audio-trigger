import collections
import os
import unittest

import pyaudio
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.voice_analysis.recorder import AudioRecorder, AudioTriggerRecorder


@pytest.fixture(autouse=True)
def audio_recorder():
    return AudioRecorder(buffer_size=1.0, rate=16000, channels=2, chunk_size=1024)


@pytest.fixture
def mock_config():
    return {
        "buffer_size": 1.0,
        "rate": 16000,
        "channels": 2,
        "chunk_size": 1024,
        "min_score": 0.7,
        "client_recordings_path": "recordings",
        "trigger_timeout": 1.0,
        "min_trigger_improvement": 0.1,
        "semitone_bin_size": 2,
        "frequency_bounds": {"lower": 150.0, "upper": 1700.0},
        "decibel_bin_size": 5,
        "decibel_bounds": {"lower": 35, "upper": 115}
    }


@pytest.fixture
def trigger(mock_config):
    with patch('src.audio.recorder.CONFIG', mock_config):
        return AudioTriggerRecorder(from_config=True)


def test_audio_recorder_initialization(audio_recorder):
    assert audio_recorder.chunk_size == 1024
    assert audio_recorder.channels == 2
    assert audio_recorder.rate == 16000
    assert audio_recorder.buffer_size == 1.0
    assert isinstance(audio_recorder.frames, collections.deque)
    assert audio_recorder.frames.maxlen == int((1.0 * 16000) / 1024)
    assert isinstance(audio_recorder.p, pyaudio.PyAudio)
    assert audio_recorder.stream is None


@patch('src.audio.recorder.pyaudio.PyAudio.get_host_api_info_by_index')
@patch('src.audio.recorder.pyaudio.PyAudio.get_device_info_by_host_api_device_index')
@pytest.mark.parametrize("device_count", [0, 1, 2, 3])
def test_load_recording_devices(mock_device_info, mock_host_info_by_id, device_count):
    # first dimension is host index, second dimension is device index
    mock_devices_info = [[
        {'maxInputChannels': 2, 'name': 'Device 1'},
        {'maxInputChannels': 1, 'name': 'Device 2'},
        {'maxInputChannels': 0, 'name': 'Device 3'},
    ]]

    def device_info_side_effect(host_idx, device_idx):
        return mock_devices_info[host_idx][device_idx]

    # Test case for no connected devices
    mock_host_info_by_id.return_value = {'deviceCount': device_count}
    mock_device_info.side_effect = device_info_side_effect

    audio_recorder = AudioRecorder()

    devices = audio_recorder.recording_devices
    test_device_list = [mock_devices_info[0][x]["name"] for x in range(device_count) if
                        mock_devices_info[0][x]["maxInputChannels"] > 0]
    assert devices == test_device_list
    assert len(devices) == len(test_device_list)


@pytest.mark.parametrize("channels", [1, 2])
def test_get_audio_data(channels, audio_recorder):
    mock_frames = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    audio_recorder.channels = channels
    audio_recorder.frames = collections.deque(mock_frames)

    audio_data = audio_recorder.get_audio_data()

    if channels == 1:
        test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    else:
        test_data = np.array([1, 3, 5, 7, 9])

    np.testing.assert_array_equal(test_data, audio_data)


@pytest.mark.parametrize("channels", [1, 2])
def test_get_egg_data(channels, audio_recorder):
    mock_frames = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    audio_recorder.channels = channels
    audio_recorder.frames = collections.deque(mock_frames)

    egg_data = audio_recorder.get_egg_data()

    if audio_recorder.channels == 1:
        assert egg_data is None
    else:
        np.testing.assert_array_equal(egg_data, np.array([2, 4, 6, 8, 10]))


@patch("src.audio.recorder.Thread")
@patch("src.audio.recorder.AudioRecorder._AudioRecorder__recording_callback", return_value=None)
@patch("src.audio.recorder.AudioRecorder._AudioRecorder__stream_task")
def test_start_stream_with_input_device_index(mock_stream_task, mock_recording_callback,mock_thread, audio_recorder):
    mock_thread_instance = MagicMock()
    mock_thread_instance.start.return_value = None
    mock_thread_instance.name = "TestThread"

    mock_thread.return_value = mock_thread_instance

    audio_recorder.start_stream(input_device_index=0)

    mock_thread.assert_called_with(target=mock_stream_task,
                                   args=(audio_recorder.recording_device, mock_recording_callback))
    mock_thread_instance.start.assert_called_once()
    assert audio_recorder.stream_thread_is_running is True
    audio_recorder.stop_stream()
    assert audio_recorder.stream_thread_is_running is False


@patch("src.audio.recorder.Thread")
@patch("src.audio.recorder.AudioRecorder._AudioRecorder__recording_callback", return_value=None)
@patch("src.audio.recorder.AudioRecorder._AudioRecorder__stream_task")
def test_start_stream_without_input_device_index(mock_stream_task, mock_recording_callback, mock_thread, audio_recorder):
    mock_thread_instance = MagicMock()
    mock_thread_instance.start.return_value = None
    mock_thread_instance.name = "TestThread"

    mock_thread.return_value = mock_thread_instance
    audio_recorder.recording_device = 0

    audio_recorder.start_stream()

    mock_thread.assert_called_with(target=mock_stream_task,
                                   args=(audio_recorder.recording_device, mock_recording_callback))
    mock_thread_instance.start.assert_called_once()
    assert audio_recorder.stream_thread_is_running is True
    audio_recorder.stop_stream()
    assert audio_recorder.stream_thread_is_running is False


@patch("src.audio.recorder.Thread")
@patch("src.audio.recorder.AudioRecorder._AudioRecorder__recording_callback", return_value=None)
@patch("src.audio.recorder.AudioRecorder._AudioRecorder__stream_task")
def test_start_stream_raising_value_error(mock_stream_task, mock_recording_callback, mock_thread, audio_recorder):
    # Test raising ValueError if input_device_index and recording_device are None
    mock_thread_instance = MagicMock()
    mock_thread_instance.start.return_value = None
    mock_thread_instance.name = "TestThread"

    mock_thread.return_value = mock_thread_instance
    audio_recorder.recording_device = None

    with pytest.raises(ValueError, match="No input device index provided."):
        audio_recorder.start_stream()

    assert mock_thread_instance.start.called is False
    assert audio_recorder.stream_thread_is_running is False


@patch("src.audio.recorder.Thread")
@patch("src.audio.recorder.AudioRecorder._AudioRecorder__recording_callback", return_value=None)
@patch("src.audio.recorder.AudioRecorder._AudioRecorder__stream_task")
def test_start_stream_while_stream_is_running(mock_stream_task, mock_recording_callback, mock_thread, audio_recorder):
    mock_thread_instance = MagicMock()
    mock_thread_instance.start.return_value = None
    mock_thread_instance.name = "TestThread"

    mock_thread.return_value = mock_thread_instance
    audio_recorder.recording_device = 0
    audio_recorder.stream_thread_is_running = True
    audio_recorder.stream_thread = mock_thread_instance

    audio_recorder.start_stream()

    assert audio_recorder.stream_thread_is_running is True
    assert mock_thread_instance.start.called is False


@patch('src.audio.recorder.wav.write')
@patch("src.audio.recorder.AudioRecorder.stop_stream")
def test_stop_stream_and_save_wav(mock_stop_stream, mock_wav_write, audio_recorder):
    mock_frames = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    with patch.object(audio_recorder, 'get_audio_data', return_value=mock_frames):
        audio_recorder.stop_stream_and_save_wav('/fake/path')
    mock_wav_write.assert_called_with('/fake/path/out.wav', 16000, mock_frames)


def test_stop_stream(audio_recorder):
    audio_recorder.stream_thread_is_running = True

    audio_recorder.stop_event = MagicMock()
    audio_recorder.stop_event.return_value.set.return_value = None

    audio_recorder.stream_thread = MagicMock()
    audio_recorder.stream_thread.return_value.join.return_value = None

    audio_recorder.stop_stream()

    assert audio_recorder.stream_thread_is_running is False


def test_stop_stream_no_stream_running(audio_recorder):
    audio_recorder.stream_thread_is_running = False

    audio_recorder.stop_event = MagicMock()
    audio_recorder.stop_event.return_value.set.return_value = None

    audio_recorder.stream_thread = MagicMock()
    audio_recorder.stream_thread.return_value.join.return_value = None

    audio_recorder.stop_stream()

    assert audio_recorder.stream_thread_is_running is False

# Test cases for AudioTriggerRecorder class

def test_audio_trigger_initialization(trigger):
    assert trigger.chunk_size == 1024
    assert trigger.channels == 2
    assert trigger.rate == 16000
    assert trigger.buffer_size == 1.0
    assert isinstance(trigger.frames, collections.deque)
    assert trigger.frames.maxlen == int((1.0 * 16000) / 1024)
    assert isinstance(trigger.p, pyaudio.PyAudio)
    assert trigger.stream is None
    assert trigger.socket is None
    assert trigger.trigger_timeout == 1.0
    assert trigger.min_score == 0.7
    assert trigger.rec_destination == "recordings"
    assert trigger.init_settings == {
        "sampling_rate": 16000,
        "save_location": "recordings",
        "buffer_size": 1.0,
        "chunk_size": 1024,
        "channels": 2,
        "min_score": 0.7,
        "retrigger_percentage_improvement": 0.1,
        "freq_bounds": (150.0, 1700.0),
        "semitone_bin_size": 2,
        "dba_bounds": (35, 115),
        "dba_bin_size": 5
    }
    assert trigger.calib_factors is None


def test_settings(trigger):
    assert trigger.settings == {
        "sampling_rate": 16000,
        "save_location": "recordings",
        "buffer_size": 1.0,
        "chunk_size": 1024,
        "channels": 2,
        "min_score": 0.7,
        "retrigger_percentage_improvement": 0.1,
        "freq_bounds": (150.0, 1700.0),
        "semitone_bin_size": 2,
        "dba_bounds": (35, 115),
        "dba_bin_size": 5,
        "device": None
    }


@patch("src.audio.recorder.os.path.exists", return_value=True)
@patch("src.audio.recorder.os.makedirs")
def test_check_rec_destination(mock_makedirs, mock_path_exists, trigger):
    trigger = AudioTriggerRecorder(rec_destination="C:/tests/recordings")
    assert trigger.rec_destination == "C:/tests/recordings"
    assert mock_path_exists.called
    assert mock_makedirs.called is False


@patch("src.audio.recorder.os.path.exists", return_value=False)
@patch("src.audio.recorder.os.makedirs")
def test_check_rec_destination_path_not_exists(mock_os_makedirs, mock_path_exists):
    AudioTriggerRecorder(rec_destination="C:/tests/recordings")
    assert mock_path_exists.called
    assert mock_os_makedirs.called_with("C:/tests/recordings")


@patch("src.audio.recorder.json.load", return_value={"37": [1, 17.8, 20], "45": [1, 18.5, 28], "50": [1, 20.1, 30]})
def test_load_calib_factors(mock_json_load):
    with patch('builtins.open', unittest.mock.mock_open()):
        trigger = AudioTriggerRecorder(dba_calib_file="/fake/path.json")
    assert trigger.calib_factors == {17.8: 20, 18.5: 28, 20.1: 30}
    assert mock_json_load.called_once()


def test_load_calib_factors_wrong_file_type():
    with pytest.raises(ValueError, match="Invalid file format. JSON file required."), patch('builtins.open', unittest.mock.mock_open()):
        AudioTriggerRecorder(dba_calib_file="/fake/path.txt")



if __name__ == "__main__":
    pytest.main()
