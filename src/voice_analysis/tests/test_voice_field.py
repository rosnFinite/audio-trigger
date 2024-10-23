import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.voice_analysis.voice_field import VoiceField


@pytest.fixture
def voice_field():
    with patch('src.audio.voice_field.DAQ_Device', autospec=True) as mock_daq:
        mock_daq_instance = mock_daq.return_value
        voice_field = VoiceField(
            rec_destination='/fake/path',
            semitone_bin_size=2,
            freq_bounds=(55, 1600),
            dba_bin_size=5,
            dba_bounds=(35, 115),
            min_score=0.7,
            retrigger_percentage_improvement=0.1
        )
        voice_field.daq = mock_daq_instance
        return voice_field


@pytest.mark.parametrize("os_path_exists_boolean", [[True, False], [True, True, True, True, False]])
def test_create_versioned_dir(os_path_exists_boolean, voice_field):
    # os_path_exists_boolean is a list of booleans that represent the return value of os.path.exists for each call
    # in this case equivalent to the number of  directories already existing
    with patch('os.path.exists', side_effect=os_path_exists_boolean), patch('os.makedirs') as mock_makedirs:
        versioned_dir = voice_field._VoiceField__create_versioned_dir('\\fake\\path')
        expected_path = f'\\fake\\path_{sum(os_path_exists_boolean)}'
        assert versioned_dir == expected_path
        mock_makedirs.assert_called_once_with(expected_path)


def test_is_bounds_valid_on_valid_input(voice_field):
    assert voice_field._VoiceField__is_bounds_valid((1, 2)) is True
    assert voice_field._VoiceField__is_bounds_valid((14234, 23248940)) is True


def test_is_bounds_valid_on_invalid_input(voice_field):
    assert voice_field._VoiceField__is_bounds_valid((1, 1)) is False
    assert voice_field._VoiceField__is_bounds_valid((1,)) is False


def test_calc_freq_lower_bounds(voice_field):
    lower_bounds = voice_field._VoiceField__calc_freq_lower_bounds(2, (55, 1600))
    expected_bounds = [55, 61.735, 69.295, 77.781, 87.306, 97.998, 109.999, 123.47, 138.59, 155.562, 174.612, 195.995,
                       219.997, 246.938, 277.179, 311.123, 349.224, 391.991, 439.995, 493.878, 554.359, 622.247,
                       698.449, 783.982, 879.99, 987.755, 1108.718, 1244.494, 1396.897, 1567.964]
    assert lower_bounds == expected_bounds


def test_calc_freq_lower_bounds_invalid_input(voice_field):
    with pytest.raises(ValueError, match="Provided frequency bounds are not valid."):
        voice_field._VoiceField__calc_freq_lower_bounds(2, (1, 1))


def test_calc_dba_lower_bounds(voice_field):
    lower_bounds = voice_field._VoiceField__calc_dba_lower_bounds(5, (35, 115))
    expected_bounds = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
    assert lower_bounds == expected_bounds


def test_calc_dba_lower_bounds_invalid_input(voice_field):
    with pytest.raises(ValueError, match="Provided db\(A\) bounds are not valid."):
        voice_field._VoiceField__calc_dba_lower_bounds(5, (1, 1))


@patch('src.audio.voice_field.json.dumps')
@patch('src.audio.voice_field.np.save')
@patch('src.audio.voice_field.wav.write')
def test_save_data(mock_wav_write, mock_np_save, mock_json_dumps, voice_field):
    trigger_data = {"audio": np.array([1, 2, 3]), "egg": np.array([4, 5, 6]), "sampling_rate": 16000}
    praat_stats = {"stat1": 1, "stat2": 2}
    freq_bin, freq, dba_bin, id = 0, 100.0, 1, 1
    save_dir = '\\fake\\\save\\dir'

    with patch('builtins.open', unittest.mock.mock_open()), patch('os.makedirs'):
        voice_field.save_data(save_dir, trigger_data, praat_stats, freq_bin, freq, dba_bin, id)

    assert mock_np_save.call_count == 2
    mock_wav_write.assert_called_with(f"{save_dir}/input_audio.wav", 16000, trigger_data["audio"])
    mock_json_dumps.assert_called()


@patch('src.audio.voice_field.json.dumps')
@patch('src.audio.voice_field.np.save')
@patch('src.audio.voice_field.wav.write')
def test_save_data_missing_egg_data(mock_wav_write, mock_np_save, mock_json_dumps, voice_field):
    trigger_data = {"audio": np.array([1, 2, 3]), "egg": None, "sampling_rate": 16000}
    praat_stats = {"stat1": 1, "stat2": 2}
    freq_bin, freq, dba_bin, id = 0, 100.0, 1, 1
    save_dir = '\\fake\\\save\\dir'

    with patch('builtins.open', unittest.mock.mock_open()), patch('os.makedirs'):
        voice_field.save_data(save_dir, trigger_data, praat_stats, freq_bin, freq, dba_bin, id)

    assert mock_np_save.call_count == 1
    mock_wav_write.assert_called_with(f"{save_dir}/input_audio.wav", 16000, trigger_data["audio"])
    mock_json_dumps.assert_called()


@patch('src.audio.voice_field.VoiceField._VoiceField__add_trigger')
@patch('src.audio.voice_field.VoiceField.emit_voice')
def test_check_trigger(mock_emit_voice, mock_add_trigger, voice_field):
    mock_sound = MagicMock()
    trigger_data = {"audio": np.array([1, 2, 3]), "egg": np.array([4, 5, 6]), "sampling_rate": 16000}
    freq, dba, score = 100.0, 50.0, 0.8

    result = voice_field.check_trigger(mock_sound, freq, dba, score, trigger_data)
    assert result is True
    mock_add_trigger.assert_called_once()
    # check_trigger will always send an emit to voice event
    # emit to trigger event is handled in __add_trigger which is only mocked here
    mock_emit_voice.assert_called_once()


@patch('src.audio.voice_field.VoiceField._VoiceField__add_trigger')
@patch('src.audio.voice_field.VoiceField.emit_voice')
def test_check_trigger_existing_score(mock_emit_voice, mock_add_trigger, voice_field):
    mock_sound = MagicMock()
    freq, dba, score = 100.0, 50.0, 0.8
    freq_bin, dba_bin = 5, 2
    voice_field.grid[dba_bin][freq_bin] = 0.9

    result = voice_field.check_trigger(mock_sound, freq, dba, score, None)
    assert result is False
    mock_add_trigger.assert_not_called()
    mock_emit_voice.assert_called_once()


@patch('src.audio.voice_field.VoiceField._VoiceField__add_trigger')
@patch('src.audio.voice_field.VoiceField.emit_voice')
def test_check_trigger_existing_score_improvement(mock_emit_voice, mock_add_trigger, voice_field):
    mock_sound = MagicMock()
    freq, dba, score = 100.0, 50.0, 0.8
    freq_bin, dba_bin = 5, 2
    voice_field.grid[dba_bin][freq_bin] = 0.5

    result = voice_field.check_trigger(mock_sound, freq, dba, score, None)
    assert result is True
    mock_add_trigger.assert_called_once()
    mock_emit_voice.assert_called_once()


@patch('src.audio.voice_field.VoiceField._VoiceField__add_trigger')
@patch('src.audio.voice_field.VoiceField.emit_voice')
@pytest.mark.parametrize("freq", [30.0, 1900.0])
def test_check_trigger_freq_out_of_bounds(mock_emit_voice, mock_add_trigger, voice_field, freq):
    mock_sound = MagicMock()
    trigger_data = {"audio": np.array([1, 2, 3]), "egg": np.array([4, 5, 6]), "sampling_rate": 16000}
    dba, score = 50.0, 0.8

    result = voice_field.check_trigger(mock_sound, freq, dba, score, trigger_data)
    assert result is False
    mock_add_trigger.assert_not_called()
    mock_emit_voice.assert_not_called()


@patch('src.audio.voice_field.VoiceField._VoiceField__add_trigger')
@patch('src.audio.voice_field.VoiceField.emit_voice')
@pytest.mark.parametrize("dba", [20.0, 130.0])
def test_check_trigger_dba_out_of_bounds(mock_emit_voice, mock_add_trigger, voice_field, dba):
    mock_sound = MagicMock()
    trigger_data = {"audio": np.array([1, 2, 3]), "egg": np.array([4, 5, 6]), "sampling_rate": 16000}
    freq, score = 1900.0, 0.8

    result = voice_field.check_trigger(mock_sound, freq, dba, score, trigger_data)
    assert result is False
    mock_add_trigger.assert_not_called()
    mock_emit_voice.assert_not_called()


@patch('src.audio.voice_field.VoiceField._VoiceField__add_trigger')
@patch('src.audio.voice_field.VoiceField.emit_voice')
def test_check_trigger_score_too_low(mock_emit_voice, mock_add_trigger, voice_field):
    mock_sound = MagicMock()
    trigger_data = {"audio": np.array([1, 2, 3]), "egg": np.array([4, 5, 6]), "sampling_rate": 16000}
    freq, dba, score = 100.0, 50.0, 0.2

    result = voice_field.check_trigger(mock_sound, freq, dba, score, trigger_data)
    assert result is False
    mock_add_trigger.assert_not_called()
    mock_emit_voice.assert_called_once()


@patch('src.audio.voice_field.VoiceField.emit_trigger')
@patch('src.audio.voice_field.measure_praat_stats')
@patch('src.audio.voice_field.VoiceField._VoiceField__create_versioned_dir')
@patch('src.audio.voice_field.VoiceField._VoiceField__submit_threadpool_task')
def test_add_trigger(mock_submit_threadpool_task, mock_create_versioned_dir, mock_measure_praat_stats, mock_emit_trigger, voice_field):
    mock_sound = MagicMock()
    freq_bin, dba_bin, score = 0, 0, 0.8
    trigger_data = {"audio": np.array([1, 2, 3]), "egg": np.array([4, 5, 6]), "sampling_rate": 16000}

    mock_create_versioned_dir.return_value = '\\fake\\dir'
    mock_measure_praat_stats.return_value = {"stat1": 1, "stat2": 2}

    with patch('builtins.open', unittest.mock.mock_open()), patch('os.makedirs'):
        voice_field._VoiceField__add_trigger(mock_sound, 60.0, freq_bin, dba_bin, score, trigger_data)

    voice_field.daq.start_acquisition.assert_called_once_with('\\fake\\dir')

    mock_submit_threadpool_task.assert_called_once()
    mock_emit_trigger.assert_called_once()


def test_emit_voice(voice_field):
    voice_field.socket = MagicMock()
    voice_field.socket.return_value.emit.return_value = None
    voice_field.emit_voice(0, 1, 100.0, 50.0, 0.8)
    voice_field.socket.emit.assert_called_once_with("voice", {"freq_bin": 0, "dba_bin": 1, "freq": 100.0, "dba": 50.0, "score": 0.8})


def test_emit_voice_no_socket(voice_field):
    voice_field.socket = None
    result = voice_field.emit_voice(0, 1, 100.0, 50.0, 0.8)
    assert result is None


def test_emit_trigger(voice_field):
    voice_field.socket = MagicMock()
    voice_field.socket.return_value.emit.return_value = None
    voice_field.emit_trigger(0, 1, 100.0, {"stat1": 1, "stat2": 2})
    voice_field.socket.emit.assert_called_once_with("trigger", {"freq_bin": 0, "dba_bin": 1, "score": 100.0, "stats": {"stat1": 1, "stat2": 2}})


def test_emit_trigger_no_socket(voice_field):
    voice_field.socket = None
    result = voice_field.emit_trigger(0, 1, 100.0, {"stat1": 1, "stat2": 2})
    assert result is None


@patch("src.audio.voice_field.VoiceField._VoiceField__create_versioned_dir")
def test_reset_grid(mock_create_versioned_dir, voice_field):
    voice_field.grid = [[1, 1, 1], [1, 1, 1]]
    voice_field.freq_bins_lb = [1, 2, 3]
    voice_field.dba_bins_lb = [1, 2, 3]
    voice_field.rec_destination = '\\fake\\path'

    mock_create_versioned_dir.return_value = '\\fake\\path_1'

    new_rec_destination = voice_field.reset_grid()

    assert any(voice_field.grid)
    assert new_rec_destination == '\\fake\\path_1'


if __name__ == "__main__":
    pytest.main()
