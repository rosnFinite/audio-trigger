import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import parselmouth
from src.voice_analysis.postprocessing.scoring import calc_quality_score, calc_pitch_score


def create_mock_sound(data, rate, freq_floor=None, freq_ceiling=None):
    mock_sound = MagicMock(spec=parselmouth.Sound)
    mock_sound.to_intensity.return_value.values = np.array([70.0, 72.0, 69.0])
    mock_pitch = MagicMock()
    mock_pitch.selected_array = np.array([[100.0], [105.0], [110.0]])
    mock_sound.to_pitch.return_value = mock_pitch
    return mock_sound


@patch('parselmouth.Sound')
def test_calc_pitch_score_with_data_and_rate(mock_sound_class):
    data = np.array([1, 2, 3, 4, 5])
    rate = 16000

    mock_intensity = MagicMock()
    mock_sound_class.return_value.to_intensity.return_value = mock_intensity

    mock_pitch = MagicMock()
    mock_sound_class.return_value.to_pitch.return_value = mock_pitch

    mock_intensity.values = np.array([70.0, 70.0, 70.0])
    mock_pitch.selected_array = np.array([[400.0], [400.0], [400.0]])

    score, mean_pitch = calc_pitch_score(data=data, rate=rate)
    assert score == 1
    assert mean_pitch == 400.0
    mock_sound_class.assert_called_with(data, sampling_frequency=rate)


@patch('parselmouth.Sound')
def test_calc_pitch_score_with_sound(mock_sound_class):
    mock_intensity = MagicMock()
    mock_sound_class.return_value.to_intensity.return_value = mock_intensity

    mock_pitch = MagicMock()
    mock_sound_class.return_value.to_pitch.return_value = mock_pitch

    mock_intensity.values = np.array([70.0, 70.0, 70.0])
    mock_pitch.selected_array = np.array([[400.0], [400.0], [400.0]])

    score, mean_pitch = calc_pitch_score(sound=mock_sound_class.return_value, freq_floor=50, freq_ceiling=1600)
    assert score == 1
    assert mean_pitch == 400.0


@patch('parselmouth.Sound')
def test_calc_pitch_score_with_sound_missing_floor_and_ceiling(mock_sound_class):
    with pytest.raises(ValueError, match="No frequency bounds provided."):
        calc_pitch_score(sound=mock_sound_class.return_value)