import math

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.voice_analysis.postprocessing.db import get_dba_level, interp_correction  # Adjust the import path to your module


def test_get_dba_level_no_correction():
    data = np.array([1, 2, 3, 4, 5])
    rate = 16000
    result = get_dba_level(data, rate)
    expected_rms = np.sqrt(np.mean(np.power(np.abs(data).astype(np.int32), 2)))
    expected_result = 20 * np.log10(expected_rms)
    assert np.isclose(result, expected_result, rtol=1e-5)


def test_get_dba_level_empty_data():
    data = np.array([])
    rate = 16000
    with pytest.raises(ValueError, match="No data provided."):
        get_dba_level(data, rate)


def test_interp_correction():
    corr_dict = {40: 1.5, 50: 2.0, 60: 2.5}
    xp, corr_interp = interp_correction(corr_dict)
    expected_xp = [40., 40.40816327, 40.81632653, 41.2244898, 41.63265306, 42.04081633, 42.44897959, 42.85714286,
                   43.26530612, 43.67346939, 44.08163265, 44.48979592, 44.89795918, 45.30612245, 45.71428571,
                   46.12244898, 46.53061224, 46.93877551, 47.34693878, 47.75510204, 48.16326531, 48.57142857,
                   48.97959184, 49.3877551, 49.79591837, 50.20408163, 50.6122449, 51.02040816, 51.42857143, 51.83673469,
                   52.24489796, 52.65306122, 53.06122449, 53.46938776, 53.87755102, 54.28571429, 54.69387755,
                   55.10204082, 55.51020408, 55.91836735, 56.32653061, 56.73469388, 57.14285714, 57.55102041,
                   57.95918367, 58.36734694, 58.7755102, 59.18367347, 59.59183673, 60]
    expected_corr_interp = [1.5, 1.52040816, 1.54081633, 1.56122449, 1.58163265, 1.60204082, 1.62244898, 1.64285714,
                            1.66326531, 1.68367347, 1.70408163, 1.7244898, 1.74489796, 1.76530612, 1.78571429,
                            1.80612245, 1.82653061, 1.84693878, 1.86734694, 1.8877551, 1.90816327, 1.92857143,
                            1.94897959, 1.96938776, 1.98979592, 2.01020408, 2.03061224, 2.05102041, 2.07142857,
                            2.09183673, 2.1122449, 2.13265306, 2.15306122, 2.17346939, 2.19387755, 2.21428571,
                            2.23469388, 2.25510204, 2.2755102, 2.29591837, 2.31632653, 2.33673469, 2.35714286,
                            2.37755102, 2.39795918, 2.41836735, 2.43877551, 2.45918367, 2.47959184, 2.5]
    assert np.allclose(expected_xp, xp)
    assert np.allclose(expected_corr_interp, corr_interp)


if __name__ == "__main__":
    pytest.main()
