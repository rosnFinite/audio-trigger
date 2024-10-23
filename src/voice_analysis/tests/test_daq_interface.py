import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.voice_analysis.daq_interface import DAQ_Device


@pytest.fixture
def mock_config():
    return {
        "analog_input_channels": ["ai0", "ai1"],
        "digital_trigger_channel": "pfi5",
        "sample_rate": 1000,
        "number_of_samples": 1000
    }


@pytest.fixture
def daq_device(mock_config):
    with patch('src.audio.daq_interface.CONFIG', mock_config):
        return DAQ_Device(
            from_config=True
        )


def test_init_from_config(daq_device, mock_config):
    assert daq_device.analog_input_channels == mock_config["analog_input_channels"]
    assert daq_device.digital_trig_channel == mock_config["digital_trigger_channel"]
    assert daq_device.sample_rate == mock_config["sample_rate"]
    assert daq_device.num_samples == mock_config["number_of_samples"]


def test_init_from_valid_args():
    sample_rate = 1000
    num_samples = 1000
    analog_input_channels = ["ai0", "ai1"]
    digital_trig_channel = "pfi5"
    daq_device = DAQ_Device(
        sample_rate=sample_rate,
        num_samples=num_samples,
        analog_input_channels=analog_input_channels,
        digital_trig_channel=digital_trig_channel
    )
    assert daq_device.analog_input_channels == analog_input_channels
    assert daq_device.digital_trig_channel == digital_trig_channel
    assert daq_device.sample_rate == sample_rate
    assert daq_device.num_samples == num_samples


def test_init_with_missing_args():
    with pytest.raises(ValueError, match='analog_input_channels and digital_trig_channel need to be provided.'):
        DAQ_Device()
    with pytest.raises(ValueError, match='analog_input_channels and digital_trig_channel need to be provided.'):
        DAQ_Device(sample_rate=10000, num_samples=1000)


def test_init_with_invalid_args():
    with pytest.raises(TypeError, match='analog_input_channels must be a list of strings.'):
        DAQ_Device(analog_input_channels="ai0", digital_trig_channel="pfi5")
    with pytest.raises(TypeError, match='digital_trig_channel must be a string.'):
        DAQ_Device(analog_input_channels=["ai0", "ai1"], digital_trig_channel=["pfi5"])


@patch('src.audio.daq_interface.nidaqmx.system.System.local')
def test_select_daq_without_device_connected(mock_local_system, mock_config):
    mock_local_system.return_value.devices = []
    with patch('src.audio.daq_interface.CONFIG', mock_config):
        daq_device = DAQ_Device(
            from_config=True
        )
    assert daq_device.device is None


@patch('src.audio.daq_interface.nidaqmx.system.System.local')
def test_select_daq_with_device_connected(mock_local_system, mock_config):
    mock_local_system.return_value.devices = ["Dev1"]
    with patch('src.audio.daq_interface.CONFIG', mock_config):
        daq_device = DAQ_Device(
            from_config=True
        )
    assert daq_device.device == "Dev1"


@patch('src.audio.daq_interface.nidaqmx.system.System.local')
def test_select_daq_with_multiple_devices_connected(mock_local_system, mock_config):
    mock_local_system.return_value.devices = ["Dev1", "Dev2", "Dev3"]
    with patch('src.audio.daq_interface.CONFIG', mock_config):
        daq_device = DAQ_Device(
            from_config=True
        )
    assert daq_device.device == "Dev1"


@patch('src.audio.daq_interface.nidaqmx.system.System.local')
def test_select_daq_with_multiple_devices_connected_and_manual_selection(mock_local_system, mock_config):
    mock_local_system.return_value.devices = ["Dev1", "Dev2", "Dev3"]
    with patch('src.audio.daq_interface.CONFIG', mock_config):
        daq_device = DAQ_Device(
            from_config=True,
            device_id="Dev2"
        )
    assert daq_device.device == "Dev2"


@patch('src.audio.daq_interface.nidaqmx.system.System.local')
def test_select_daq_raise_value_error_on_invalid_selection(mock_local_system, mock_config):
    mock_local_system.return_value.devices = ["Dev1", "Dev2", "Dev3"]
    with pytest.raises(ValueError, match="'Dev7' is not in list"):
        with patch('src.audio.daq_interface.CONFIG', mock_config):
            DAQ_Device(
                from_config=True,
                device_id="Dev7"
            )


def test_start_acquisition_without_connected_device(daq_device):
    daq_device.device = None
    with pytest.raises(AttributeError, match='No DAQ device connected. Cannot start acquisition.'):
        daq_device.start_acquisition('/fake/path')


# TODO: Test case for acquisition method (hard because nidaqmx.system.System.local().devices is not a list of strings)
"""
@patch('src.audio.daq_interface.nidaqmx.Task')
@patch('src.audio.daq_interface.os.path.join')
@patch('src.audio.daq_interface.np.savetxt')
@patch('src.audio.daq_interface.nidaqmx.system.System.local')
def test_start_acquisition_with_device_connected(mock_local_system, mock_savetxt, mock_path_join, mock_nidaqmx_task, mock_config):
    mock_task_in = MagicMock()
    mock_task_trig = MagicMock()
    mock_nidaqmx_task.side_effect = [mock_task_in, mock_task_trig]

    mock_path_join.return_value = '/fake/path/measurements.csv'
    mock_data = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    mock_task_in.read.return_value = mock_data

    mock_local_system.return_value.devices = [{"name": "Dev1"}]
    with patch('src.audio.daq_interface.CONFIG', mock_config):
        daq_device = DAQ_Device(
            from_config=True
        )

    daq_device.start_acquisition('/fake/path')

    mock_task_in.ai_channels.add_ai_voltage_chan.assert_called()
    mock_task_in.timing.cfg_samp_clk_timing.assert_called_with(
        rate=daq_device.sample_rate,
        sample_mode=mock_nidaqmx_task.FINITE,
        samps_per_chan=daq_device.num_samples
    )
    mock_task_in.triggers.start_trigger.cfg_dig_edge_start_trig.assert_called()
    mock_task_trig.do_channels.add_do_chan.assert_called()

    mock_task_in.start.assert_called()
    mock_task_trig.write.assert_any_call([True])
    mock_task_trig.write.assert_any_call([False])

    mock_task_in.read.assert_called_with(number_of_samples_per_channel=daq_device.num_samples)
    mock_savetxt.assert_called_with('/fake/path/measurements.csv', mock_data, delimiter=',')
"""

if __name__ == "__main__":
    pytest.main()
