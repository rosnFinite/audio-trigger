import logging
import os
from typing import List, Optional
import nidaqmx
import nidaqmx.constants
import nidaqmx.error_codes
import nidaqmx.errors
import nidaqmx.system
import numpy as np

from src.config_utils import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DAQ_Device:
    def __init__(self,
                 sample_rate: Optional[int] = 1000,
                 num_samples: Optional[int] = 1000,
                 analog_input_channels: Optional[List[str]] = None,
                 digital_trig_channel: Optional[str] = None,
                 device_id: Optional[str] = None,
                 from_config: bool = False):
        self.device = self.__select_daq(device_id)
        if from_config:
            self.analog_input_channels = CONFIG["voice_field"]["analog_input_channels"]
            self.digital_trig_channel = CONFIG["voice_field"]["digital_trigger_channel"]
            self.sample_rate = CONFIG["voice_field"]["sample_rate"]
            self.num_samples = CONFIG["voice_field"]["number_of_samples"]
        else:
            if analog_input_channels is None or digital_trig_channel is None:
                raise ValueError("analog_input_channels and digital_trig_channel need to be provided.")
            if type(analog_input_channels) is not list:
                raise TypeError("analog_input_channels must be a list of strings.")
            if type(digital_trig_channel) is not str:
                raise TypeError("digital_trig_channel must be a string.")
            self.analog_input_channels = analog_input_channels
            self.digital_trig_channel = digital_trig_channel
            self.sample_rate = sample_rate
            self.num_samples = num_samples

    @staticmethod
    def __select_daq(device_id: str = None) -> Optional[nidaqmx.system.Device]:
        """Selects a connected NI-DAQmx device either by providing its id or by
        auto-detecting a device if no device_id is given.
        Auto-detection will select the first device in the list of connected devices per default.

        Will return None, without a thrown exception IF:
        - No DAQ device is connected to the system (missing USB link)
        - No NI-DAQmx installation was found on the system (missing driver)
        - NI-DAQms is not supported on this system (e.g. macOS)
        """
        try:
            # check if a daq device is connected
            if len(nidaqmx.system.System.local().devices) == 0:
                logger.critical("No DAQ device connected. Continuing without...")
                return None
            device_list = list(nidaqmx.system.System.local().devices)
            logger.info(f"{len(device_list)} DAQ devices connected")
            # auto select first device in list if no specific ID was provided
            if device_id is None:
                device = device_list[0]
                logger.info(f"Device: {device} was auto selected.")
                return device
            else:
                device = device_list[device_list.index(device_id)]
                logger.info(f"Device: {device} was manually selected.")
                return device
        except nidaqmx.errors.DaqNotFoundError:
            logger.critical("No NI-DAQmx installation found on this system. Continuing without...")
            return None
        except nidaqmx.errors.DaqNotSupportedError:
            logger.critical("NI-DAQmx not supported on this device. Continuing without...")
            return None

    def start_acquisition(self, save_dir: str) -> None:
        """Starts the data acquisition process with the provided settings and saves the acquired data to a file.
        Provided 'save_dir' must be a valid path to a directory where the data will be saved. Filename will be
        'measurements.csv'.
        Full path -> 'save_dir'/measurements.csv

        Parameters
        ----------
        save_dir : str
            Parent directory path where the acquired data will be saved as 'measurements.csv'.
        """
        if self.device is None:
            raise AttributeError("No DAQ device connected. Cannot start acquisition.")
        with nidaqmx.Task() as task_in, nidaqmx.Task() as task_trig:
            # configure analog input channels
            for ai_channel in self.analog_input_channels:
                task_in.ai_channels.add_ai_voltage_chan(f"/{self.device.name}/{ai_channel}")

            # timing for analog task
            task_in.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=self.num_samples
            )

            # reference trigger for analog input task
            task_in.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=f"/{self.device.name}/{self.digital_trig_channel}",
                trigger_edge=nidaqmx.constants.Edge.RISING
            )

            # configure digital trigger output for acquisition and camera recording
            task_trig.do_channels.add_do_chan(
                lines=f"/{self.device.name}/{self.digital_trig_channel}",
                line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE
            )
            
            data = None
            try:
                task_in.start()
                
                task_trig.write([True])
                task_trig.write([False])

                timestamps = np.array([x/self.sample_rate for x in range(self.num_samples)])
                measurements = np.array(task_in.read(number_of_samples_per_channel=self.num_samples))
                
                # extract every dimension from measurements (each is an input channel) and collect every data to save in list
                d_list = [timestamps[:, np.newaxis]]
                for dim in measurements:
                    d_list.append(dim[:, np.newaxis])
                data = np.hstack(tuple(d_list))
            except nidaqmx.errors.DaqError:
                logger.critical("DAQ process could NOT be started. Check if another program is accessing DAQ resources.")
        if data is not None:
            header = ",".join(["time"] + self.analog_input_channels)
            np.savetxt(os.path.join(save_dir, "measurements.csv"), data, delimiter=",", header=header, comments="")
        else:
            with open(os.path.join(save_dir, "measurment_error_info.txt"), "w") as error_file:
                print("Critical error occured. DAQ measurement process could NOT be started. Check if another program "
                      "is reserving/using DAQ resources.", file=error_file)
