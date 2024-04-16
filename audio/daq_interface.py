import logging
import os
from typing import List
import nidaqmx
import nidaqmx.constants
import nidaqmx.error_codes
import nidaqmx.errors
import nidaqmx.system
import numpy as np


class DAQ_Device:
    def __init__(self,
                 sample_rate: int,
                 num_samples: int, 
                 analog_input_channels: List[str], 
                 digital_trig_channel: str, 
                 device_id: str = None):
        self.device = self.__select_daq(device_id)
        self.analog_input_channels = analog_input_channels
        self.digital_trig_channel = digital_trig_channel
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        pass

    @staticmethod
    def __select_daq(device_id: str = None) -> nidaqmx.system.Device:
        """Selects a connected NI-DAQmx device either by providing its id or by
        auto-detecting a device if no device_id is given.
        Auto-detection will select the first device in the list of connected devices per default.

        Will return None, without a thrown exception if:
        - No DAQ device is connected to the system (missing USB link)
        - No NI-DAQmx installation was found on the system (missing driver)
        - NI-DAQms is not supported on this system (e.g. macOS)
        """
        try:
            # check if a daq device is connected
            if len(nidaqmx.system.System.local().devices) == 0:
                logging.critical("No DAQ device connected. Continuing without...")
                return None
            device_list = list(nidaqmx.system.System.local().devices)
            logging.info(f"{len(device_list)} DAQ devices connected")
            # auto select first device in list if no specific ID was provided
            if device_id is None:
                device = device_list[0]
                logging.info(f"Device: {device} was auto selected.")
                return device
            else:
                device = device_list[device_list.index(device_id)]
                logging.info(f"Device: {device} was manually selected.")
                return device
        except nidaqmx.errors.DaqNotFoundError:
            logging.critical("No NI-DAQmx installation found on this system. Continuing without...")
            return None
        except nidaqmx.errors.DaqNotSupportedError:
            logging.critical("NI-DAQmx not supported on this device. Continuing without...")
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
            return
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

            task_in.start()

            task_trig.write([True])
            task_trig.write([False])

            data = np.array(task_in.read(number_of_samples_per_channel=self.num_samples))
        np.savetxt(os.path.join(save_dir, "measurements.csv"), data, delimiter=",")