import logging
import os
import concurrent.futures
import threading
import time
import nidaqmx
import nidaqmx.constants
import nidaqmx.error_codes
import nidaqmx.errors
import nidaqmx.system
import numpy as np

from typing import List, Optional
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import DigitalSingleChannelWriter

from src.config_utils import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DAQ_Device:
    def __init__(self,
                 sample_rate: Optional[int] = 20000,
                 sampling_time: Optional[float] = None,
                 analog_input_channels: Optional[List[str]] = None,
                 digital_trig_channel: Optional[str] = None,
                 device_id: Optional[str] = None,
                 from_config: bool = False):
        self.device = self.__select_daq(device_id)
        if from_config:
            self.analog_input_channels = CONFIG["voice_field"]["analog_input_channels"]
            self.digital_trig_channel = CONFIG["voice_field"]["digital_trigger_channel"]
            self.sample_rate = CONFIG["voice_field"]["sample_rate"]
            self.sampling_time = CONFIG["recorder"]["buffer_size"]
            self.num_samples = int(self.sample_rate * self.sampling_time)
        else:
            if analog_input_channels is None or digital_trig_channel is None:
                raise ValueError("analog_input_channels and digital_trig_channel need to be provided.")
            if type(analog_input_channels) is not list:
                raise TypeError("analog_input_channels must be a list of strings.")
            if type(digital_trig_channel) is not str:
                raise TypeError("digital_trig_channel must be a string.")
            if sampling_time is None:
                raise ValueError(f"sampling_time needs to be provided to calculate number of samples to acquire per channel")
            self.analog_input_channels = analog_input_channels
            self.digital_trig_channel = digital_trig_channel
            self.sample_rate = sample_rate
            self.sampling_time = sampling_time
            self.num_samples = int(sample_rate * sampling_time)
        if self.device is not None:
            self.task_in = nidaqmx.Task()
            self.task_out = nidaqmx.Task()
            self.__setup_tasks()
            logger.debug(f"Tasks created - input_buf_size: {self.task_in.in_stream.input_buf_size}, auto_start: {self.task_in.in_stream.auto_start}, channels_to_read: {self.task_in.in_stream.channels_to_read}, avail_samp_per_chan: {self.task_in.in_stream.avail_samp_per_chan}, input_onbrd_buf_size: {self.task_in.in_stream.input_onbrd_buf_size}, offset: {self.task_in.in_stream.offset}")

            # attributes for the thread pool
            self._file_lock = threading.Lock()
            self.id = 0
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def __reinit_in_task(self):
        if self.device is None:
            return
        self.task_in.close()
        self.task_in = nidaqmx.Task()
        for ai_channel in self.analog_input_channels:
            self.task_in.ai_channels.add_ai_voltage_chan(f"/{self.device.name}/{ai_channel}")
        self.task_in.timing.cfg_samp_clk_timing(
            rate=self.sample_rate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.num_samples
        )
        self.task_in.triggers.reference_trigger.cfg_dig_edge_ref_trig(
            pretrigger_samples=self.num_samples-2,
            trigger_source=f"/{self.device.name}/{self.digital_trig_channel}",
            trigger_edge=nidaqmx.constants.Edge.RISING
        )
        self.task_in.in_stream.over_write = nidaqmx.constants.OverwriteMode.OVERWRITE_UNREAD_SAMPLES
        self.task_in.in_stream.relative_to = nidaqmx.constants.ReadRelativeTo.FIRST_PRETRIGGER_SAMPLE
        self.task_in.start()
    
    def __setup_tasks(self):
        if self.device is None:
            return
        self.__reinit_in_task()
        
        self.task_out.do_channels.add_do_chan(
            lines=f"/{self.device.name}/{self.digital_trig_channel}",
            line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE
        )
        
        self.task_out.start()
        
    def __save_as_csv(self, data, save_dir):
        if self.device is None:
            return
        with self._file_lock:
            if data is not None:
                header = ",".join(["time"] + self.analog_input_channels)
                save_path = os.path.join(save_dir, "measurement.csv")
                np.savetxt(save_path, data, delimiter=",", header=header, comments="")
                logger.info(f"Successfully saved acquired DAQ data and saved it to {save_path}")
            else:
                logger.warning("No data has been acquired to be stored.")
                with open(os.path.join(save_dir, "measurment_error_info.txt"), "w") as error_file:
                    print("Critical error occured. DAQ measurement process could NOT be started. Check if another program "
                        "is reserving/using DAQ resources.", file=error_file)
    
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
        data = None
        try:
            logger.debug("Starting acquisition...")
            # create trigger signal
            self.task_out.write([True, False])
            
            measurements = self.task_in.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            timestamps = np.array([x/self.sample_rate for x in range(len(measurements))])
            
            measurements = np.array(measurements)
            # extract every dimension from measurements (each is an input channel) and collect every data to save in list
            d_list = [timestamps[:, np.newaxis]]
            if len(measurements.shape) == 2:
                for dim in measurements:
                    d_list.append(dim[:, np.newaxis])
            else:
                d_list.append(measurements[:, np.newaxis])
            
            data = np.hstack(tuple(d_list))
            self.__save_as_csv(data, save_dir)
        except nidaqmx.errors.DaqError as e:
            logger.critical(e)
            self.delete_all_tasks()
        # prepare for next acquisition
        # in_task needs to be reinit to be able to retrigger
        self.__reinit_in_task()   
        
    
    def delete_all_tasks(self):
        if self.device is None:
            return
        try:
            self.task_in.close()
            self.task_out.close()
        except nidaqmx.errors.DaqError:
            logger.warning("Tasks seem to be closed already.")
