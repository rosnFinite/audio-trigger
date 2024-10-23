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
import parselmouth

from typing import List, Optional

from src.voice_analysis.postprocessing.scoring import calc_pitch_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DAQ_Device:
    def __init__(self,
                 sampling_rate: Optional[int] = 20000,
                 sampling_time: Optional[float] = None,
                 analog_input_channels: Optional[List[str]] = None,
                 digital_trig_channel: Optional[str] = None,
                 device_id: Optional[str] = None,
                 trigger = None,
                 voice_field = None,
                 socket= None):
        self.device = self.__select_daq(device_id)
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
        self.sampling_rate = sampling_rate
        self.sampling_time = sampling_time
        self.num_samples = int(sampling_rate * sampling_time)
        
        self.trigger = trigger
        self.voice_field = voice_field
        self.socket = socket
        
        self.data_buffer = None
        self.prev_trigger_t = None
        
        # TODO: Handle case when no ni-daqmx is installed
        try:
            self.cont_read_task = None
            self.cam_trig_task = nidaqmx.Task()
        except nidaqmx.errors.DaqNotFoundError:
            logger.warning("No NI-DAQmx installation found on this system. Continuing without...")
        # attributes for the thread pool
        self._file_lock = threading.Lock()
        self.id = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    def __setup_cont_read_task(self):
        logger.info("Starting continuous acquisition task.")
        self.cont_read_task = nidaqmx.Task()
        for ai_channel in self.analog_input_channels:
            self.cont_read_task.ai_channels.add_ai_voltage_chan(f"/{self.device.name}/{ai_channel}")
        
        self.cont_read_task.timing.cfg_samp_clk_timing(
            rate=self.sampling_rate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.num_samples
        )
    
    def __setup_tasks(self):
        # REWRITTEN
        for ai_channel in self.analog_input_channels:
            self.cont_read_task.ai_channels.add_ai_voltage_chan(f"/{self.device.name}/{ai_channel}")
        
        self.cont_read_task.timing.cfg_samp_clk_timing(
            rate=self.sampling_rate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.num_samples
        )
        
        self.cam_trig_task.do_channels.add_do_chan(
            lines=f"/{self.device.name}/{self.digital_trig_channel}",
            line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE
        )
        
    
    def start_cont_acquisition(self, socket, trigger, voice_field):
        self.__setup_cont_read_task()
        
        
        def cont_read_task_callback(task_idx, event_type, num_samples, callback_data=[socket, trigger, voice_field]):
            cb_start_t = time.time()
            if self.prev_trigger_t is not None:
                time_diff = time.time() - self.prev_trigger_t
                if time_diff < 1:
                    logger.debug(
                        f"Trigger callback processing temporarily disabled (Waiting on cam ready). Time diff: {time_diff} < 1")
                    return 0
                else:
                    # update status to running if timeout is over
                    if socket is not None:
                        socket.emit("status_update_complete",
                                    {"status": "running", "save_location": self.rec_destination})
                        self.prev_trigger_t = None
            self.data_buffer = self.cont_read_task.read(number_of_samples_per_channel=self.num_samples)
            if len(self.data_buffer) < 2:
                logger.critical("No microphone and/or sound level meter connect to the DAQ board. " 
                                "Be sure that channel with lowes id is reserved for audio and next higher id for dB data.")
                try:
                    self.cont_read_task.close()
                    self.cam_trig_task.close()
                except Exception:
                    logger.warning("Tried closing tasks, but they are already closed or never started.")
                raise ValueError("Missing audio and/or dB data input. Check if devices are connected to correct channels.")
            # audio data is always on channel with lowest ID and dB on next higher ID channel
            audio_data = self.data_buffer[0]
            db_data = self.data_buffer[1]
            
            sound = parselmouth.Sound(audio_data, sampling_frequency=self.sampling_rate)
            score, dom_freq = calc_pitch_score(sound=sound,
                                            freq_floor=self.voice_field.freq_min,
                                            freq_ceiling=self.sampling_rate // 2)
            dba_level = np.mean(db_data)
            is_trig = self.trigger.trigger(sound, dom_freq, dba_level, score,
                                        trigger_data={"audio": audio_data, "egg": None, "sampling_rate": self.sampling_rate}, cb_start_t=cb_start_t)
            if is_trig:
                self.prev_trigger_t = time.time()
                if self.socket is not None:
                # update status to waiting if trigger was detected
                    self.socket.emit("status_update_complete",
                                {"status": "waiting", "save_location": self.rec_destination})
            return 0

        self.cont_read_task.register_every_n_samples_acquired_into_buffer_event(self.num_samples, cont_read_task_callback)
        
        self.cont_read_task.start()
    
    def stop_cont_acquisition(self):
        logger.info("Stopping continous acquisition task.")
        self.cont_read_task.close()
    
    def __save_as_csv(self, data, save_dir):
        if self.device is None:
            return
        with self._file_lock:
            if data is not None:
                header = ",".join(["time"] + self.analog_input_channels)
                save_path = os.path.join(save_dir, "measurements.csv")
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
    
    def start_trig_acquisition(self) -> any:
        """Starts the data acquisition process with the provided settings and saves the acquired data to a file.
        Provided 'save_dir' must be a valid path to a directory where the data will be saved. Filename will be
        'measurements.csv'.
        Full path -> 'save_dir'/measurements.csv
        """
        if self.device is None:
            raise AttributeError("No DAQ device connected. Cannot start acquisition.")
        self.__setup_read_data_task()
        try:
            logger.debug("Starting acquisition on trigger...")
            # create trigger signal
            self.cam_trig_task.write([True, False])
            
            measurements = self.cont_read_task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            timestamps = np.array([x/self.sampling_rate for x in range(len(measurements))])
            
            # fit newly read data with data from continous read (data_buffer)
            data = []
            for i, channel_data in enumerate(self.data_buffer):
                data.append(channel_data.extend(measurements[i]))
            
            measurements = np.array(data)
            # extract every dimension from measurements (each is an input channel) and collect every data to save in list
            d_list = [timestamps[:, np.newaxis]]
            if len(measurements.shape) == 2:
                for dim in measurements:
                    d_list.append(dim[:, np.newaxis])
            else:
                d_list.append(measurements[:, np.newaxis])
            
            data = np.hstack(tuple(d_list))
            # self.save_as_csv(data, save_dir)
        except nidaqmx.errors.DaqError as e:
            logger.critical(e)
            self.delete_all_tasks()
        # prepare for next acquisition
        # in_task needs to be reinit to be able to retrigger
        self.__reinit_in_task()
        return data
        
    
    def delete_all_tasks(self):
        if self.device is None:
            return
        try:
            self.task_in.close()
            self.task_out.close()
        except nidaqmx.errors.DaqError:
            logger.warning("Tasks seem to be closed already.")

