import logging
import os
import time
from typing import List, Optional, Tuple

import socketio

from src.config_utils import CONFIG
from src.voice_analysis.daq_interface import DAQ_Device
from src.voice_analysis.voice_field import Trigger, VoiceField

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VoiceRangeProfiler:
    def __init__(self,
                 sampling_rate: int = 44100,
                 sampling_time: float = 0.2,
                 trigger_timeout: float = 1.0,
                 rec_destination: str = f"recordings/{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}",
                 min_score: float = 0.7,
                 retrigger_percentage_improvement: float = 0.1,
                 semitone_bin_size: int = 2,
                 freq_bounds: Tuple[float, float] = (150.0, 1700.0),
                 dba_bin_size: int = 5,
                 dba_bounds: Tuple[int, int] = (35, 115),
                 socket: Optional[socketio.Client] = None,
                 from_config: bool = False) -> None:
        if from_config:
            logger.info("Loading trigger settings from config file...")
            super().__init__(CONFIG["buffer_size"], CONFIG["rate"], CONFIG["channels"], CONFIG["chunk_size"])
            self.min_score = CONFIG["min_score"]
            self.rec_destination = CONFIG["client_recordings_path"]
            self.trigger_timeout = CONFIG["trigger_timeout"]
            retrigger_percentage_improvement = CONFIG["min_trigger_improvement"]
            semitone_bin_size = CONFIG["semitone_bin_size"]
            freq_bounds = (CONFIG["frequency_bounds"]["lower"], CONFIG["frequency_bounds"]["upper"])
            dba_bin_size = CONFIG["decibel_bin_size"]
            dba_bounds = (CONFIG["decibel_bounds"]["lower"], CONFIG["decibel_bounds"]["upper"])
        else:
            self.sampling_rate = sampling_rate
            self.sampling_time = sampling_time
            self.min_score = min_score
            self.rec_destination = os.path.join(os.path.dirname(os.path.abspath(__file__)), rec_destination)
            self.trigger_timeout = trigger_timeout

        # check if trigger destination folder exists, else create
        self.__check_rec_destination()
        
        # create a websocket connection
        self.socket = socket
        if self.socket is not None:
            if self.socket.connected:
                logger.info("Websocket connection established.")
            else:
                logger.warning("SocketIO client without connection to server provided. Try to connect to default url "
                               "'http://localhost:5001'...")
                try:
                    self.socket.connect("http://localhost:5001")
                    logger.info("Websocket connection to default server successfully established.")
                except Exception:
                    logger.critical("Websocket connection to default server failed.")
        
        self.voice_field = VoiceField(
            semitone_bin_size=semitone_bin_size,
            freq_bounds=freq_bounds,
            db_bin_size=dba_bin_size,
            db_bounds=dba_bounds,
        )
        self.daq = None
        self.trigger = Trigger(self.voice_field, self.daq, self.rec_destination, self.min_score,
                               retrigger_percentage_improvement, socket)
        self.examination_is_running = False
        
        self.init_settings = {
            "sampling_rate": self.sampling_rate,
            "save_location": self.rec_destination,
            "min_score": self.min_score,
            "retrigger_percentage_improvement": retrigger_percentage_improvement,
            "freq_bounds": freq_bounds,
            "semitone_bin_size": semitone_bin_size,
            "dba_bounds": dba_bounds,
            "dba_bin_size": dba_bin_size
        }
        logger.info(f"Successfully created trigger: {self.init_settings}")
    
    @property
    def settings(self) -> dict:
        """Get the settings of the trigger instance.

        Returns
        -------
        dict
            The settings of the trigger instance.
        """
        return {**self.init_settings, "device": self.recording_device}
    
    def __check_rec_destination(self) -> None:
        """Check if the destination folder for the recorded audio files exists. If not, create it.
        """
        logger.info(f"Checking rec destination: {self.rec_destination}")
        if os.path.exists(self.rec_destination):
            return
        os.makedirs(self.rec_destination)
        
    def configure_daq_inputs(self, anlg_input_channels: List[str], cam_trig_channel: str, device_id: Optional[str]= None) -> None:
        self.daq = DAQ_Device(sampling_rate=self.sampling_rate, 
                              sampling_time=self.sampling_time, 
                              analog_input_channels=anlg_input_channels,
                              digital_trig_channel=cam_trig_channel,
                              device_id=device_id,
                              trigger=self.trigger,
                              voice_field=self.voice_field,
                              socket=self.socket)
    
    def start_examination(self):
        if self.daq is None:
            logger.error("DAQ device needs to be configured via configure_daq_inputs before examination can be performed.")
            raise ValueError("No DAQ device configured. Please be sure to call configure_daq_inputs before starting examination.")
        self.daq.start_cont_acquisition(self.socket, self.trigger, self.voice_field)
        self.examination_is_running = True
        
    def stop_examination(self):
        if self.daq is None:
            logger.error("No DAQ device configured.")
            raise ValueError("No DAQ device configured.")
        self.daq.stop_cont_acquisition()
        self.examination_is_running = False

