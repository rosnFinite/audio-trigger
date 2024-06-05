import logging
import traceback
from typing import Optional

import socketio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def emit_exception(socket: socketio.Client,
                   exception_level: str,
                   exception: Optional[callable(BaseException)] = None,
                   exception_title: Optional[str] = None,
                   exception_description: Optional[str] = None) -> None:
    if socket is None:
        logger.info("No socket connection available. Cannot emit exception.")
        return
    if exception is None and (exception_title is None or exception_description is None):
        raise ValueError("Either an exception or an exception title and description must be provided.")
    # both of the following expression will primarily use optional argument or the traceback module to get the exception
    # title and description
    title = exception_title if exception_title is not None else traceback.format_exception_only(type(exception), exception)[0].strip()
    description = exception_description if exception_description is not None else traceback.format_tb(exception.__traceback__)[0]
    socket.emit("client_error", {"type": exception_level, "title": title, "message": description})
    logger.exception(f"Exception wit exception_level: {exception_level} occurred, {title} at {description}")