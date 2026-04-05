import json
import logging
import logging.config
from pathlib import Path
import atexit

logger = logging.getLogger("cheque_ocr")

_queue_handler = None


def setup_logging():
    global _queue_handler
    config_file = Path(__file__).parent / "logger_config.json"
    with open(config_file) as f_in:
        config = json.load(f_in)

    logging.config.dictConfig(config)
    _queue_handler = logging.getHandlerByName("queue_handler")
    if _queue_handler is not None:
        _queue_handler.listener.start()
        atexit.register(_shutdown_logging)


def _shutdown_logging():
    global _queue_handler
    if _queue_handler is None:
        return
    handler = _queue_handler
    _queue_handler = None

    if hasattr(handler, "listener") and handler.listener is not None:
        handler.listener.stop()

    for h in handler.handlers:
        h.close()
