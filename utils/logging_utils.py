import logging
import logging.handlers
import multiprocessing
import sys


def setup_logging(log_file):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_queue = multiprocessing.Queue()

    queue_handler = logging.handlers.QueueHandler(log_queue)
    file_handler = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter(log_fmt)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_fmt))

    listener = logging.handlers.QueueListener(log_queue, file_handler, console_handler)
    listener.start()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)

    return listener