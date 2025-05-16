# -*- coding: utf-8 -*-
# file: logger.py
# date: 2025-05-16


import logging
from typing import Optional


# Configure the logger
logging.basicConfig(
    #level=logging.DEBUG,  # Set the minimum logging level
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log messages to a file
        logging.StreamHandler()         # Log messages to the console
    ]
)


def get_logger(name: Optional[str]):
    if name is None:
        name == __name__
    return logging.getLogger(name)
