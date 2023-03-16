"""
utils

Author: Aaron Berk <aaronsberk@gmail.com>
Copyright © 2023, Aaron Berk, all rights reserved.
Created: 16 March 2023
"""
import os
from datetime import datetime


def get_tstamp():
    """Generate a timestamp as a string."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
