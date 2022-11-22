"""Util file for rest of code"""
from pathlib import Path


def get_proj_root():
    ## attention: this assumes that utils is in a particular folder
    return Path(__file__).parent.parent.parent
