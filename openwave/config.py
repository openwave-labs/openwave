"""Configuration settings for the OpenWave project.
This module handles loading and managing configuration options.
"""

from pathlib import Path
import configparser


config = configparser.ConfigParser()
config.read(Path(__file__).resolve().parent / "config.ini")

screen_width = config["screen"]["width"]
screen_height = config["screen"]["height"]
