import os
import sys

import pandas as pd
import yaml

import powerplantmatching as ppm

path = os.getcwd() + "/powerplantmatching/asean_test/"
with open(
    path + "config.yaml",
    "r",
) as f:
    config = yaml.safe_load(f)


ppm.powerplants(config=config, from_url=False, update=True).to_csv(
    path + "test_powerplants.csv"
)