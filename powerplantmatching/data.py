import logging
import os
import re
from distutils.log import debug
import json

import numpy as np
import pandas as pd
import pandas_gbq as pd_gbq
import pycountry
import requests
from deprecation import deprecated

from .cleaning import (
    clean_name,
)
from .core import _data_in, _package_data, get_config
from .utils import (
    config_filter,
    correct_manually,
    fill_geoposition,
    get_raw_file,
    set_column_name,
)

logger = logging.getLogger(__name__)
cget = pycountry.countries.get


def KITAMOTO(raw=False, update=False, config=None):
    """for matching with FIT data"""
    config = get_config() if config is None else config
    data_config = config['KITAMOTO']

    df = pd_gbq.read_gbq(f"""SELECT plant_name, business_name, asset_class, output_mw, latitude, longitude, admin_1
    FROM {data_config['bq_table']}
    WHERE invalid_latlon = False""", project_id=config['gcp_project'])

    if raw:
        return df

    df = df.rename(columns={'business_name':'Name', 'output_mw':'Capacity', 'latitude':'lat', 'longitude':'lon'})

    df['Technology'] = df.asset_class.map(data_config['technology_map'])

    # TODO: create project_ID
    df.sort_values(['Technology', 'Name', 'Capacity', 'lat'], inplace=True)
    df = df.reset_index(drop=True).reset_index().rename(columns={'index':'projectID'})
    df['projectID'] = df['projectID'].apply(lambda x: f"KITAMOTO_{x}")

    # filter to target techs
    df = df[df.Technology.isin(config['target_technologies'])]
    # filter to target admin_1s
    df = df[df.admin_1.isin(config['target_admin_1s'])]

    # if config["clean_name_before_aggregation"]:
    #     df = df.pipe(clean_name)

    if config["remove_matched_data"]:
        logger.info("Removing already matched data from KITAMOTO")
        current_mappings = pd.read_csv(config["current_mappings"])
        df = df[~df["projectID"].isin(current_mappings.KITAMOTO)]

    df = df[config["target_columns"]]
    df = df.pipe(set_column_name, "KITAMOTO")

    logger.info(f"KITAMOTO: {len(df)} projects")

    return df


def FIT(raw=False, update=False, config=None):
    config = get_config() if config is None else config
    data_config = config['FIT']

    df = pd_gbq.read_gbq(f"""SELECT facility_id, power_company, asset_class, power_output_mw, admin_1,
    latitude, longitude
    FROM {data_config['bq_table']}""", project_id=config['gcp_project'])

    if raw:
        return df

    df = df.rename(columns={'facility_id':'projectID',
                            'power_company': 'Name',
                            'power_output_mw': 'Capacity',
                            'latitude':'lat',
                            'longitude':'lon'})

    df['Technology'] = df.asset_class.map(data_config['technology_map'])

    # filter to target techs
    df = df[df.Technology.isin(config['target_technologies'])]
    # filter to target admin_1s
    df = df[df.admin_1.isin(config['target_admin_1s'])]


    # if config["clean_name_before_aggregation"]:
    #     df = df.pipe(clean_name)

    if config["remove_matched_data"]:
        logger.info("Removing already matched data from FIT")
        current_mappings = pd.read_csv(config["current_mappings"])
        df = df[~df["projectID"].isin(current_mappings.FIT)]

    df = df[config["target_columns"]]
    df = df.pipe(set_column_name, "FIT")

    logger.info(f"FIT: {len(df)} projects")
    return df


def HJKS(raw=False, update=False, config=None):
    config = get_config() if config is None else config
    data_config = config['FIT']

    df = pd_gbq.read_gbq(f"""SELECT plant_code, plant_name, unit_name, power_company, asset_class, capacity_mw, grid_region
    FROM {data_config['bq_table']}""", project_id=config['gcp_project'])

    if raw:
        return df

    df = df.rename(columns={'plant_code': 'projectID',
                            'plant_name': 'Name',
                            'capacity_mw': 'Capacity',
                            'power_company': 'power_company',
                            'grid_region': 'grid_region'})
    
    df['Technology'] = df.asset_class.map(data_config['technology_map'])

    # filter to target techs
    df = df[df.Technology.isin(config['target_technologies'])]
    # filter to target admin_1s
    df = df[df.gridregion.isin(config['target_grid_regions'])]
