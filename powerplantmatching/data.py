import logging
import os
import re

import numpy as np
import pandas as pd
import pandas_gbq as pd_gbq
import pycountry

from .cleaning import clean_name
from .core import get_config
from .utils import set_column_name

logger = logging.getLogger(__name__)
cget = pycountry.countries.get


def GEM(raw=False, update=True, config=None):
    # config = get_config() if config is None else config  # Don't ever use default config from PPM
    data_config = config["GEM"]

    target_admin1s = config["target_admin_1s"]

    df = pd_gbq.read_gbq(
        f"""SELECT unit_id, plant_id, unit_name, plant_name, admin_1, latitude, longitude,
    capacity, technology, primary_fuel, secondary_fuel, tertiary_fuel, quaternary_fuel, quinary_fuel,
    start_date, retired_date, planned_retire_date, operating_status, location_accuracy
    FROM {data_config['bq_table']}
    WHERE admin_1 IN ('{"','".join(target_admin1s)}')""",
        project_id=config["gcp_project"],
    )
    # df = pd.read_csv("japan/gem_unmatched.csv")

    # logger.warning("Using combined mappings file")
    # # remove matches in combined mappings
    # df_map = pd.read_csv("japan/combined_mappings.csv")
    # df = df[~df["plant_id"].isin(df_map["GEM"])]

    if raw:
        return df

    project_id = "plant_id" if data_config["aggregate_units"] else "unit_id"
    df = df.rename(
        columns={
            project_id: "projectID",
            "plant_name": "Name",
            "capacity": "Capacity",
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    df["Technology"] = df.technology.map(data_config["technology_map"])

    path = config["current_mappings"]
    if data_config["remove_matched_data"]:
        if os.path.exists(path):
            logger.info("Removing already matched data from GEM")
            current_mappings = pd.read_csv(path)
            df = df[~df["projectID"].isin(current_mappings.GEM)]
        else:
            logger.info(f"Current mappings do not exist: {path}")
        # logger.warning("Using combined mappings file")
        # combined_mappings = pd.read_csv("japan/matches/combined_mappings.csv")
        # # df = df[~df["projectID"].isin(combined_mappings.GEM)]
        # dfs = []
        # for i, gp in combined_mappings.groupby('match_index'):
        #     if gp.FIT.notna().any() and gp.GEM.notna().any():
        #         dfs.append(gp)
        # dfs = pd.concat(dfs)
        # df = df[~df["projectID"].isin(dfs.GEM)]

    # filter to target techs
    df = df[df.Technology.isin(config["target_technologies"])]

    # filter to operating
    df = df[df.operating_status.isin(config["target_gem_operating_status"])]

    if data_config["clean_name"]:
        logger.info("Cleaning GEM names")
        df = df.pipe(lambda x: clean_name(x, config))

    if data_config["aggregate_units"]:
        logger.info("Aggregating GEM units")
        df = (
            df.groupby(["projectID"])
            .agg(
                {
                    "Capacity": "sum",
                    "lat": pd.Series.mean,
                    "lon": pd.Series.mean,
                    "Name": pd.Series.mode,
                    "Technology": pd.Series.mode,
                    "admin_1": "first",
                }
            )
            .reset_index()
        )

    df = df[config["target_columns"]]
    df = df.pipe(set_column_name, "GEM")

    logger.info(f"GEM: {len(df)} projects")

    return df


def KITAMOTOGEM(raw=False, update=True, config=None):
    data_config = config["KITAMOTOGEM"]

    df = pd_gbq.read_gbq(
        f"""SELECT plant_name, plant_name_ja, business_name, asset_class, output_mw, admin_1,
    latitude, longitude, hash_id
    FROM {data_config['bq_table']}
    WHERE invalid_latlon = False""",
        project_id=config["gcp_project"],
    )

    # logger.warning("Using combined mappings file")
    # # remove matches in combined mappings
    # df_map = pd.read_csv("japan/combined_mappings.csv")
    # df = df[~df["hash_id"].isin(df_map["KITAMOTO"])]

    if raw:
        return df

    df = df.rename(
        columns={
            "plant_name": "Name",
            "business_name": "power_company",
            "output_mw": "Capacity",
            "latitude": "lat",
            "longitude": "lon",
            "hash_id": "projectID",
        }
    )

    df["Technology"] = df.asset_class.map(data_config["technology_map"])

    if data_config["remove_matched_data"]:
        logger.info("Removing already matched data from KITAMOTOGEM")
        path = config["current_mappings"]
        if os.path.exists(path):
            current_mappings = pd.read_csv(path)
            df = df[~df["projectID"].isin(current_mappings.KITAMOTO)]
        else:
            logger.info(f"Current mappings do not exist: {path}")

    if data_config["aggregate_units"]:
        logger.info("Aggregating KITAMOTOGEM units")
        df = (
            df.groupby(["Name", "power_company", "Technology"])
            .agg(
                {
                    "Capacity": "sum",
                    "lat": pd.Series.mean,
                    "lon": pd.Series.mean,
                    "admin_1": pd.Series.mode,
                    "projectID": set,
                }
            )
            .reset_index()
        )

    # filter to target techs
    df = df[df.Technology.isin(config["target_technologies"])]
    # filter to target admin_1s
    df = df[df.admin_1.isin(config["target_admin_1s"])]

    if data_config["clean_name"]:
        logger.info("Cleaning KITAMOTOGEM names")
        df = df.pipe(lambda x: clean_name(x, config))

    df = df[config["target_columns"]]
    df = df.pipe(set_column_name, "KITAMOTOGEM")

    logger.info(f"KITAMOTOGEM: {len(df)} projects")

    return df


def HJKS(raw=False, update=True, config=None):
    data_config = config["HJKS"]

    df = pd_gbq.read_gbq(
        f"""SELECT plant_code, plant_name, unit_name, power_company, asset_class,
    capacity_mw, grid_region
    FROM {data_config['bq_table']}""",
        project_id=config["gcp_project"],
    )

    if raw:
        return df

    # plant code is not a unique ID
    df["unit_code"] = df.plant_code + "<SEP>" + df.unit_name
    project_id = "plant_code" if data_config["aggregate_units"] else "unit_code"

    df = df.rename(
        columns={
            project_id: "projectID",
            "plant_name": "Name",
            "capacity_mw": "Capacity",
            "power_company": "power_company",
            "grid_region": "grid_region",
        }
    )

    df["asset_class"] = df["asset_class"].fillna("unknown")
    df["Technology"] = df.asset_class.map(data_config["technology_map"])

    if data_config["remove_matched_data"]:
        path = config["current_mappings"]
        if os.path.exists(path):
            logger.info("Removing already matched data from HJKS")
            current_mappings = pd.read_csv(path)
            df = df[~df["projectID"].isin(current_mappings.HJKS)]
        else:
            logger.info(f"Current mappings do not exist: {path}")

    if data_config["aggregate_units"]:
        logger.info("Aggregating HJKS units")
        df["Name"] = df["Name"].apply(
            lambda x: re.split(r"\d", x)[0].strip()
            if isinstance(x, str)
            else [re.split(r"\d", x)[0].strip() for x in x]
        )

        df = (
            df.groupby("projectID")
            .agg(
                {
                    "Capacity": "sum",
                    "unit_name": list,
                    "Name": pd.Series.mode,
                    "power_company": pd.Series.mode,
                    "grid_region": pd.Series.mode,
                    "Technology": "first",
                }
            )
            .reset_index()
        )
        df["Name"] = df["Name"].apply(
            lambda x: "; ".join(x) if not isinstance(x, str) else x
        )
        # df = (df.groupby('Name')
        #         .agg({'Capacity': 'sum', 'power_company': pd.Series.mode, 'grid_region': pd.Series.mode, 'Technology': 'first', 'projectID': set})
        #         .reset_index())

    # filter to target techs
    df = df[df.Technology.isin(config["target_technologies"])]
    # filter to target admin_1s
    df = df[df.grid_region.isin(config["target_grid_regions"])]

    if data_config["clean_name"]:
        logger.info("Cleaning HJKS names")
        df = df.pipe(lambda x: clean_name(x, config))

    df = df[config["target_columns"]]
    df = df.pipe(set_column_name, "HJKS")
    logger.info(f"HJKS: {len(df)} projects")
    return df


def FIT(raw=False, update=False, config=None):
    data_config = config["FIT"]

    df = pd_gbq.read_gbq(
        f"""SELECT facility_id, power_company, asset_class, power_output_mw, admin_1,
    latitude, longitude, location_type, operation_start_report_date, is_replacement, procurement_period_end_date
    FROM {data_config['bq_table']}""",
        project_id=config["gcp_project"],
    )

    if raw:
        return df

    df = df.rename(
        columns={
            "facility_id": "projectID",
            "power_company": "Name",
            "power_output_mw": "Capacity",
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    df["Technology"] = df.asset_class.map(data_config["technology_map"])

    # filter to target techs
    df = df[df.Technology.isin(config["target_technologies"])]
    # filter to target admin_1s
    # if config["filter_to_target_admin_1s"]:
    df = df[df.admin_1.isin(config["target_admin_1s"])]

    if config["only_operating_fit_projects"]:
        df = df[df.operation_start_report_date.notna()]

    if data_config["remove_matched_data"]:
        path = config["current_mappings"]
        if os.path.exists(path):
            logger.info("Removing already matched data from FIT")
            current_mappings = pd.read_csv(path)
            df = df[~df["projectID"].isin(current_mappings.FIT)]
        else:
            logger.info(f"Current mappings do not exist: {path}")

    if data_config["clean_name"]:
        logger.info("Cleaning FIT names")
        df = df.pipe(lambda x: clean_name(x, config))

    if data_config["aggregate_units"]:
        logger.info("Aggregating FIT units")
        df = (
            df.groupby(["Name", "lat", "lon", "Technology", "admin_1"])
            .agg({"Capacity": "sum", "projectID": set})
            .reset_index()
        )

    df = df[config["target_columns"]]
    df = df.pipe(set_column_name, "FIT")

    logger.info(f"FIT: {len(df)} projects")
    return df


############ I'm trying to demonstrate matching all 4 sources at once to test the matching code


def KITAMOTO_ALL(raw=False, update=False, config=None):
    data_config = config["KITAMOTO"]
    df = pd_gbq.read_gbq(
        f"""SELECT plant_name, business_name, asset_class, output_mw,
    latitude, longitude, admin_1, grid_region, hash_id, plant_name_ja, business_name_ja
    FROM {config['KITAMOTO']['bq_table']}
    WHERE invalid_latlon = False""",
        project_id=config["gcp_project"],
    )

    df = df.rename(
        columns={
            "plant_name": "Name",
            "business_name": "owner",
            "output_mw": "Capacity",
            "latitude": "lat",
            "longitude": "lon",
            "hash_id": "projectID",
        }
    )

    df["Technology"] = df.asset_class.map(data_config["technology_map"])

    # filter to target techs
    df = df[df.Technology.isin(config["target_technologies"])]
    # filter to target admin_1s
    df = df[df.admin_1.isin(config["target_admin_1s"])]

    if data_config["aggregate_units"]:
        logger.info("Aggregating KITAMOTO units")
        df = (
            df.groupby(["Name", "owner", "Technology"])
            .agg(
                {
                    "Capacity": "sum",
                    "projectID": set,
                    "lat": pd.Series.mean,
                    "lon": pd.Series.mean,
                    "admin_1": pd.Series.mode,
                    "grid_region": pd.Series.mode,
                }
            )
            .reset_index()
        )

    if data_config["clean_name"]:
        logger.info("Cleaning KITAMOTO names")
        df = df.pipe(lambda x: clean_name(x, config))

    df = df.pipe(set_column_name, "KITAMOTO")

    for col in config["target_columns"]:
        if col not in df.columns:
            df[col] = None
    return df


def FIT_ALL(raw=False, update=False, config=None):
    data_config = config["FIT"]
    df = pd_gbq.read_gbq(
        f"""SELECT facility_id, power_company, asset_class, power_output_mw, admin_1,
        latitude, longitude, location_type, facility_location, operation_start_report_date, is_replacement, procurement_period_end_date, power_company_ja, grid_region
        FROM {config['FIT']['bq_table']}
        WHERE asset_class IN ('biomass','hydropower','geothermal')""",
        project_id=config["gcp_project"],
    )
    df = df.rename(
        columns={
            "facility_id": "projectID",
            "power_company": "owner",
            "power_output_mw": "Capacity",
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    df["Technology"] = df.asset_class.map(data_config["technology_map"])
    # filter to target techs
    df = df[df.Technology.isin(config["target_technologies"])]
    # filter to target admin_1s
    df = df[df.admin_1.isin(config["target_admin_1s"])]

    if config["only_operating_fit_projects"]:
        df = df[df.operation_start_report_date.notna()]

    if data_config["aggregate_units"]:
        logger.info("Aggregating FIT units")
        df = (
            df.groupby(["owner", "lat", "lon", "Technology", "admin_1"])
            .agg({"Capacity": "sum", "projectID": set})
            .reset_index()
        )

    df = df.pipe(set_column_name, "FIT")

    for col in config["target_columns"]:
        if col not in df.columns:
            df[col] = None

    logger.info(f"FIT: {len(df)} projects")
    return df


def HJKS_ALL(raw=False, update=False, config=None):
    data_config = config["HJKS"]
    df = pd_gbq.read_gbq(
        f"""SELECT plant_code, plant_name, unit_name, power_company, asset_class, capacity_mw, grid_region,
        start_date, retire_date, plant_name_ja
        FROM {config['HJKS']['bq_table']}""",
        project_id=config["gcp_project"],
    )

    df["unit_code"] = df.plant_code + "<SEP>" + df.unit_name
    project_id = "plant_code" if data_config["aggregate_units"] else "unit_code"

    df = df.rename(
        columns={
            project_id: "projectID",
            "plant_name": "Name",
            "capacity_mw": "Capacity",
            "power_company": "power_company",
            "grid_region": "grid_region",
        }
    )

    df["asset_class"] = df["asset_class"].fillna("unknown")
    df["Technology"] = df.asset_class.map(data_config["technology_map"])

    if data_config["aggregate_units"]:
        logger.info("Aggregating HJKS units")
        df["Name"] = df["Name"].apply(
            lambda x: re.split(r"\d", x)[0].strip()
            if isinstance(x, str)
            else [re.split(r"\d", x)[0].strip() for x in x]
        )

        df = (
            df.groupby("projectID")
            .agg(
                {
                    "Capacity": "sum",
                    "unit_name": list,
                    "Name": pd.Series.mode,
                    "power_company": pd.Series.mode,
                    "grid_region": pd.Series.mode,
                    "Technology": "first",
                }
            )
            .reset_index()
        )
        df["Name"] = df["Name"].apply(
            lambda x: "; ".join(x) if not isinstance(x, str) else x
        )

    # filter to target techs
    df = df[df.Technology.isin(config["target_technologies"])]
    # filter to target admin_1s
    df = df[df.grid_region.isin(config["target_grid_regions"])]

    if data_config["clean_name"]:
        logger.info("Cleaning HJKS names")
        df = df.pipe(lambda x: clean_name(x, config))

    df = df.pipe(set_column_name, "HJKS")

    for col in config["target_columns"]:
        if col not in df.columns:
            df[col] = None
    logger.info(f"HJKS: {len(df)} projects")
    return df


def GEM_ALL(raw=False, update=True, config=None):
    data_config = config["GEM"]
    df = pd_gbq.read_gbq(
        f"""SELECT plant_id, unit_name, plant_name, admin_1, latitude, longitude,
        capacity, technology, unit_id, primary_fuel, secondary_fuel, tertiary_fuel, quaternary_fuel, quinary_fuel,
        start_date, retired_date, planned_retire_date, operating_status, operating_status_detail, location_accuracy, river, has_ccs, is_captive, captive_use, coal_source,
        FROM {config['GEM']['bq_table']}
        WHERE admin_0 = 'JPN'""",
        project_id=config["gcp_project"],
    )

    project_id = "plant_code" if data_config["aggregate_units"] else "unit_code"

    df = df.rename(
        columns={
            project_id: "projectID",
            "plant_name": "Name",
            "capacity": "Capacity",
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    df["Technology"] = df.technology.map(data_config["technology_map"])

    # filter to target techs
    df = df[df.Technology.isin(config["target_technologies"])]

    # filter to operating
    df = df[df.operating_status.isin(config["target_gem_operating_status"])]

    if data_config["clean_name"]:
        logger.info("Cleaning GEM names")
        df = df.pipe(lambda x: clean_name(x, config))

    if data_config["aggregate_units"]:
        logger.info("Aggregating GEM units")
        df = (
            df.groupby(["projectID"])
            .agg(
                {
                    "Capacity": "sum",
                    "lat": pd.Series.mean,
                    "lon": pd.Series.mean,
                    "Name": pd.Series.mode,
                    "Technology": pd.Series.mode,
                    "admin_1": "first",
                }
            )
            .reset_index()
        )

    df = df.pipe(set_column_name, "GEM")

    for col in config["target_columns"]:
        if col not in df.columns:
            df[col] = None

    logger.info(f"GEM: {len(df)} projects")

    return df
